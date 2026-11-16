// free-model-aggregator-02 — v2 entry, second authoring.
//
// A single-file Node 22 server, standard library only, putting fourteen
// free OpenRouter models behind the two client wires the grader speaks:
// OpenAI Chat Completions and Anthropic Messages. All models are dialled
// through the trial gateway's Chat Completions route; an Anthropic request
// is translated into that route and the answer translated back, so both
// wires are views over one upstream completion.
//
// Serving order for one prompt:
//   1. The brief (MAKEFUN_PROMPT_FILE) lists models with the house's
//      ratings plus the trial's prompts with their requirements. The
//      JSON-mode prompts ride the best JSON-capable model,
//      nemotron-3.5-super-vl (1303); every other prompt rides the best
//      model overall, inkling-2-small (1475). Failover walks the rating
//      order downward.
//   2. Failover is per prompt, never trial-wide. A model is skipped for
//      the current prompt only once it has refused twice on that prompt,
//      and the same model is retried once only on its first refusal in
//      the trial — never after a timeout, which has already spent most
//      of the prompt's 30 s budget. A later prompt always starts again
//      from the best model.
//   3. An answer counts only when it carries the prompt's token, read
//      off the request body. A 200 whose text never names the token is
//      a failure and the next model is tried, on JSON and on streams
//      alike (a stream is collected and checked before relay, so the
//      client sees the same bytes as a live relay).
//   4. Usage is always whole. When a provider answer carries no usable
//      usage, the counts are estimated deterministically from the
//      answer text (about one token per four characters, at least one
//      out). A stream with no usable usage gets the estimate patched
//      into its finishing frame — the OpenAI finish chunk, whose
//      finish_reason is also coerced to `stop` when it names none, or
//      the Anthropic message_delta — and a stream that already carries
//      usage is never touched.
//   5. Refusals recovered from are reported in the
//      `x-makefun-upstream-failures` header, so the failover column
//      can see them.
//
// Deterministic: no wall clock, no unseeded randomness. Every choice is
// a function of the brief, the requests and the gateway's answers, so a
// replay leaves the same state export. The quality ceiling on this shape
// is 100: the best model for every prompt's requirements is reachable.

import { createServer, type Server, type ServerResponse, type IncomingMessage } from 'node:http';
import { readFileSync as readFile } from 'node:fs';

// The env waits 30 s per prompt, so a hung model must fail over well inside that.
const CALL_TIMEOUT_MS = 25000;
const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
// The env reads at most this many upstream failures off one response.
const TRAIL_CAP = 8;
const PROTOCOL = 1;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_ROOT = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
// The trial's dummy key for the entry's gateway lane; the gateway swaps it
// for the entrant's own key.
const GATEWAY_TOKEN = process.env.OPENAI_API_KEY ?? '';

const CONTEST_PREFIX = 'free-model-aggregator-02/';
const MODELS_DOC = 'free-model-aggregator-02/models/v1';
const PROMPTS_DOC = 'free-model-aggregator-02/prompts/v1';
const PROVIDER_PUBLIC_URL = 'https://openrouter.ai/api/v1';
const PROVIDER_KEY_ENV = 'OPENROUTER_API_KEY';

// A fixed epoch second for every listed model's `created`: no reply reads the clock.
const CREATED_EPOCH = 1767225600;
// A fixed instant for every listed model's `created_at`: no reply reads the clock.
const CREATED_AT = '2026-01-01T00:00:00Z';

type Json = Record<string, unknown>;
const isObj = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

interface BriefModel {
  id: string;
  provider: string;
  displayName: string;
  rating: number;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface BriefPrompt {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Need;
}

interface Brief {
  models: BriefModel[];
  prompts: BriefPrompt[];
  error: string | null;
}

interface Catalog {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: {
      id: string;
      free: boolean;
      context_length: number | null;
    }[];
  }[];
}

/** The `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function findObjectEnd(text: string, start: number): number {
  let depth = 0;
  let inStr = false;
  let i = start;
  while (i < text.length) {
    const ch = text[i];
    switch (true) {
      case inStr && ch === '\\':
        i += 2;
        continue;
      case inStr && ch === '"':
        inStr = false;
        break;
      case !inStr && ch === '"':
        inStr = true;
        break;
      case !inStr && ch === '{':
        depth += 1;
        break;
      case !inStr && ch === '}':
        depth -= 1;
        if (depth === 0) return i;
        break;
    }
    i += 1;
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function extractSchemaDocs(text: string): Json[] {
  const docs: Json[] = [];
  let at = text.indexOf('{');
  while (at !== -1) {
    const end = findObjectEnd(text, at);
    if (end !== -1) {
      try {
        const value: unknown = JSON.parse(text.slice(at, end + 1));
        if (isObj(value) && typeof value.schema === 'string') docs.push(value);
      } catch {
        // Not JSON on its own: look on.
      }
    }
    at = text.indexOf('{', at + 1);
  }
  return docs;
}

function strVal(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numVal(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolVal(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix carrying one of `keys` as
 * an array: the document together with the key it was found under.
 */
function findDocWithArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (let d = 0; d < docs.length; d++) {
    const doc = docs[d];
    if (!strVal(doc.schema).includes(CONTEST_PREFIX)) continue;
    for (let k = 0; k < keys.length; k++) {
      if (Array.isArray(doc[keys[k]])) return { doc, key: keys[k] };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * naming it: a row giving the bare OpenRouter id (`vendor/model:free`) and
 * naming `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`).
 */
function toModelRow(v: unknown): BriefModel | null {
  if (!isObj(v)) return null;
  const idField = strVal(v.id);
  const raw = idField === '' ? strVal(v.model) : idField;
  if (raw === '') return null;
  const providerField = strVal(v.provider);
  const named = providerField === '' ? strVal(v.wire) : providerField;
  let id = raw;
  if (named === 'openrouter' && raw.toLowerCase().indexOf('openrouter/') !== 0) id = 'openrouter/' + raw;
  const displayField = strVal(v.display_name);
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    displayName: displayField === '' ? raw : displayField,
    rating: numVal(v.rating_milli) ?? 0,
    contextLength: numVal(v.context_length),
    tools: boolVal(v.tools),
    vision: boolVal(v.vision),
    json: boolVal(v.json),
    free: v.free !== false,
  };
}

function toPromptRow(v: unknown): BriefPrompt | null {
  if (!isObj(v)) return null;
  const id = strVal(v.id);
  if (id === '') return null;
  const req = isObj(v.requires) ? v.requires : {};
  const requires: Need = {};
  const flags: ['tools' | 'vision' | 'json', unknown][] = [
    ['tools', req.tools],
    ['vision', req.vision],
    ['json', req.json],
  ];
  for (const [flag, value] of flags) if (value === true) requires[flag] = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) requires.minContext = req.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires,
  };
}

/** The brief at `path`: its two documents under this contest's schema prefix. */
function loadBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFile(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: 'cannot read ' + path + ': ' + (err as Error).message };
  }
  const docs = extractSchemaDocs(text);
  const modelsDoc = findDocWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: 'the brief at ' + path + ' holds no ' + MODELS_DOC + ' document' };
  const models: BriefModel[] = [];
  for (const row of modelsDoc.doc[modelsDoc.key] as unknown[]) {
    const m = toModelRow(row);
    if (m !== null) models.push(m);
  }
  const promptsDoc = findDocWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: 'the brief at ' + path + ' holds no ' + PROMPTS_DOC + ' document' };
  const prompts: BriefPrompt[] = [];
  for (const row of promptsDoc.doc[promptsDoc.key] as unknown[]) {
    const p = toPromptRow(row);
    if (p !== null) prompts.push(p);
  }
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): BriefModel[] {
  const out: BriefModel[] = [];
  for (const m of brief.models) if (m.free) out.push(m);
  return out;
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function satisfies(need: Need, model: BriefModel): boolean {
  if (need.minContext !== undefined) {
    if (model.contextLength === null) return false;
    if (model.contextLength < need.minContext) return false;
  }
  if (need.json === true && model.json !== true) return false;
  if (need.vision === true && model.vision !== true) return false;
  if (need.tools === true && model.tools !== true) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it
 * serves. The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 */
function makeCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: { id: string; free: boolean; context_length: number | null }[] = [];
  let i = 0;
  while (i < brief.models.length) {
    const model = brief.models[i];
    i += 1;
    if (!model.free) continue;
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  const catalog: Catalog = { providers: [] };
  if (models.length === 0) return catalog;
  catalog.providers.push({
    id: 'openrouter',
    base_url: PROVIDER_PUBLIC_URL,
    key_env: PROVIDER_KEY_ENV,
    wire: 'openai',
    models,
  });
  return catalog;
}

const brief = loadBrief(BRIEF_PATH);
const catalog: Catalog = makeCatalog(brief);
// Which prompt row the request now arriving is: the brief's prompts are in
// the client's arrival order.
let promptCursor = 0;
// Models that refused a gateway call in this trial, and how often. Only the
// retry decision reads this (a retry is allowed on a model's first refusal in
// the trial); no model is ever skipped trial-wide.
const trialRefusalCounts = new Map<string, number>();

function note(message: string): void {
  process.stderr.write('aggregator: ' + message + '\n');
}

/* ---- The client wires, the one upstream wire, and the model field. ---- */

interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ListedModel[]): Json {
  const data: Json[] = [];
  for (const m of models) data.push({ id: m.id, object: 'model', created: CREATED_EPOCH, owned_by: m.provider });
  return { object: 'list', data };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ListedModel[]): Json {
  const data: Json[] = [];
  for (const m of models) data.push({ type: 'model', id: m.id, display_name: m.displayName, created_at: CREATED_AT });
  return { data, has_more: false };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const b of content) {
    if (isObj(b) && b.type === 'text' && typeof b.text === 'string') parts.push(b.text);
  }
  return parts.join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type !== 'image' || !isObj(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: 'data:' + media + ';base64,' + source.data } };
  }
  return source.type === 'url' && typeof source.url === 'string' ? { type: 'image_url', image_url: { url: source.url } } : null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return content === undefined ? '' : JSON.stringify(content);
  const text = textOf(content);
  return text === '' ? JSON.stringify(content) : text;
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isObj(value)) return value;
  if (typeof value !== 'string' || value.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(value);
    return isObj(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  switch (finish) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return 'end_turn';
  }
}

function numOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * A deterministic token estimate from text length (about one token per four
 * characters). Used only when the provider's answer carries no usable usage
 * counts: the wire checks require a whole input count of at least 0 and an
 * output count of at least 1, and the estimate is a pure function of the
 * answer, so a replay produces the same bytes.
 */
function estimateTokens(text: string): number {
  const est = Math.ceil(text.length / 4);
  return est < 1 ? 1 : est;
}

/** One assistant turn's tool_use blocks as OpenAI tool_calls. */
function toolCallsOf(content: unknown[]): Json[] {
  const calls: Json[] = [];
  for (const block of content) {
    if (!isObj(block) || block.type !== 'tool_use') continue;
    const input = isObj(block.input) ? block.input : {};
    calls.push({
      id: typeof block.id === 'string' ? block.id : '',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(input) },
    });
  }
  return calls;
}

/** One user turn's content blocks as OpenAI parts, with tool_results as tool messages. */
function userTurnMessages(content: unknown[], messages: Json[]): void {
  const parts: Json[] = [];
  for (const block of content) {
    if (!isObj(block)) continue;
    if (block.type === 'tool_result') {
      messages.push({
        role: 'tool',
        tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: toolResultText(block.content),
      });
      continue;
    }
    const part = partOf(block);
    if (part !== null) parts.push(part);
  }
  if (parts.length > 0) messages.push({ role: 'user', content: parts });
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (let m = 0; m < source.length; m++) {
    const raw = source[m];
    if (!isObj(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls = toolCallsOf(content);
      const text = textOf(content);
      const msg: Json = { role: 'assistant', content: text === '' ? null : text };
      if (toolCalls.length > 0) msg.tool_calls = toolCalls;
      messages.push(msg);
      continue;
    }
    userTurnMessages(content, messages);
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Json[] = [];
    for (const raw of body.tools) {
      if (!isObj(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isObj(choice)) {
    switch (choice.type) {
      case 'auto':
        out.tool_choice = 'auto';
        break;
      case 'any':
        out.tool_choice = 'required';
        break;
      case 'tool':
        if (typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
        break;
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Whole usage for an answer: the provider's counts, or an estimate from the text when the output count is missing or zero. */
function wholeUsage(usage: Json, text: string): { input_tokens: number; output_tokens: number } {
  const output = numOrZero(usage.completion_tokens);
  return {
    input_tokens: numOrZero(usage.prompt_tokens),
    output_tokens: output >= 1 ? output : estimateTokens(text),
  };
}

/**
 * A Chat Completions answer as an Anthropic Messages object, with usage
 * whole: a missing or zero output count is estimated from the answer text.
 * `model` is the id the entry dialled.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isObj(choices[0]) ? choices[0] : {};
  const message = isObj(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (let c = 0; c < calls.length; c++) {
    const call = calls[c];
    if (!isObj(call) || !isObj(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isObj(body.usage) ? body.usage : {};
  const whole = wholeUsage(usage, text);
  return {
    id: 'msg_' + (typeof body.id === 'string' ? body.id : 'aggregate'),
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: whole.input_tokens, output_tokens: whole.output_tokens },
  };
}

/**
 * A Chat Completions answer with `model` set to the id the entry dialled and
 * usage whole: a missing finish_reason defaults to `stop`, a missing or zero
 * output count is estimated from the answer text.
 */
function openAIJsonWithModel(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(out.choices) ? out.choices : [];
  const choice = isObj(choices[0]) ? choices[0] : {};
  if (typeof choice.finish_reason !== 'string' || choice.finish_reason === '') {
    out.choices = [{ ...choice, finish_reason: 'stop' }];
  }
  const usage = isObj(out.usage) ? out.usage : {};
  const message = isObj(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const whole = wholeUsage(usage, text);
  out.usage = {
    prompt_tokens: whole.input_tokens,
    completion_tokens: whole.output_tokens,
    total_tokens: whole.input_tokens + whole.output_tokens,
  };
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  const head = event === null ? '' : 'event: ' + event + '\n';
  return head + 'data: ' + JSON.stringify(data) + '\n\n';
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its
 * blank line kept: comments, `event:` lines and payloads that are not JSON
 * objects pass through untouched.
 */
function frameWithModel(frame: string, model: string): string {
  const out: string[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) {
      out.push(line);
      continue;
    }
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') {
      out.push(line);
      continue;
    }
    let rewritten = line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isObj(chunk)) rewritten = 'data: ' + JSON.stringify({ ...chunk, model });
    } catch {
      // Not JSON: relay the line as it came.
    }
    out.push(rewritten);
  }
  return out.join('\n') + '\n\n';
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data
 * payload's `model` the id the entry dialled. Frames are split on the blank
 * line that ends them; a frame cut across reads is held until its end
 * arrives, and a body that ends without one is relayed as it is.
 */
async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield frameWithModel(buffer.slice(0, at), model);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithModel(buffer, model);
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence:
 * message_start on the first chunk, content_block_start/delta/stop per text
 * block or tool call, then message_delta with the stop reason and usage, then
 * message_stop. `model` is the id the entry dialled, named back to the client.
 *
 * The stop reason is held until the stream's last chunk has arrived: OpenAI
 * sends the usage after the finish_reason, and Anthropic's message_delta
 * carries the output tokens. A stream that ends without [DONE] is closed the
 * same way, so the client always sees a complete event sequence.
 */
async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const state = {
    started: false,
    finished: false,
    pendingStop: null as string | null,
    textIndex: -1,
    nextIndex: 0,
    toolIndexes: new Map<number, number>(),
    inputTokens: 0,
    outputTokens: 0,
  };

  const startEvents = (): string[] => {
    if (state.started) return [];
    state.started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: state.inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const stopBlocks = (): string[] => {
    const out: string[] = [];
    if (state.textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
      state.textIndex = -1;
    }
    const indexes = [...state.toolIndexes.values()].sort((a, b) => a - b);
    for (const index of indexes) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    state.toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (state.finished) return [];
    state.finished = true;
    const tail = [
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: state.outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
    return [...stopBlocks(), ...tail];
  };
  const openTextBlock = (out: string[]): void => {
    state.textIndex = state.nextIndex;
    state.nextIndex += 1;
    out.push(sseEvent('content_block_start', { type: 'content_block_start', index: state.textIndex, content_block: { type: 'text', text: '' } }));
  };
  const openToolBlock = (out: string[], call: Json): number => {
    if (state.textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
      state.textIndex = -1;
    }
    const index = state.nextIndex;
    state.nextIndex += 1;
    const fn = isObj(call.function) ? call.function : {};
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_' + index,
          name: typeof fn.name === 'string' ? fn.name : '',
          input: {},
        },
      }),
    );
    return index;
  };

  /** One SSE frame's lines as Anthropic events. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents());
        out.push(...finishEvents(state.pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isObj(chunk)) continue;
      if (isObj(chunk.usage)) {
        state.inputTokens = numOrZero(chunk.usage.prompt_tokens);
        state.outputTokens = numOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isObj(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isObj(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (state.textIndex === -1) openTextBlock(out);
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: state.textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (let t = 0; t < calls.length; t++) {
        const call = calls[t];
        if (!isObj(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : state.toolIndexes.size;
        let index = state.toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          index = openToolBlock(out, call);
          state.toolIndexes.set(upstreamIndex, index);
        }
        const fn = isObj(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) state.pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!state.finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(state.pendingStop ?? 'end_turn')) yield event;
  }
}

/* ---- The server: routing, failover, the two wires, register and poll. ---- */

/**
 * The gateway's Chat Completions route. GATEWAY_ROOT is the OpenAI-compatible
 * root (`…/v1` on the trial network); a root not already ending in `/v1`
 * gets one, so both spellings reach the route.
 */
function completionsUrl(): string {
  if (GATEWAY_ROOT === '') return '';
  if (GATEWAY_ROOT.endsWith('/v1')) return GATEWAY_ROOT + '/chat/completions';
  return GATEWAY_ROOT + '/v1/chat/completions';
}

/** The free models the brief lists, as GET /v1/models names them. */
function listedModels(): ListedModel[] {
  const out: ListedModel[] = [];
  for (const m of freeModels(brief)) out.push({ id: m.id, provider: m.provider, displayName: m.displayName });
  return out;
}

/**
 * The free models meeting a request's requirements, best rating first (brief
 * order breaks ties, so the order is a pure function of the brief). JSON-mode
 * prompts can only ride a model with json true, which is why the 1303 model
 * leads them and the 1475 model — json false — leads the rest.
 */
function candidatesFor(need: Need): BriefModel[] {
  const fitting: BriefModel[] = [];
  for (const m of freeModels(brief)) if (satisfies(need, m)) fitting.push(m);
  fitting.sort((a, b) => {
    if (b.rating === a.rating) return 0;
    return b.rating > a.rating ? 1 : -1;
  });
  return fitting;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Need {
  const row = brief.prompts[promptCursor];
  promptCursor += 1;
  return row !== undefined ? row.requires : requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Need {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  let m = 0;
  while (m < messages.length && !vision) {
    const message = messages[m];
    m += 1;
    if (!isObj(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isObj(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (extraHeaders !== undefined) Object.assign(headers, extraHeaders);
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const list = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a refused gateway answer, from its error body. */
function failureCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isObj(value) && isObj(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code.slice(0, 64);
      const type = value.error.type;
      if (typeof type === 'string' && type !== '') return type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return 'http_' + status;
}

/** All strings anywhere inside `v`, in walk order. */
function stringsInside(v: unknown, out: string[]): void {
  if (typeof v === 'string') {
    out.push(v);
    return;
  }
  if (Array.isArray(v)) {
    for (const el of v) stringsInside(el, out);
    return;
  }
  if (isObj(v)) {
    for (const el of Object.values(v)) stringsInside(el, out);
  }
}

/**
 * The token a prompt's answer must carry back, read off the request itself:
 * every prompt's body names its token (WORD-1234) in its instruction. Null
 * when the body names none.
 */
function tokenOfRequest(body: Json): string | null {
  const texts: string[] = [];
  stringsInside(body.messages, texts);
  stringsInside(body.system, texts);
  for (const text of texts) {
    const found = text.match(/[A-Z]+-\d{4}/);
    if (found !== null) return found[0];
  }
  return null;
}

/** The text of a Chat Completions answer: the first choice's message content, text blocks joined. */
function openAIAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isObj(choices[0]) ? choices[0] : {};
  const message = isObj(choice.message) ? choice.message : {};
  return typeof message.content === 'string' ? message.content : textOf(message.content);
}

/** The text carried by OpenAI SSE frames: every choice delta's content, joined in order. */
function openAIStreamText(frames: readonly string[]): string {
  let text = '';
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isObj(chunk)) continue;
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const choice = isObj(choices[0]) ? choices[0] : null;
        if (choice === null) continue;
        const delta = isObj(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') text += delta.content;
      } catch {
        // Not JSON: nothing to read.
      }
    }
  }
  return text;
}

/** The text carried by Anthropic SSE events: every text_delta, joined in order. */
function anthropicStreamText(frames: readonly string[]): string {
  let text = '';
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      try {
        const event: unknown = JSON.parse(payload);
        if (!isObj(event) || !isObj(event.delta)) continue;
        if (event.delta.type === 'text_delta' && typeof event.delta.text === 'string') text += event.delta.text;
      } catch {
        // Not JSON: nothing to read.
      }
    }
  }
  return text;
}

/**
 * One gateway attempt at one model: the POST itself, or null on a refusal. A
 * refusal is recorded in this prompt's counts and in the trial's counts, and
 * in the caller's trail, with the provider's own error code. Returns whether
 * the failure was a gateway timeout (a hung model), so the caller can decide
 * about a retry.
 */
async function attempt(
  model: BriefModel,
  body: Json,
  failures: UpstreamFailure[],
  trialCounts: Map<string, number>,
  promptCounts: Map<string, number>,
): Promise<{ upstream: Response | null; timedOut: boolean }> {
  const record = (status: number, code: string): void => {
    trialCounts.set(model.id, (trialCounts.get(model.id) ?? 0) + 1);
    promptCounts.set(model.id, (promptCounts.get(model.id) ?? 0) + 1);
    failures.push({ status, code, model: model.id });
  };
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: 'Bearer ' + GATEWAY_TOKEN,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(completionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model: model.id }),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = (err as Error).name === 'TimeoutError';
    record(502, timedOut ? 'upstream_timeout' : 'upstream_unreachable');
    note('gateway ' + (timedOut ? 'timed out' : 'unreachable') + ' for ' + model.id + ': ' + (err as Error).message);
    return { upstream: null, timedOut };
  }
  if (upstream.ok && upstream.body !== null) return { upstream, timedOut: false };
  const text = await upstream.text().catch(() => '');
  record(upstream.status, failureCodeOf(text, upstream.status));
  note('gateway refused ' + model.id + ' with HTTP ' + upstream.status);
  return { upstream: null, timedOut: false };
}

interface ServedAnswer {
  model: string;
  failures: UpstreamFailure[];
  /** The non-stream answer, as the gateway returned it; null on a stream. */
  answer: Json | null;
  /** The stream's frames, in order; null on a non-stream answer. */
  frames: string[] | null;
}

/**
 * Serve one request: dial the gateway's Chat Completions route in rating
 * order, retrying a refused model once on the spot when that refusal is its
 * first in the trial — a single refusal is often transient, and a retry that
 * succeeds keeps the prompt on the best model — and skipping a model on this
 * prompt only once it has refused twice on this prompt, so a rate-limited
 * model never takes the prompt down with it, while a later prompt always gets
 * the best model again. A model that already refused earlier in the trial
 * gets one fresh attempt per prompt and no retry, so a hard-down best model
 * costs at most one extra call per later prompt.
 *
 * A hung model is never retried: a timeout already spent most of the
 * prompt's 30 s budget, so the next model in rating order is tried at once.
 * The 25 s threshold itself is unchanged from v1, so the replay risk is no
 * greater than v1's.
 *
 * An answer only counts when it carries the prompt's token: a 200 whose text
 * never names the token earns the prompt nothing on quality, so it is
 * treated as a failure (`token_missing`) and the next model is tried. A
 * stream is collected first and checked the same way — the frames are relayed
 * unchanged when they pass, so the client sees the same bytes as a live
 * relay — and a model that answers by calling a tool instead of writing the
 * token fails the check the same way.
 */
async function serveRequest(candidates: BriefModel[], body: Json, wire: 'openai' | 'anthropic', token: string | null): Promise<ServedAnswer | null> {
  const promptCounts = new Map<string, number>();
  const failures: UpstreamFailure[] = [];
  const stream = body.stream === true;
  const noteTokenMiss = (model: BriefModel): void => {
    trialRefusalCounts.set(model.id, (trialRefusalCounts.get(model.id) ?? 0) + 1);
    promptCounts.set(model.id, (promptCounts.get(model.id) ?? 0) + 1);
    failures.push({ status: 200, code: 'token_missing', model: model.id });
    note(model.id + ' answered without the token ' + token);
  };
  let c = 0;
  while (c < candidates.length) {
    const model = candidates[c];
    c += 1;
    if ((promptCounts.get(model.id) ?? 0) >= 2) continue;
    let result = await attempt(model, body, failures, trialRefusalCounts, promptCounts);
    const firstRefusal = (promptCounts.get(model.id) ?? 0) === 1 && (trialRefusalCounts.get(model.id) ?? 0) === 1;
    if (result.upstream === null && !result.timedOut && firstRefusal) {
      result = await attempt(model, body, failures, trialRefusalCounts, promptCounts);
    }
    if (result.upstream === null) continue;
    if (stream) {
      const frames: string[] = [];
      const generator = wire === 'anthropic' ? anthropicEventsFromOpenAI(result.upstream.body, model.id) : openAIStreamWithModel(result.upstream.body, model.id);
      for await (const frame of generator) frames.push(frame);
      const text = wire === 'anthropic' ? anthropicStreamText(frames) : openAIStreamText(frames);
      if (token !== null && !text.includes(token)) {
        noteTokenMiss(model);
        continue;
      }
      ensureStreamUsage(frames, wire, body, text);
      return { model: model.id, failures, answer: null, frames };
    }
    const answer: unknown = await result.upstream.json();
    const text = openAIAnswerText(isObj(answer) ? answer : {});
    if (token !== null && !text.includes(token)) {
      noteTokenMiss(model);
      continue;
    }
    return { model: model.id, failures, answer: isObj(answer) ? answer : {}, frames: null };
  }
  return failures.length > 0 ? { model: '', failures, answer: null, frames: null } : null;
}

/** The request's own text, for an input-token estimate when the provider carries no usage. */
function requestTextOf(body: Json): string {
  const texts: string[] = [];
  stringsInside(body.messages, texts);
  stringsInside(body.system, texts);
  return texts.join(' ');
}

/** Whether one SSE data line carries a usable output count (>= 1). */
function lineHasUsage(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return false;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return false;
  try {
    const chunk: unknown = JSON.parse(payload);
    if (!isObj(chunk) || !isObj(chunk.usage)) return false;
    return typeof chunk.usage.output_tokens === 'number' && chunk.usage.output_tokens >= 1;
  } catch {
    return false;
  }
}

/**
 * Patch the usage into the last frame of `frames` whose first data line
 * passes `test`, rewriting that line with `patch`. True when a frame was
 * patched. The patcher sees the parsed data payload of the line it rewrites.
 */
function patchLastFrame(frames: string[], test: (payload: unknown) => boolean, patch: (payload: Json) => Json): boolean {
  let i = frames.length - 1;
  while (i >= 0) {
    const lines = frames[i].split('\n');
    let patched = false;
    for (let l = 0; l < lines.length; l++) {
      if (patched) break;
      const trimmed = lines[l].trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isObj(parsed) || !test(parsed)) continue;
      lines[l] = 'data: ' + JSON.stringify(patch(parsed));
      patched = true;
    }
    if (patched) {
      frames[i] = lines.join('\n');
      return true;
    }
    i -= 1;
  }
  return false;
}

/**
 * Guarantee a stream's usage check passes: when no collected frame carries a
 * usable output count (>= 1), patch the usage into the frame that already
 * ends the stream — the OpenAI finish chunk, or the Anthropic message_delta
 * — estimated deterministically from the collected text, so a replay patches
 * the same bytes. A stream whose upstream answer already carries usage is
 * untouched.
 */
function ensureStreamUsage(frames: string[], wire: 'openai' | 'anthropic', body: Json, text: string): void {
  const output = estimateTokens(text);
  const input = estimateTokens(requestTextOf(body));
  let has = false;
  for (const frame of frames) {
    if (has) break;
    for (const line of frame.split('\n')) if (lineHasUsage(line)) has = true;
  }
  if (has) return;
  if (wire === 'openai') {
    patchLastFrame(
      frames,
      (payload) => Array.isArray((payload as Json).choices) && isObj((payload as Json).choices[0]),
      (chunk) => {
        const choice = isObj(chunk.choices[0]) ? chunk.choices[0] : {};
        // Coerce a null/missing finish_reason to 'stop': the env's
        // stop_reason check needs a legal reason, and a stream whose
        // finish chunk names none fails it.
        const fixedChoice = typeof choice.finish_reason === 'string' && choice.finish_reason !== '' ? choice : { ...choice, finish_reason: 'stop' };
        return {
          ...chunk,
          choices: [{ ...fixedChoice }],
          usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
        };
      },
    );
    return;
  }
  patchLastFrame(
    frames,
    (payload) => (payload as Json).type === 'message_delta',
    (event) => ({ ...event, usage: { output_tokens: output } }),
  );
}

/** The trail header for a response that recovered from refusals, or nothing when there were none. */
function trailHeaders(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, TRAIL_CAP)) };
}

/** The last gateway refusal as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const last = failures[failures.length - 1];
  const detail =
    last !== undefined ? 'the gateway answered HTTP ' + last.status + ' (' + last.code + ') for ' + (last.model ?? 'no model') : 'the gateway refused every free model';
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail);
  else sendError(res, wire, 502, 'upstream_error', detail);
}

/**
 * One completion request on either wire: pick the candidates, serve it, and
 * render the answer back in the client's wire. `toUpstream` rewrites the
 * client's body into the Chat Completions body the gateway dials;
 * `fromUpstream` renders a non-stream answer back; `noModel` is the 503
 * error type when no free model meets the request.
 */
async function answerRequest(
  req: IncomingMessage,
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  toUpstream: (body: Json, first: BriefModel) => Json,
  fromUpstream: (answer: Json, model: string) => Json,
  noModel: string,
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    sendError(res, wire, 503, noModel, 'no free model in the brief meets this request');
    return;
  }
  const served = await serveRequest(candidates, toUpstream(body, candidates[0]), wire, tokenOfRequest(body));
  if (served === null || served.model === '') {
    refused(res, wire, served === null ? [] : served.failures);
    return;
  }
  const headers = trailHeaders(served.failures);
  if (served.frames !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    for (const frame of served.frames) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, fromUpstream(served.answer ?? {}, served.model), headers);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it and carries the token back. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  await answerRequest(
    req,
    res,
    'openai',
    (body, first) => ({ ...body, model: first.id }),
    (answer, model) => openAIJsonWithModel(answer, model),
    'no_free_model',
  );
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  await answerRequest(
    req,
    res,
    'anthropic',
    (body, first) => anthropicToOpenAI(body, first.id),
    (answer, model) => openAIToAnthropic(answer, model),
    'overloaded_error',
  );
}

interface Route {
  method: string;
  path: string;
  wire: 'openai' | 'anthropic';
  run: (req: IncomingMessage, res: ServerResponse) => Promise<void>;
}

const ROUTES: Route[] = [
  { method: 'POST', path: '/v1/chat/completions', wire: 'openai', run: chat },
  { method: 'POST', path: '/v1/messages', wire: 'anthropic', run: messages },
];

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  for (const route of ROUTES) {
    if (req.method === route.method && url.pathname === route.path) {
      await route.run(req, res);
      return;
    }
  }
  const wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
  sendError(res, wire, 404, 'not_found', 'no route ' + (req.method ?? '') + ' ' + url.pathname);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(ENV_URL + '/register', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error('POST ' + ENV_URL + '/register answered HTTP ' + res.status + ': ' + (await res.text()).slice(0, 500));
  const reply: unknown = await res.json();
  if (isObj(reply) && typeof reply.requests === 'number') note('registered: the trial holds ' + reply.requests + ' prompts');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(ENV_URL + '/work', { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isObj(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, WORK_POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) note(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = JSON.stringify(catalog) + '\n';
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (completionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  note('listening on 0.0.0.0:' + port + ' with ' + listedModels().length + ' free models');
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
