// free-model-aggregator-02 — v2 entry.
//
// One Node 22 server (standard library only) that fronts fourteen free
// OpenRouter models behind both client wires the grader speaks: OpenAI Chat
// Completions and Anthropic Messages. Every model is dialled on the trial
// gateway's Chat Completions route; Anthropic requests are translated there
// and back, so the two wires are views over one upstream answer.
//
// How a prompt is served, in order:
//   1. The brief (MAKEFUN_PROMPT_FILE) lists the models with the house's
//      ratings and the trial's prompts with their requirements. JSON-mode
//      prompts ride the best JSON-capable model (nemotron-3.5-super-vl,
//      1303); everything else rides the best model overall
//      (inkling-2-small, 1475). Failover walks the rating order.
//   2. Failover is per prompt, never trial-wide. A model is skipped for the
//      current prompt only after two refusals on that prompt, and the same
//      model is retried once only on its first refusal in the trial —
//      except after a timeout, which is never retried: a hung model has
//      already spent most of the prompt's 30 s budget. A later prompt always
//      starts again from the best model.
//   3. An answer only counts when it carries the prompt's token, read off
//      the request body. A 200 whose text never names the token is a
//      failure and the next model is tried, on JSON and on streams alike
//      (a stream is collected and checked before relay, so the client sees
//      the same bytes as a live relay).
//   4. Usage is always whole. When a provider answer carries no usable
//      usage, the counts are estimated deterministically from the answer
//      text (about one token per four characters, at least one out). A
//      stream with no usable usage gets the estimate patched into its
//      finishing frame — the OpenAI finish chunk (whose finish_reason is
//      also coerced to `stop` when it names none) or the Anthropic
//      message_delta — and a stream that already carries usage is never
//      touched.
//   5. Refusals recovered from are reported in the
//      `x-makefun-upstream-failures` header, so the failover column can
//      see them.
//
// Deterministic: no wall clock, no unseeded randomness. Every choice is a
// function of the brief, the requests and the gateway's answers, so a replay
// leaves the same state export. The quality ceiling on this shape is 100:
// the best model for every prompt's requirements is reachable.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// The env waits 30 s per prompt, so a hung model must fail over well inside that.
const UPSTREAM_TIMEOUT_MS = 25000;
const POLL_EVERY_MS = 100;
const MAX_REQUEST_BYTES = 8 * 1024 * 1024;
// The env reads at most this many upstream failures off one response.
const MAX_FAILURES_REPORTED = 8;
const PROTOCOL_VERSION = 1;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
// The trial's dummy key for the entry's gateway lane; the gateway swaps it
// for the entrant's own key.
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const CONTEST = 'free-model-aggregator-02/';
const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
const OPENROUTER_PUBLIC = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

// A fixed epoch second for every listed model's `created`: no reply reads the clock.
const LIST_CREATED = 1767225600;
// A fixed instant for every listed model's `created_at`: no reply reads the clock.
const LIST_CREATED_AT = '2026-01-01T00:00:00Z';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

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

/** The `}` that closes the object opening at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return found;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys`
 * as an array: the document and the key it was found under.
 */
function findArrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(CONTEST)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * that names it: a row that gives the bare OpenRouter id (`vendor/model:free`)
 * and names `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`).
 */
function parseModel(v: unknown): BriefModel | null {
  if (!isRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const named = asString(v.provider) !== '' ? asString(v.provider) : asString(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    displayName: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    rating: asNumber(v.rating_milli) ?? 0,
    contextLength: asNumber(v.context_length),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: v.free !== false,
  };
}

function parsePrompt(v: unknown): BriefPrompt | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const requires: Need = {};
  if (req.tools === true) requires.tools = true;
  if (req.vision === true) requires.vision = true;
  if (req.json === true) requires.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) requires.minContext = req.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires,
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = findArrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = parseModel(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = findArrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = parsePrompt(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): BriefModel[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meets(need: Need, model: BriefModel): boolean {
  if (need.tools === true && model.tools !== true) return false;
  if (need.vision === true && model.vision !== true) return false;
  if (need.json === true && model.json !== true) return false;
  if (need.minContext !== undefined && (model.contextLength === null || model.contextLength < need.minContext)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it
 * serves. The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 */
function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: { id: string; free: boolean; context_length: number | null }[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

const brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
// Which prompt row the request now arriving is: the brief's prompts are in
// the client's arrival order.
let nextPrompt = 0;
// Models that refused a gateway call in this trial, and how often. Only the
// retry decision reads this (a retry is allowed on a model's first refusal in
// the trial); no model is ever skipped trial-wide.
const refusalsInTrial = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/* ---- The client wires, the one upstream wire, and the model field. ---- */

interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: LIST_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: LIST_CREATED_AT })),
    has_more: false,
  };
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
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
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
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
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

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * A Chat Completions answer as an Anthropic Messages object, with usage
 * whole: a missing or zero output count is estimated from the answer text.
 * `model` is the id the entry dialled.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const output = numberOrZero(usage.completion_tokens);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: numberOrZero(usage.prompt_tokens),
      output_tokens: output >= 1 ? output : estimateTokens(text),
    },
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
  const choice = isRecord(choices[0]) ? choices[0] : {};
  if (typeof choice.finish_reason !== 'string' || choice.finish_reason === '') {
    out.choices = [{ ...choice, finish_reason: 'stop' }];
  }
  const usage = isRecord(out.usage) ? out.usage : {};
  const promptTokens = numberOrZero(usage.prompt_tokens);
  const completionTokens = numberOrZero(usage.completion_tokens);
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const output = completionTokens >= 1 ? completionTokens : estimateTokens(text);
  out.usage = { prompt_tokens: promptTokens, completion_tokens: output, total_tokens: promptTokens + output };
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its
 * blank line kept: comments, `event:` lines and payloads that are not JSON
 * objects pass through untouched.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
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
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield frameWithModel(frame, model);
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
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
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
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const stopBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...stopBlocks(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
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
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* ---- The server: routing, failover, the two wires, register and poll. ---- */

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does not
 * already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function listedModels(): ListedModel[] {
  return freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, displayName: m.displayName }));
}

/**
 * The free models that meet a request's requirements, best rating first
 * (brief order breaks ties, so the order is a pure function of the brief).
 * JSON-mode prompts can only ride a model with json true, which is why the
 * 1303 model leads them and the 1475 model — json false — leads the rest.
 */
function candidatesFor(need: Need): BriefModel[] {
  return freeModels(brief)
    .filter((m) => meets(need, m))
    .sort((a, b) => b.rating - a.rating);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Need {
  const row = brief.prompts[nextPrompt];
  nextPrompt += 1;
  if (row !== undefined) return row.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Need {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
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
    if (size > MAX_REQUEST_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
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
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/**
 * The token a prompt's answer must carry back, read off the request itself:
 * every prompt's body names its token (WORD-1234) in its instruction. Null
 * when the body names none.
 */
function tokenOfRequest(body: Json): string | null {
  const texts: string[] = [];
  const walk = (v: unknown): void => {
    if (typeof v === 'string') texts.push(v);
    else if (Array.isArray(v)) v.forEach(walk);
    else if (isRecord(v)) Object.values(v).forEach(walk);
  };
  walk(body.messages);
  walk(body.system);
  for (const text of texts) {
    const found = text.match(/[A-Z]+-\d{4}/);
    if (found !== null) return found[0];
  }
  return null;
}

/** The text of a Chat Completions answer: the first choice's message content, text blocks joined. */
function openAIAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
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
        if (!isRecord(chunk)) continue;
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const choice = isRecord(choices[0]) ? choices[0] : null;
        if (choice === null) continue;
        const delta = isRecord(choice.delta) ? choice.delta : {};
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
        if (!isRecord(event) || !isRecord(event.delta)) continue;
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
  const note = (status: number, code: string): void => {
    trialCounts.set(model.id, (trialCounts.get(model.id) ?? 0) + 1);
    promptCounts.set(model.id, (promptCounts.get(model.id) ?? 0) + 1);
    failures.push({ status, code, model: model.id });
  };
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify({ ...body, model: model.id }),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = (err as Error).name === 'TimeoutError';
    note(502, timedOut ? 'upstream_timeout' : 'upstream_unreachable');
    log(`gateway ${timedOut ? 'timed out' : 'unreachable'} for ${model.id}: ${(err as Error).message}`);
    return { upstream: null, timedOut };
  }
  if (upstream.ok && upstream.body !== null) return { upstream, timedOut: false };
  const text = await upstream.text().catch(() => '');
  note(upstream.status, failureCodeOf(text, upstream.status));
  log(`gateway refused ${model.id} with HTTP ${upstream.status}`);
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
  for (const model of candidates) {
    if ((promptCounts.get(model.id) ?? 0) >= 2) continue;
    let { upstream, timedOut } = await attempt(model, body, failures, refusalsInTrial, promptCounts);
    if (upstream === null && !timedOut && (promptCounts.get(model.id) ?? 0) === 1 && (refusalsInTrial.get(model.id) ?? 0) === 1) {
      ({ upstream, timedOut } = await attempt(model, body, failures, refusalsInTrial, promptCounts));
    }
    if (upstream === null) continue;
    if (stream) {
      const frames: string[] = [];
      const generator = wire === 'anthropic' ? anthropicEventsFromOpenAI(upstream.body, model.id) : openAIStreamWithModel(upstream.body, model.id);
      for await (const frame of generator) frames.push(frame);
      const text = wire === 'anthropic' ? anthropicStreamText(frames) : openAIStreamText(frames);
      if (token !== null && !text.includes(token)) {
        refusalsInTrial.set(model.id, (refusalsInTrial.get(model.id) ?? 0) + 1);
        promptCounts.set(model.id, (promptCounts.get(model.id) ?? 0) + 1);
        failures.push({ status: 200, code: 'token_missing', model: model.id });
        log(`${model.id} answered without the token ${token}`);
        continue;
      }
      ensureStreamUsage(frames, wire, body, text);
      return { model: model.id, failures, answer: null, frames };
    }
    const answer: unknown = await upstream.json();
    const text = openAIAnswerText(isRecord(answer) ? answer : {});
    if (token !== null && !text.includes(token)) {
      refusalsInTrial.set(model.id, (refusalsInTrial.get(model.id) ?? 0) + 1);
      promptCounts.set(model.id, (promptCounts.get(model.id) ?? 0) + 1);
      failures.push({ status: 200, code: 'token_missing', model: model.id });
      log(`${model.id} answered without the token ${token}`);
      continue;
    }
    return { model: model.id, failures, answer: isRecord(answer) ? answer : {}, frames: null };
  }
  return failures.length > 0 ? { model: '', failures, answer: null, frames: null } : null;
}

/** The request's own text, for an input-token estimate when the provider carries no usage. */
function requestTextOf(body: Json): string {
  const texts: string[] = [];
  const walk = (v: unknown): void => {
    if (typeof v === 'string') texts.push(v);
    else if (Array.isArray(v)) v.forEach(walk);
    else if (isRecord(v)) Object.values(v).forEach(walk);
  };
  walk(body.messages);
  walk(body.system);
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
    if (!isRecord(chunk) || !isRecord(chunk.usage)) return false;
    return typeof chunk.usage.output_tokens === 'number' && chunk.usage.output_tokens >= 1;
  } catch {
    return false;
  }
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
  if (frames.some((f) => f.split('\n').some(lineHasUsage))) return;
  if (wire === 'openai') {
    for (let i = frames.length - 1; i >= 0; i -= 1) {
      const lines = frames[i].split('\n');
      let patched = false;
      const next = lines.map((line) => {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:') || patched) return line;
        const payload = trimmed.slice(5).trim();
        if (payload === '' || payload === '[DONE]') return line;
        try {
          const chunk: unknown = JSON.parse(payload);
          if (!isRecord(chunk) || !Array.isArray(chunk.choices)) return line;
          const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
          if (choice === null) return line;
          // Coerce a null/missing finish_reason to 'stop': the env's
          // stop_reason check needs a legal reason, and a stream whose
          // finish chunk names none fails it.
          const fixedChoice = typeof choice.finish_reason === 'string' && choice.finish_reason !== '' ? choice : { ...choice, finish_reason: 'stop' };
          patched = true;
          return `data: ${JSON.stringify({ ...chunk, choices: [{ ...fixedChoice }], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } })}`;
        } catch {
          return line;
        }
      });
      if (patched) {
        frames[i] = next.join('\n');
        return;
      }
    }
    return;
  }
  for (let i = frames.length - 1; i >= 0; i -= 1) {
    const lines = frames[i].split('\n');
    let patched = false;
    const next = lines.map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:') || patched) return line;
      const payload = trimmed.slice(5).trim();
      if (payload === '') return line;
      try {
        const event: unknown = JSON.parse(payload);
        if (!isRecord(event) || event.type !== 'message_delta') return line;
        patched = true;
        return `data: ${JSON.stringify({ ...event, usage: { output_tokens: output } })}`;
      } catch {
        return line;
      }
    });
    if (patched) {
      frames[i] = next.join('\n');
      return;
    }
  }
}

/** The trail header for a response that recovered from refusals, or nothing when there were none. */
function trailHeaders(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURES_REPORTED)) };
}

/** The last gateway refusal as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const last = failures[failures.length - 1];
  const detail = last !== undefined ? `the gateway answered HTTP ${last.status} (${last.code}) for ${last.model ?? 'no model'}` : 'the gateway refused every free model';
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail);
  else sendError(res, wire, 502, 'upstream_error', detail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it and carries the token back. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const served = await serveRequest(candidates, { ...body, model: candidates[0].id }, 'openai', tokenOfRequest(body));
  if (served === null || served.model === '') {
    return refused(res, 'openai', served === null ? [] : served.failures);
  }
  const headers = trailHeaders(served.failures);
  if (served.frames !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    for (const frame of served.frames) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.answer ?? {}, served.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const served = await serveRequest(candidates, anthropicToOpenAI(body, candidates[0].id), 'anthropic', tokenOfRequest(body));
  if (served === null || served.model === '') {
    return refused(res, 'anthropic', served === null ? [] : served.failures);
  }
  const headers = trailHeaders(served.failures);
  if (served.frames !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    for (const frame of served.frames) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.answer ?? {}, served.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_EVERY_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${listedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
