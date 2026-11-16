// free-model-aggregator-02 — v3 aggregator.
//
// One long-running program: it listens on a port it chooses, registers with
// the env, serves the OpenAI and Anthropic completion wires, and answers
// every prompt with a real completion from one of the fourteen free models,
// dialled on the gateway's Chat Completions route.
//
// v3 changes over v2 (see DIAGNOSIS-V3 in the run record):
//   1. Death memory. A model is dead for the rest of the trial after one
//      timeout, or after it refused on two different prompts. A burst that
//      kills a model on one prompt does NOT kill it for the trial (that was
//      v1's bug); only cross-prompt evidence retires it. Death saves the
//      probe calls and, on a hang, the whole timeout on later prompts.
//   2. First-attempt timeout 15 s (was 25 s). A hung top model leaves room
//      for a slow fallback inside the env's 30 s cap: 15 s hang + 12 s
//      fallback = 27 s, served; under v2 it was 25 s + 12 s = 37 s, lost.
//   3. Walk cap. Six consecutive gateway refusals on one prompt means the
//      outage is systemic (an account limit, a provider outage), not a
//      model: stop walking, fail the prompt, and keep the account budget
//      for the prompts that follow. An outage prompt costs ~6 calls
//      instead of ~33, so it cannot trip the 20 req/min account cap and
//      cascade into the rest of the trial.
//   4. Tool calls the client did not declare are stripped from relayed
//      answers (wire insurance on the tools check).
// Kept from v2: per-prompt failover (walk the rating order, retry once on a
// model's first refusal in the trial), usage injection into the finish
// chunk / message_delta (chars/4), finish_reason coercion (null/empty or
// illegal -> "stop" / "end_turn"), token-missing failover, routing read
// from the brief at runtime, catalog exactly the fourteen :free models.
//
// Routing is a pure function of the brief and the gateway answers seen so
// far. No wall clock, no randomness: a replayed trial routes identically.

import http from 'node:http';
import fs from 'node:fs';

const FIRST_ATTEMPT_TIMEOUT_MS = 15000;
const MAX_CONSECUTIVE_REFUSALS = 6;
const MAX_HEADER_ROWS = 8;
const FIXED_CREATED = 1767225600;

// ---------------------------------------------------------------- brief ----

function balancedJson(text, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

function extractJsonBlob(text, marker) {
  const at = text.indexOf(marker);
  if (at < 0) throw new Error('brief is missing ' + marker);
  const raw = balancedJson(text, at);
  if (raw === null) throw new Error('brief blob is unbalanced at ' + marker);
  return JSON.parse(raw);
}

// The brief carries the sealed model list (with ratings and capability
// flags) and the prompt manifest. The catalog is built from the model
// list: exactly the fourteen :free models, once each, in brief order.
function loadBrief() {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) throw new Error('MAKEFUN_PROMPT_FILE is not set');
  const text = fs.readFileSync(file, 'utf8');
  const modelsDoc = extractJsonBlob(text, '{"schema":"free-model-aggregator-02/models/v1"');
  const rows = modelsDoc.models;
  if (!Array.isArray(rows) || rows.length === 0) throw new Error('brief carries no models');
  const catalog = {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: rows.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
  return { catalog, rows };
}

const { catalog, rows: MODEL_ROWS } = loadBrief();

// Routing order: rating descending, brief order breaks ties (stable sort).
// Kept as a pure function of the brief, computed once.
const BY_RATING = [...MODEL_ROWS].sort((a, b) => b.rating_milli - a.rating_milli);

function eligibleModels(requires) {
  return BY_RATING.filter((m) => {
    if (requires.json && !m.json) return false;
    if (requires.tools && !m.tools) return false;
    return true;
  });
}

function bareId(fullId) {
  return fullId.startsWith('openrouter/') ? fullId.slice('openrouter/'.length) : fullId;
}

// ------------------------------------------------------------ trial state ----

const dead = new Set(); // model full id -> retired for the rest of the trial
const refusalsByModel = new Map(); // model full id -> total refusal count
const refusedOnPrompts = new Map(); // model full id -> Set of prompt tokens

function noteRefusal(modelId, promptKey, timedOut) {
  refusalsByModel.set(modelId, (refusalsByModel.get(modelId) ?? 0) + 1);
  if (!refusedOnPrompts.has(modelId)) refusedOnPrompts.set(modelId, new Set());
  refusedOnPrompts.get(modelId).add(promptKey);
  if (timedOut) {
    dead.add(modelId);
    return;
  }
  if (refusedOnPrompts.get(modelId).size >= 2) dead.add(modelId);
}

// ------------------------------------------------------------ small utils ----

const TOKEN_RE = /[A-Z]+-\d{4}/;

function tokenOf(body) {
  const m = TOKEN_RE.exec(JSON.stringify(body));
  return m ? m[0] : null;
}

function estimateTokens(text) {
  return Math.max(0, Math.floor((text ?? '').length / 4));
}

const OPENAI_LEGAL_STOP = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
const ANTHROPIC_LEGAL_STOP = new Set(['end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'pause_turn']);

function coerceOpenAIStop(reason) {
  return typeof reason === 'string' && OPENAI_LEGAL_STOP.has(reason) ? reason : 'stop';
}

function openAIStopToAnthropic(reason) {
  switch (reason) {
    case 'length': return 'max_tokens';
    case 'tool_calls': return 'tool_use';
    case 'content_filter': return 'refusal';
    default: return 'end_turn';
  }
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      if (!raw) { resolve({}); return; }
      try { resolve(JSON.parse(raw)); } catch { reject(new Error('request body is not JSON')); }
    });
    req.on('error', reject);
  });
}

function sendJson(res, status, value, extraHeaders = {}) {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', ...extraHeaders });
  res.end(body);
}

// ------------------------------------------------------- gateway dialling ----

const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

async function dialChatCompletions(chatBody, stream, modelId) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FIRST_ATTEMPT_TIMEOUT_MS);
  try {
    const res = await fetch(GATEWAY_BASE + '/chat/completions', {
      method: 'POST',
      headers: {
        authorization: 'Bearer ' + GATEWAY_KEY,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ ...chatBody, model: modelId, stream }),
      signal: controller.signal,
    });
    const text = await res.text();
    return { status: res.status, text };
  } catch (err) {
    if (err && err.name === 'AbortError') return { status: 0, text: '', timedOut: true };
    return { status: 0, text: '', networkError: true };
  } finally {
    clearTimeout(timer);
  }
}

// Parse one gateway answer (JSON or SSE) into a uniform shape. Returns
// null when the body is not a usable answer at all.
function parseGatewayAnswer(status, text, stream) {
  if (status < 200 || status >= 300) return null;
  if (stream) {
    let content = '';
    let usage = null;
    let finishReason = null;
    let toolCalls = [];
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk;
      try { chunk = JSON.parse(payload); } catch { continue; }
      const choice = chunk?.choices?.[0];
      if (!choice) continue;
      if (typeof choice.delta?.content === 'string') content += choice.delta.content;
      if (Array.isArray(choice.delta?.tool_calls)) toolCalls = toolCalls.concat(choice.delta.tool_calls);
      if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') finishReason = choice.finish_reason;
      if (chunk.usage && typeof chunk.usage === 'object') {
        usage = {
          prompt_tokens: Number(chunk.usage.prompt_tokens ?? 0) || 0,
          completion_tokens: Number(chunk.usage.completion_tokens ?? 0) || 0,
        };
      }
    }
    return { content, usage, finishReason, toolCalls };
  }
  let value;
  try { value = JSON.parse(text); } catch { return null; }
  const choice = value?.choices?.[0];
  if (!choice || typeof choice.message !== 'object' || choice.message === null) return null;
  const message = choice.message;
  const content = typeof message.content === 'string' ? message.content
    : Array.isArray(message.content)
      ? message.content.filter((b) => b?.type === 'text').map((b) => b.text ?? '').join('')
      : '';
  const usage = value.usage && typeof value.usage === 'object'
    ? {
        prompt_tokens: Number(value.usage.prompt_tokens ?? 0) || 0,
        completion_tokens: Number(value.usage.completion_tokens ?? 0) || 0,
      }
    : null;
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  return { content, usage, finishReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : null, toolCalls };
}

function failureCode(status, text, timedOut, networkError) {
  if (timedOut) return 'timeout';
  if (networkError) return 'network_error';
  try {
    const value = JSON.parse(text);
    const code = value?.error?.code ?? value?.error?.message ?? value?.code;
    if (typeof code === 'string' && code !== '') return code.slice(0, 120);
  } catch { /* fall through */ }
  return 'http_' + status;
}

// Keep only tool calls the client declared. An answer that names a tool
// the client never declared fails the wire's tools check outright.
function keepDeclaredCalls(toolCalls, declaredNames) {
  const allowed = new Set(declaredNames);
  return (toolCalls ?? []).filter((call) => {
    const name = call?.function?.name;
    return typeof name === 'string' && allowed.has(name);
  });
}

// ------------------------------------------------------------- wire shapes ----

function openAIAnswer(modelId, content, toolCalls, stopReason, usage) {
  return {
    id: 'chatcmpl-entry',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: modelId,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: stopReason,
      },
    ],
    usage: {
      prompt_tokens: usage.prompt_tokens,
      completion_tokens: usage.completion_tokens,
      total_tokens: usage.prompt_tokens + usage.completion_tokens,
    },
  };
}

function anthropicAnswer(modelId, content, toolBlocks, stopReason, usage) {
  const blocks = [{ type: 'text', text: content }];
  for (const block of toolBlocks) blocks.push(block);
  return {
    type: 'message',
    id: 'msg-entry',
    model: modelId,
    role: 'assistant',
    content: blocks,
    stop_reason: stopReason,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

function openAIStreamFrames(modelId, content, stopReason, usage) {
  const half = Math.ceil(content.length / 2);
  const chunk = (delta, finish) => ({
    id: 'chatcmpl-entry',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [
    `data: ${JSON.stringify(chunk({ role: 'assistant', content: content.slice(0, half) }, null))}\n\n`,
    `data: ${JSON.stringify(chunk({ content: content.slice(half) }, null))}\n\n`,
    `data: ${JSON.stringify({
      ...chunk({}, stopReason),
      usage: { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens, total_tokens: usage.prompt_tokens + usage.completion_tokens },
    })}\n\n`,
    'data: [DONE]\n\n',
  ];
  return frames.join('');
}

function anthropicStreamFrames(modelId, content, stopReason, usage) {
  const half = Math.ceil(content.length / 2);
  const event = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  const parts = [
    event('message_start', {
      type: 'message_start',
      message: {
        id: 'msg-entry',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
    event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: content.slice(0, half) } }),
    event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: content.slice(half) } }),
    event('content_block_stop', { type: 'content_block_stop', index: 0 }),
    event('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: usage.completion_tokens },
    }),
    event('message_stop', { type: 'message_stop' }),
  ];
  return parts.join('');
}

// ------------------------------------------------- anthropic <-> openai wire ----

function anthropicToChat(body) {
  const messages = [];
  if (typeof body.system === 'string' && body.system !== '') {
    messages.push({ role: 'system', content: body.system });
  } else if (Array.isArray(body.system)) {
    const text = body.system.filter((b) => b?.type === 'text').map((b) => b.text ?? '').join('');
    if (text !== '') messages.push({ role: 'system', content: text });
  }
  for (const m of Array.isArray(body.messages) ? body.messages : []) {
    const role = m?.role === 'assistant' ? 'assistant' : 'user';
    if (typeof m?.content === 'string') {
      messages.push({ role, content: m.content });
      continue;
    }
    if (!Array.isArray(m?.content)) continue;
    const text = m.content.filter((b) => b?.type === 'text').map((b) => b.text ?? '').join('');
    const toolUses = m.content.filter((b) => b?.type === 'tool_use');
    const toolResults = m.content.filter((b) => b?.type === 'tool_result');
    if (text !== '' || toolUses.length === 0) messages.push({ role, content: text });
    for (const use of toolUses) {
      messages.push({
        role: 'assistant',
        content: '',
        tool_calls: [{ id: use.id ?? 'call-entry', type: 'function', function: { name: use.name, arguments: JSON.stringify(use.input ?? {}) } }],
      });
    }
    for (const result of toolResults) {
      messages.push({
        role: 'tool',
        tool_call_id: result.tool_use_id ?? 'call-entry',
        content: typeof result.content === 'string' ? result.content : JSON.stringify(result.content ?? ''),
      });
    }
  }
  const chat = { messages };
  if (typeof body.max_tokens === 'number') chat.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') chat.temperature = body.temperature;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    chat.tools = body.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description ?? '', parameters: t.input_schema ?? {} },
    }));
  }
  return chat;
}

// ---------------------------------------------------------------- the walk ----

// Serve one prompt: walk the eligible models in rating order until one
// answers with the token. Returns { served, answer } where answer is the
// relay payload (already wire-shaped) or null when nothing served.
async function servePrompt(wire, body) {
  const token = tokenOf(body);
  const promptKey = token ?? JSON.stringify(body).slice(0, 64);
  const requires = {
    json: wire === 'openai' && body?.response_format?.type === 'json_object',
    tools: Array.isArray(body?.tools) && body.tools.length > 0,
  };
  const stream = body?.stream === true;
  const declaredTools = (body?.tools ?? []).map((t) => t?.name).filter((n) => typeof n === 'string');
  const order = eligibleModels(requires);
  const failures = [];
  let consecutiveRefusals = 0;

  for (const model of order) {
    if (dead.has(model.id)) continue;
    const chatBody = wire === 'anthropic' ? anthropicToChat(body) : { ...body };
    delete chatBody.model;
    // A model gets one retry, and only on its first refusal in the trial.
    const firstRefusal = (refusalsByModel.get(model.id) ?? 0) === 0;
    const attempts = firstRefusal ? 2 : 1;
    for (let attempt = 0; attempt < attempts; attempt++) {
      const dial = await dialChatCompletions(chatBody, stream, model.id);
      if (dial.status >= 200 && dial.status < 300) {
        const parsed = parseGatewayAnswer(dial.status, dial.text, stream);
        if (parsed === null || parsed.content === '') {
          // A 200 that carries no usable answer: a content failure, not a
          // gateway refusal. Move on without a retry or a header row.
          consecutiveRefusals = 0;
          break;
        }
        if (token !== null && !parsed.content.includes(token)) {
          // The model answered but never named the token: it earns the
          // prompt nothing. Fail over; do not retry the same model.
          consecutiveRefusals = 0;
          break;
        }
        // Served. Relay the provider's answer, whole: usage injected when
        // the provider sent none, stop reason coerced to a legal one, the
        // model field the id actually dialled, undeclared tool calls cut.
        const usage = parsed.usage ?? {
          prompt_tokens: estimateTokens(JSON.stringify(chatBody)),
          completion_tokens: Math.max(1, estimateTokens(parsed.content)),
        };
        if (usage.completion_tokens < 1) usage.completion_tokens = Math.max(1, estimateTokens(parsed.content));
        const keptCalls = keepDeclaredCalls(parsed.toolCalls, declaredTools);
        const openAIStop = coerceOpenAIStop(parsed.finishReason);
        const headers = failures.length > 0
          ? { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_HEADER_ROWS)) }
          : {};
        if (wire === 'anthropic') {
          const toolBlocks = keptCalls.map((call) => ({
            type: 'tool_use',
            id: call.id ?? 'call-entry',
            name: call.function.name,
            input: (() => { try { return JSON.parse(call.function.arguments ?? '{}'); } catch { return {}; } })(),
          }));
          const anthropicStop = openAIStopToAnthropic(openAIStop);
          if (stream) {
            return { served: true, headers, streamBody: anthropicStreamFrames(model.id, parsed.content, anthropicStop, usage), contentType: 'text/event-stream' };
          }
          return { served: true, headers, body: anthropicAnswer(model.id, parsed.content, toolBlocks, anthropicStop, usage) };
        }
        if (stream) {
          return { served: true, headers, streamBody: openAIStreamFrames(model.id, parsed.content, openAIStop, usage), contentType: 'text/event-stream' };
        }
        return { served: true, headers, body: openAIAnswer(model.id, parsed.content, keptCalls, openAIStop, usage) };
      }
      // A refusal (429, 5xx, timeout, network error). The header parser
      // requires a real HTTP status (100..599); a bare 0 would discard the
      // WHOLE header, including legitimate 429 rows. A timeout is a 504
      // (the upstream never answered) and a network error a 502.
      const timedOut = dial.timedOut === true;
      const headerStatus = dial.status >= 100 && dial.status <= 599
        ? dial.status
        : timedOut ? 504 : 502;
      failures.push({
        status: headerStatus,
        code: failureCode(dial.status, dial.text, timedOut, dial.networkError === true),
        model: model.id,
      });
      noteRefusal(model.id, promptKey, timedOut);
      consecutiveRefusals += 1;
      if (consecutiveRefusals >= MAX_CONSECUTIVE_REFUSALS) {
        return { served: false, headers: {}, failures };
      }
      if (timedOut || dead.has(model.id)) break; // a hung model is dead; do not retry it
      if (attempt === 0 && firstRefusal) continue; // the one retry
      break;
    }
  }
  return { served: false, headers: {}, failures };
}

// ------------------------------------------------------------------ server ----

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      const auth = req.headers.authorization ?? '';
      const apiKey = req.headers['x-api-key'];
      if (typeof apiKey === 'string' && apiKey !== '') {
        sendJson(res, 200, {
          object: 'list',
          data: catalog.providers[0].models.map((m) => ({
            id: m.id,
            created_at: FIXED_CREATED,
            display_name: bareId(m.id),
            type: 'model',
          })),
          has_more: false,
          first_id: catalog.providers[0].models[0].id,
          last_id: catalog.providers[0].models[catalog.providers[0].models.length - 1].id,
        });
        return;
      }
      void auth;
      sendJson(res, 200, {
        object: 'list',
        data: catalog.providers[0].models.map((m) => ({
          id: m.id,
          object: 'model',
          created: FIXED_CREATED,
          owned_by: 'openrouter',
        })),
      });
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      const body = await readBody(req);
      const result = await servePrompt('openai', body);
      if (!result.served) {
        sendJson(res, 503, { error: { message: 'no free model served this prompt', type: 'service_unavailable', code: 'no_model_served' } });
        return;
      }
      if (result.streamBody !== undefined) {
        res.writeHead(200, { 'content-type': result.contentType, ...result.headers });
        res.end(result.streamBody);
        return;
      }
      sendJson(res, 200, result.body, result.headers);
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/messages') {
      const body = await readBody(req);
      const result = await servePrompt('anthropic', body);
      if (!result.served) {
        sendJson(res, 503, { type: 'error', error: { type: 'overloaded_error', message: 'no free model served this prompt' } });
        return;
      }
      if (result.streamBody !== undefined) {
        res.writeHead(200, { 'content-type': result.contentType, ...result.headers });
        res.end(result.streamBody);
        return;
      }
      sendJson(res, 200, result.body, result.headers);
      return;
    }
    sendJson(res, 404, { error: { message: 'not found', type: 'not_found' } });
  } catch (err) {
    if (!res.headersSent) sendJson(res, 500, { error: { message: String(err?.message ?? err), type: 'internal_error' } });
  }
});

// --list prints the catalog and exits before any server starts.
if (process.argv.includes('--list')) {
  process.stdout.write(JSON.stringify(catalog) + '\n');
  process.exit(0);
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/$/, '');

async function postJson(path, value) {
  const res = await fetch(ENV_URL + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(value),
  });
  return { status: res.status, text: await res.text() };
}

async function getJson(path) {
  const res = await fetch(ENV_URL + path);
  return JSON.parse(await res.text());
}

async function main() {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  let registered = false;
  for (let attempt = 0; attempt < 10 && !registered; attempt++) {
    try {
      const r = await postJson('/register', { protocol_version: 1, port, catalog });
      if (r.status >= 200 && r.status < 300) registered = true;
    } catch { /* retry */ }
    if (!registered) await new Promise((r) => setTimeout(r, 500));
  }
  if (!registered) {
    console.error('entry: register failed');
    process.exit(1);
  }
  for (;;) {
    let work;
    try {
      work = await getJson('/work');
    } catch {
      await new Promise((r) => setTimeout(r, 500));
      continue;
    }
    if (work && work.done === true) {
      server.close();
      process.exit(0);
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}

main().catch((err) => {
  console.error('entry: fatal', err);
  process.exit(1);
});
