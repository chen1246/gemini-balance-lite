//Author: PublicAffairs
//Project: https://github.com/PublicAffairs/openai-gemini
//MIT License : https://github.com/PublicAffairs/openai-gemini/blob/main/LICENSE


import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      // Comma-separated keys are supported ("k1,k2,k3"). Gemini free-tier rate
      // limits are enforced per Google Cloud PROJECT, not per API key, so keys
      // sourced from separate projects each add a full quota. requestGoogle()
      // rotates across them and retries on 429 instead of failing outright.
      const auth = request.headers.get("Authorization");
      const rawKey = auth?.split(" ")[1] ?? "";
      const apiKeys = rawKey.split(",").map(k => k.trim()).filter(Boolean);
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKeys)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKeys)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKeys)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

// ---------------------------------------------------------------------------
// Multi-key rotation with quota-aware back-off.
//
// Why: Gemini's free tier allows ~250k tokens/minute (TPM) *per project* plus a
// per-project daily cap. A long agent session re-sends the whole context on
// every tool round-trip, so one key saturates almost instantly and Google
// answers 429 RESOURCE_EXHAUSTED. Because the limit is per project — not per
// key — supplying keys from several projects multiplies the ceiling.
//
// What this does:
//   1. Picks a key that is not currently cooling down (randomised).
//   2. On 429/503 it parks that key for the delay Google asks for (RetryInfo
//      retryDelay), or a token-aware estimate when that is absent, then retries.
//   3. Retries instantly if another key is free; otherwise waits briefly
//      (capped) before trying again.
//
// State is per-isolate and best-effort: Deno Deploy may run several isolates, so
// cooldowns are not globally shared. That is fine — the goal is simply to stop
// hammering a key that just said "stop".
// ---------------------------------------------------------------------------
const MAX_KEY_ATTEMPTS = 4;
const FREE_TIER_TPM = 250000;      // free-tier tokens-per-minute, per project
const MAX_RETRY_WAIT_MS = 5000;    // never block a single request longer than this

const keyCooldown = new Map();     // apiKey -> epoch ms at which it becomes usable

const pickKey = (keys) => {
  if (!keys || keys.length === 0) { return undefined; }
  const now = Date.now();
  const ready = keys.filter(k => (keyCooldown.get(k) ?? 0) <= now);
  const pool = ready.length ? ready : keys;
  return pool[Math.floor(Math.random() * pool.length)];
};

const coolKey = (key, ms) => {
  if (!key) { return; }
  const until = Date.now() + ms;
  if ((keyCooldown.get(key) ?? 0) < until) { keyCooldown.set(key, until); }
  if (keyCooldown.size > 256) {          // opportunistic cleanup
    const now = Date.now();
    for (const [k, v] of keyCooldown) { if (v <= now) { keyCooldown.delete(k); } }
  }
};

const parseRetryDelayMs = (text, approxTokens) => {
  try {
    const details = JSON.parse(text)?.error?.details ?? [];
    for (const item of details) {
      const m = typeof item?.retryDelay === "string"
        ? item.retryDelay.match(/^([\d.]+)s$/)
        : null;
      if (m) {
        return Math.min(Math.ceil(parseFloat(m[1]) * 1000), 60000);
      }
    }
  } catch { /* body was not JSON */ }
  // Token-aware fallback (same model as gemini-flux): cooldown ≈ tokens / TPM.
  const minutes = approxTokens / FREE_TIER_TPM;
  return Math.min(Math.max(Math.ceil(minutes * 60000), 3000), 60000);
};

const isQuotaError = (status, text) =>
  status === 429 || status === 503 || /RESOURCE_EXHAUSTED|exceeded your current quota/i.test(text);

// Request Google with key rotation. On success the caller receives the live
// Response (body untouched). On final failure it receives a rebuilt Response
// carrying Google's error body, so the client still sees the real status/message.
const requestGoogle = async ({ url, keys, method = "POST", body, contentType = "application/json" }) => {
  const keyList = Array.isArray(keys) && keys.length ? keys : [undefined];
  const approxTokens = body ? Math.ceil(body.length / 4) : 0;
  let response;
  let text = "";
  for (let attempt = 0; attempt < MAX_KEY_ATTEMPTS; attempt++) {
    const key = pickKey(keyList);
    response = await fetch(url, {
      method,
      headers: makeHeaders(key, contentType ? { "Content-Type": contentType } : undefined),
      body,
    });
    if (response.ok) { return response; }
    try { text = await response.text(); } catch { text = ""; }
    if (!isQuotaError(response.status, text)) { break; }
    coolKey(key, parseRetryDelayMs(text, approxTokens));
    if (attempt === MAX_KEY_ATTEMPTS - 1) { break; }
    const now = Date.now();
    const anyReady = keyList.some(k => (keyCooldown.get(k) ?? 0) <= now);
    if (!anyReady) {
      const soonest = Math.min(...keyList.map(k => (keyCooldown.get(k) ?? 0) - now));
      if (soonest > MAX_RETRY_WAIT_MS) { break; }   // not worth blocking the caller
      await new Promise(r => setTimeout(r, Math.max(soonest, 250)));
    }
    console.log(`Gemini quota hit (${response.status}); retrying (attempt ${attempt + 2}/${MAX_KEY_ATTEMPTS})`);
  }
  return new Response(text, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
};

async function handleModels (apiKeys) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(pickKey(apiKeys)),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  } else {
    try { body = await response.text(); } catch { body = null; }
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKeys) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    if (!req.model.startsWith("gemini-")) {
      req.model = DEFAULT_EMBEDDINGS_MODEL;
    }
    model = "models/" + req.model;
  }
  if (!Array.isArray(req.input)) {
    req.input = [ req.input ];
  }
  const response = await requestGoogle({
    url: `${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`,
    keys: apiKeys,
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    }, null, "  ");
  } else {
    try { body = await response.text(); } catch { body = null; }
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-2.5-flash";
async function handleCompletions (req, apiKeys) {
  let model = DEFAULT_MODEL;
  switch (true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("gemma-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  let body = await transformRequest(req);
  const extra = req.extra_body?.google
  if (extra) {
    if (extra.safety_settings) {
      body.safetySettings = extra.safety_settings;
    }
    if (extra.cached_content) {
      body.cachedContent = extra.cached_content;
    }
    if (extra.thinking_config) {
      body.generationConfig.thinkingConfig = extra.thinking_config;
    }
  }
  switch (true) {
    case model.endsWith(":search"):
      model = model.substring(0, model.length - 7);
      // eslint-disable-next-line no-fallthrough
    case req.model.endsWith("-search-preview"):
    case req.tools?.some(tool => tool.function?.name === 'googleSearch'):
      body.tools = body.tools || [];
      body.tools.push({googleSearch: {}});
  }
  console.log(body.tools)
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }
  const response = await requestGoogle({
    url,
    keys: apiKeys,
    body: JSON.stringify(body),
  });

  body = response.body;
  if (response.ok) {
    let id = "chatcmpl-" + generateId(); //"chatcmpl-8pMMaqXMK68B3nyDBrapTDrhkHBQK";
    const shared = {};
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
          shared,
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: req.stream_options?.include_usage,
          model, id, last: [],
          shared,
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        body = JSON.parse(body);
        if (!body.candidates) {
          throw new Error("Invalid completion object");
        }
      } catch (err) {
        console.error("Error parsing response:", err);
        return new Response(body, fixCors(response)); // output as is
      }
      body = processCompletionsResponse(body, model, id);
    }
  } else {
    try { body = await response.text(); } catch { body = null; }
  }
  return new Response(body, fixCors(response));
}

// Recursively strip OpenAI / Anthropic-only JSON-Schema keys that Gemini's
// proto-based Schema validator rejects (it returns HTTP 400 "Unknown name ...").
// WorkBuddy / Claude-style tool definitions send additionalProperties:false,
// strict:true, $schema, title, default, examples — none of which Gemini accepts.
const GEMINI_FORBIDDEN_SCHEMA_KEYS = [
  "additionalProperties",
  "strict",
  "$schema",
  "title",
  "default",
  "examples",
];

// `isPropertyMap` guards the single most dangerous edge case: inside a
// `properties` object the KEYS are user field names, not schema keywords.
// A tool argument literally called "title"/"default"/"examples" must survive.
// Deleting it while `required` still references it makes Gemini fail with
// "parameters.required[N]: property is not defined".
const stripForbiddenSchemaKeys = (node, isPropertyMap = false) => {
  if (node === null || typeof node !== "object") {
    return;
  }
  if (Array.isArray(node)) {
    node.forEach((item) => stripForbiddenSchemaKeys(item, false));
    return;
  }
  if (isPropertyMap) {
    // Keys are field names — recurse into each field's schema, delete nothing here.
    for (const value of Object.values(node)) {
      stripForbiddenSchemaKeys(value, false);
    }
    return;
  }
  for (const key of GEMINI_FORBIDDEN_SCHEMA_KEYS) {
    if (Object.prototype.hasOwnProperty.call(node, key)) {
      delete node[key];
    }
  }
  if (node.properties && typeof node.properties === "object" && !Array.isArray(node.properties)) {
    stripForbiddenSchemaKeys(node.properties, true);
  }
  if (node.items !== undefined) {
    stripForbiddenSchemaKeys(node.items, false);
  }
  // Gemini also rejects required[] entries that are absent from properties.
  if (Array.isArray(node.required)) {
    const names = node.properties && typeof node.properties === "object"
      ? Object.keys(node.properties)
      : [];
    node.required = node.required.filter((name) => typeof name === "string" && names.includes(name));
    if (node.required.length === 0) {
      delete node.required;
    }
  }
  for (const [key, value] of Object.entries(node)) {
    if (key === "properties" || key === "items" || key === "required") {
      continue;
    }
    stripForbiddenSchemaKeys(value, false);
  }
};
const adjustSchema = (schema) => {
  stripForbiddenSchemaKeys(schema);
  return schema;
};

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));
const fieldsMap = {
  frequency_penalty: "frequencyPenalty",
  max_completion_tokens: "maxOutputTokens",
  max_tokens: "maxOutputTokens",
  n: "candidateCount", // not for streaming
  presence_penalty: "presencePenalty",
  seed: "seed",
  stop: "stopSequences",
  temperature: "temperature",
  top_k: "topK", // non-standard
  top_p: "topP",
};
const thinkingBudgetMap = {
  low: 1024,
  medium: 8192,
  high: 24576,
};
const transformConfig = (req) => {
  let cfg = {};
  //if (typeof req.stop === "string") { req.stop = [req.stop]; } // no need
  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }
  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        adjustSchema(req.response_format);
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
        // eslint-disable-next-line no-fallthrough
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  if (req.reasoning_effort) {
    cfg.thinkingConfig = { thinkingBudget: thinkingBudgetMap[req.reasoning_effort] };
  }
  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new HttpError("Invalid image data: " + url, 400);
    }
    ({ mimeType, data } = match.groups);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

const transformFnResponse = ({ content, tool_call_id }, parts) => {
  if (!parts.calls) {
    throw new HttpError("No function calls found in the previous message", 400);
  }
  let response;
  try {
    response = JSON.parse(content);
  } catch {
    // Tool output is frequently NOT JSON — shell stdout, file listings, stack
    // traces, any multi-line plain text. Gemini's functionResponse.response must
    // be an object, so wrap the raw text instead of failing the whole request.
    console.error("Non-JSON function response, wrapping as text");
    response = { result: typeof content === "string" ? content : String(content ?? "") };
  }
  if (typeof response !== "object" || response === null || Array.isArray(response)) {
    response = { result: response };
  }
  if (!tool_call_id) {
    throw new HttpError("tool_call_id not specified", 400);
  }
  const { i, name } = parts.calls[tool_call_id] ?? {};
  if (!name) {
    throw new HttpError("Unknown tool_call_id: " + tool_call_id, 400);
  }
  if (parts[i]) {
    throw new HttpError("Duplicated tool_call_id: " + tool_call_id, 400);
  }
  parts[i] = {
    functionResponse: {
      id: tool_call_id.startsWith("call_") ? null : tool_call_id,
      name,
      response,
    }
  };
};

const transformFnCalls = ({ tool_calls }) => {
  const calls = {};
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type }, i) => {
    if (type !== "function") {
      throw new HttpError(`Unsupported tool_call type: "${type}"`, 400);
    }
    let args;
    try {
      args = JSON.parse(argstr);
    } catch {
      // Tolerate empty / non-JSON argument strings (e.g. no-arg tools) rather
      // than failing the whole request with 400.
      console.error("Non-JSON function arguments, defaulting to {}");
      args = {};
    }
    calls[id] = {i, name};
    return {
      functionCall: {
        id: id.startsWith("call_") ? null : id,
        name,
        args,
      },
      // Gemini 3 requires a thoughtSignature on the first functionCall of every
      // step. Replayed OpenAI-format history never carries one (the field has no
      // OpenAI equivalent), which triggers:
      //   400 "... missing a thought_signature in functionCall parts"
      // Google's documented escape hatch is the literal sentinel below. It must
      // stay a plain string — base64-encoding it (as some SDKs do with bytes)
      // makes Gemini reject it as "not valid".
      ...(i === 0 ? { thoughtSignature: "skip_thought_signature_validator" } : {}),
    };
  });
  parts.calls = calls;
  return parts;
};

const transformMsg = async ({ content }) => {
  const parts = [];
  if (!Array.isArray(content)) {
    // system, user: string
    // assistant: string or null (Required unless tool_calls is specified.)
    parts.push({ text: content });
    return parts;
  }
  // user:
  // An array of content parts with a defined type.
  // Supported options differ based on the model being used to generate the response.
  // Can contain text, image, or audio inputs.
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          }
        });
        break;
      default:
        throw new HttpError(`Unknown "content" item type: "${item.type}"`, 400);
    }
  }
  if (content.every(item => item.type === "image_url")) {
    parts.push({ text: "" }); // to avoid "Unable to submit request because it must have a text parameter"
  }
  return parts;
};

const transformMessages = async (messages) => {
  if (!messages) { return; }
  const contents = [];
  let system_instruction;
  for (const item of messages) {
    switch (item.role) {
      case "system":
        system_instruction = { parts: await transformMsg(item) };
        continue;
      case "tool": {
        // Gemini's Content role whitelist is now USER / MODEL only. Newer models
        // (e.g. gemini-3.6-flash) reject the legacy role outright:
        //   400 Role 'function' is not supported. Please use a valid role: ...
        // The official Gen AI SDK also returns functionResponse parts on a "user"
        // turn. Consecutive tool results must still be merged into ONE turn, so we
        // tag the parts array with an internal marker instead of relying on role —
        // JSON.stringify ignores array properties, so it is never sent upstream.
        let { parts } = contents[contents.length - 1] ?? {};
        if (!parts?.isFnResponse) {
          const calls = parts?.calls;
          parts = []; parts.calls = calls;
          parts.isFnResponse = true;
          contents.push({
            role: "user", // functionResponse must ride on a user turn
            parts
          });
        }
        transformFnResponse(item, parts);
        continue;
      }
      case "assistant":
        item.role = "model";
        break;
      case "user":
        break;
      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
    contents.push({
      role: item.role,
      parts: item.tool_calls ? transformFnCalls(item) : await transformMsg(item)
    });
  }
  if (system_instruction) {
    if (!contents[0]?.parts.some(part => part.text)) {
      contents.unshift({ role: "user", parts: { text: " " } });
    }
  }
  //console.info(JSON.stringify(contents, 2));
  return { system_instruction, contents };
};

const transformTools = (req) => {
  let tools, tool_config;
  if (req.tools) {
    const funcs = req.tools.filter(tool => tool.type === "function" && tool.function?.name !== 'googleSearch');
    if (funcs.length > 0) {
      funcs.forEach(adjustSchema);
      tools = [{ function_declarations: funcs.map(schema => schema.function) }];
    }
  }
  if (req.tool_choice) {
    const allowed_function_names = req.tool_choice?.type === "function" ? [ req.tool_choice?.function?.name ] : undefined;
    if (allowed_function_names || typeof req.tool_choice === "string") {
      tool_config = {
        function_calling_config: {
          mode: allowed_function_names ? "ANY" : req.tool_choice.toUpperCase(),
          allowed_function_names
        }
      };
    }
  }
  return { tools, tool_config };
};

const transformRequest = async (req) => ({
  ...await transformMessages(req.messages),
  safetySettings,
  generationConfig: transformConfig(req),
  ...transformTools(req),
});

const generateId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = { //https://ai.google.dev/api/rest/v1/GenerateContentResponse#finishreason
  //"FINISH_REASON_UNSPECIFIED": // Default value. This value is unused.
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
  //"OTHER": "OTHER",
};
const SEP = "\n\n|>";
const transformCandidates = (key, cand) => {
  const message = { role: "assistant", content: [] };
  for (const part of cand.content?.parts ?? []) {
    if (part.functionCall) {
      const fc = part.functionCall;
      message.tool_calls = message.tool_calls ?? [];
      message.tool_calls.push({
        id: fc.id ?? "call_" + generateId(),
        type: "function",
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.args),
        }
      });
    } else {
      message.content.push(part.text);
    }
  }
  message.content = message.content.join(SEP) || null;
  return {
    index: cand.index || 0, // 0-index is absent in new -002 models response
    [key]: message,
    logprobs: null,
    finish_reason: message.tool_calls ? "tool_calls" : reasonsMap[cand.finishReason] || cand.finishReason,
    //original_finish_reason: cand.finishReason,
  };
};
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data) => ({
  completion_tokens: data.candidatesTokenCount,
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount
});

const checkPromptBlock = (choices, promptFeedback, key) => {
  if (choices.length) { return; }
  if (promptFeedback?.blockReason) {
    console.log("Prompt block reason:", promptFeedback.blockReason);
    if (promptFeedback.blockReason === "SAFETY") {
      promptFeedback.safetyRatings
        .filter(r => r.blocked)
        .forEach(r => console.log(r));
    }
    choices.push({
      index: 0,
      [key]: null,
      finish_reason: "content_filter",
      //original_finish_reason: data.promptFeedback.blockReason,
    });
  }
  return true;
};

const processCompletionsResponse = (data, model, id) => {
  const obj = {
    id,
    choices: data.candidates.map(transformCandidatesMessage),
    created: Math.floor(Date.now()/1000),
    model: data.modelVersion ?? model,
    //system_fingerprint: "fp_69829325d0",
    object: "chat.completion",
    usage: data.usageMetadata && transformUsage(data.usageMetadata),
  };
  if (obj.choices.length === 0 ) {
    checkPromptBlock(obj.choices, data.promptFeedback, "message");
  }
  return JSON.stringify(obj);
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
function parseStream (chunk, controller) {
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true); // eslint-disable-line no-constant-condition
}
function parseStreamFlush (controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true;
  }
}

const delimiter = "\n\n";
const sseline = (obj) => {
  obj.created = Math.floor(Date.now()/1000);
  return "data: " + JSON.stringify(obj) + delimiter;
};
function toOpenAiStream (line, controller) {
  let data;
  try {
    data = JSON.parse(line);
    if (!data.candidates) {
      throw new Error("Invalid completion chunk object");
    }
  } catch (err) {
    console.error("Error parsing response:", err);
    if (!this.shared.is_buffers_rest) { line =+ delimiter; }
    controller.enqueue(line); // output as is
    return;
  }
  const obj = {
    id: this.id,
    choices: data.candidates.map(transformCandidatesDelta),
    //created: Math.floor(Date.now()/1000),
    model: data.modelVersion ?? this.model,
    //system_fingerprint: "fp_69829325d0",
    object: "chat.completion.chunk",
    usage: data.usageMetadata && this.streamIncludeUsage ? null : undefined,
  };
  if (checkPromptBlock(obj.choices, data.promptFeedback, "delta")) {
    controller.enqueue(sseline(obj));
    return;
  }
  console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
  const cand = obj.choices[0];
  cand.index = cand.index || 0; // absent in new -002 models response
  const finish_reason = cand.finish_reason;
  cand.finish_reason = null;
  if (!this.last[cand.index]) { // first
    controller.enqueue(sseline({
      ...obj,
      choices: [{ ...cand, tool_calls: undefined, delta: { role: "assistant", content: "" } }],
    }));
  }
  delete cand.delta.role;
  if ("content" in cand.delta) { // prevent empty data (e.g. when MAX_TOKENS)
    controller.enqueue(sseline(obj));
  }
  cand.finish_reason = finish_reason;
  if (data.usageMetadata && this.streamIncludeUsage) {
    obj.usage = transformUsage(data.usageMetadata);
  }
  cand.delta = {};
  this.last[cand.index] = obj;
}
function toOpenAiStreamFlush (controller) {
  if (this.last.length > 0) {
    for (const obj of this.last) {
      controller.enqueue(sseline(obj));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}