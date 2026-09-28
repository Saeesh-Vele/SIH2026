/**
 * Backend client.
 *
 * Two ways to run a query: `runQuery` waits for the whole thing, `streamQuery`
 * reports each controller step as it lands. The console uses the streaming one
 * so the trace fills in while the VQA model loads and decodes.
 *
 * Every call except the health probe carries `Authorization: Bearer <idToken>`.
 * The token is fetched per request — Firebase hands back a cached one and
 * refreshes it itself when it nears expiry — and never goes in a URL.
 */

import { getFirebaseAuth } from "./firebase";

import type {
  Intent,
  QueryResult,
  StagedFile,
  TraceStep,
  UploadMode,
  UploadResult,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The signed-in user's ID token as a header, or a 401 if nobody is signed in. */
async function authHeader(): Promise<{ Authorization: string }> {
  const user = getFirebaseAuth()?.currentUser;
  if (!user) throw new ApiError("Sign in to continue.", 401);
  return { Authorization: `Bearer ${await user.getIdToken()}` };
}

/** FastAPI puts the message in `detail`, which may itself be a validation list. */
async function readError(response: Response): Promise<string> {
  let detail: unknown;
  try {
    detail = (await response.json())?.detail;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => (d as { msg?: string }).msg ?? String(d)).join("; ");
  }
  return `${response.status} ${response.statusText}`;
}

export async function checkHealth(signal?: AbortSignal) {
  const response = await fetch(`${API_BASE}/health`, { signal });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return (await response.json()) as {
    status: "ok" | "degraded";
    version: string;
    mongo: boolean;
    tasks: string[];
  };
}

export async function uploadScene(
  mode: UploadMode,
  files: StagedFile[],
  benchmarkMode: boolean,
): Promise<UploadResult> {
  const form = new FormData();
  form.set("mode", mode);
  form.set("benchmark_mode", String(benchmarkMode));
  // Order matters: the backend zips files onto the roles the mode expects.
  for (const staged of files) form.append("files", staged.file, staged.name);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    headers: await authHeader(),
    body: form,
  });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return (await response.json()) as UploadResult;
}

interface QueryBody {
  query: string;
  upload_id: string | null;
  intent?: Intent;
}

/** Raw JSON shape of QueryResponse, before camel-casing. */
interface QueryResponseJson {
  query_id: string;
  trace_id: string;
  status: QueryResult["status"];
  intent: Intent | null;
  task_selected: QueryResult["taskSelected"];
  answer: string;
  confidence: number | null;
  degraded: boolean;
  degraded_reason: string | null;
  overlays: QueryResult["overlays"];
  models_used: string[];
  metrics: { label: string; value: string }[];
  steps: TraceStep[];
  error: string | null;
}

function toResult(json: QueryResponseJson): QueryResult {
  return {
    queryId: json.query_id,
    traceId: json.trace_id,
    status: json.status,
    intent: json.intent,
    taskSelected: json.task_selected,
    answer: json.answer,
    confidence: json.confidence,
    degraded: json.degraded,
    degradedReason: json.degraded_reason,
    overlays: json.overlays ?? [],
    modelsUsed: json.models_used ?? [],
    metrics: json.metrics ?? [],
    steps: json.steps ?? [],
    error: json.error,
  };
}

export async function runQuery(body: QueryBody, signal?: AbortSignal): Promise<QueryResult> {
  const response = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { ...(await authHeader()), "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return toResult(await response.json());
}

export interface StreamHandlers {
  /** Node names the controller expects to run, sent before anything does. */
  onPlan?: (nodes: string[], queryId: string) => void;
  onStep?: (step: TraceStep) => void;
}

/**
 * Run a query over SSE.
 *
 * EventSource cannot POST, so this reads the body stream and parses frames by
 * hand. Frames are separated by a blank line; a frame's `data:` lines are
 * joined before parsing.
 */
export async function streamQuery(
  body: QueryBody,
  handlers: StreamHandlers = {},
  signal?: AbortSignal,
): Promise<QueryResult> {
  const response = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: {
      ...(await authHeader()),
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) throw new ApiError(await readError(response), response.status);
  if (!response.body) throw new ApiError("The server sent no response body", 500);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: QueryResult | null = null;
  let streamError: string | null = null;

  const handleFrame = (frame: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    if (data.length === 0) return;

    const payload = JSON.parse(data.join("\n"));
    if (event === "plan") handlers.onPlan?.(payload.nodes, payload.query_id);
    else if (event === "step") handlers.onStep?.(payload as TraceStep);
    else if (event === "result") result = toResult(payload as QueryResponseJson);
    else if (event === "error") streamError = payload.message ?? "The run failed";
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      handleFrame(buffer.slice(0, split));
      buffer = buffer.slice(split + 2);
      split = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) handleFrame(buffer);

  if (streamError) throw new ApiError(streamError, 500);
  if (!result) throw new ApiError("The run ended without a result", 500);
  return result;
}
