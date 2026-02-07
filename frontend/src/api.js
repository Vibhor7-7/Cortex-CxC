/**
 * Cortex API Client
 *
 * Fetch-based wrapper for the Cortex backend REST API.
 * All functions return Promises and handle errors uniformly.
 *
 * Base URL defaults to http://localhost:8000 and can be changed
 * via {@link setBaseUrl}.
 *
 * @module api
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/** @type {string} Backend base URL (no trailing slash) */
let BASE_URL = "http://localhost:8000";

/** Max retries for transient failures (5xx / network errors) */
const MAX_RETRIES = 3;

/** Base delay (ms) for exponential back-off between retries */
const RETRY_BASE_DELAY_MS = 500;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Change the backend base URL at runtime.
 * @param {string} url – New base URL, e.g. "http://192.168.1.5:8000"
 */
export function setBaseUrl(url) {
  BASE_URL = url.replace(/\/+$/, ""); // strip trailing slashes
}

/**
 * Return the current backend base URL.
 * @returns {string}
 */
export function getBaseUrl() {
  return BASE_URL;
}

/**
 * Custom error class with HTTP status information.
 */
export class ApiError extends Error {
  /**
   * @param {string}  message  – Human-readable description
   * @param {number}  status   – HTTP status code (0 for network errors)
   * @param {*}       [body]   – Parsed response body (if any)
   */
  constructor(message, status, body = null) {
    super(message);
    this.name = "ApiError";
    /** @type {number} */
    this.status = status;
    /** @type {*} */
    this.body = body;
  }
}

/**
 * Sleep helper for retry back-off.
 * @param {number} ms
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Determine whether an HTTP status code is retryable (server error).
 * @param {number} status
 * @returns {boolean}
 */
function isRetryable(status) {
  return status === 0 || status >= 500;
}

/**
 * Core fetch wrapper with:
 *  – JSON parsing
 *  – Retry with exponential back-off for 5xx / network errors
 *  – Uniform {@link ApiError} on failure
 *
 * @param {string}       path     – URL path relative to BASE_URL (e.g. "/api/chats")
 * @param {RequestInit}  [init]   – Standard fetch options
 * @param {number}       [attempt] – Current retry attempt (internal)
 * @returns {Promise<*>}  Parsed JSON body
 * @throws {ApiError}
 */
async function request(path, init = {}, attempt = 1) {
  const url = `${BASE_URL}${path}`;

  /** @type {Response|null} */
  let res = null;

  try {
    res = await fetch(url, {
      ...init,
      headers: {
        "Accept": "application/json",
        ...(init.headers || {}),
      },
    });
  } catch (networkErr) {
    // Network / CORS / DNS failure
    if (attempt < MAX_RETRIES) {
      const delay = RETRY_BASE_DELAY_MS * 2 ** (attempt - 1);
      console.warn(
        `[api] Network error on ${init.method || "GET"} ${path} – retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`,
        networkErr
      );
      await sleep(delay);
      return request(path, init, attempt + 1);
    }
    throw new ApiError(
      `Network error: ${networkErr.message}`,
      0
    );
  }

  // Server errors → retry
  if (isRetryable(res.status) && attempt < MAX_RETRIES) {
    const delay = RETRY_BASE_DELAY_MS * 2 ** (attempt - 1);
    console.warn(
      `[api] ${res.status} on ${init.method || "GET"} ${path} – retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`
    );
    await sleep(delay);
    return request(path, init, attempt + 1);
  }

  // Try to parse body
  let body = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    try {
      body = await res.json();
    } catch {
      body = null;
    }
  } else {
    body = await res.text();
  }

  if (!res.ok) {
    const detail =
      (body && typeof body === "object" && body.detail) ||
      (typeof body === "string" && body) ||
      res.statusText;
    throw new ApiError(
      `${res.status} ${res.statusText}: ${detail}`,
      res.status,
      body
    );
  }

  return body;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Fetch all conversations formatted for 3D visualization.
 *
 * Returns nodes with positions, clusters, summaries, etc.
 *
 * @returns {Promise<{
 *   nodes: Array<{
 *     id: string,
 *     title: string,
 *     summary: string|null,
 *     topics: string[],
 *     cluster_id: number|null,
 *     cluster_name: string|null,
 *     message_count: number,
 *     position: [number,number,number],
 *     start_position: [number,number,number],
 *     magnitude: number,
 *     created_at: string
 *   }>,
 *   total_nodes: number,
 *   clusters: Array<{cluster_id:number, cluster_name:string, count:number}>
 * }>}
 * @throws {ApiError}
 */
export async function fetchChats() {
  return request("/api/chats/visualization");
}

/**
 * Semantic search over conversations via the backend hybrid search.
 *
 * The backend embeds the query with nomic-embed-text, searches the
 * local vector store, and returns ranked results.
 *
 * @param {string}  query           – Natural-language search query
 * @param {number}  [limit=30]      – Max results to return
 * @param {number}  [clusterFilter] – Optional cluster_id to restrict results
 * @returns {Promise<{
 *   query: string,
 *   results: Array<{
 *     conversation_id: string,
 *     title: string,
 *     summary: string|null,
 *     topics: string[],
 *     cluster_id: number|null,
 *     cluster_name: string|null,
 *     score: number,
 *     message_preview: string|null
 *   }>,
 *   total_results: number,
 *   search_time_ms: number
 * }>}
 * @throws {ApiError}
 */
export async function searchChats(query, limit = 30, clusterFilter = null) {
  const body = {
    query,
    limit,
  };
  if (clusterFilter != null && clusterFilter >= 0) {
    body.cluster_filter = clusterFilter;
  }

  return request("/api/search/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/**
 * Get full conversation details including all messages.
 *
 * @param {string} id – Conversation UUID
 * @returns {Promise<{
 *   id: string,
 *   title: string,
 *   summary: string|null,
 *   topics: string[],
 *   cluster_id: number|null,
 *   cluster_name: string|null,
 *   message_count: number,
 *   created_at: string,
 *   updated_at: string,
 *   messages: Array<{
 *     id: string,
 *     conversation_id: string,
 *     role: "user"|"assistant"|"system",
 *     content: string,
 *     sequence_number: number,
 *     created_at: string
 *   }>
 * }>}
 * @throws {ApiError}
 */
export async function fetchChatDetails(id) {
  return request(`/api/chats/${encodeURIComponent(id)}`);
}

/**
 * Upload an HTML chat export file for ingestion.
 *
 * The backend will parse it (ChatGPT / Claude), summarise,
 * generate embeddings, and store the conversation.
 *
 * @param {File}    file            – HTML file to upload
 * @param {boolean} [autoReprocess=false] – Re-run UMAP / clustering after ingest
 * @returns {Promise<{
 *   success: boolean,
 *   conversation_id: string|null,
 *   title: string|null,
 *   message_count: number,
 *   error: string|null,
 *   processing_time_ms: number
 * }>}
 * @throws {ApiError}
 */
export async function uploadChatFile(file, autoReprocess = false) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("auto_reprocess", String(autoReprocess));

  return request("/api/ingest/", {
    method: "POST",
    body: formData,
    // NOTE: Do NOT set Content-Type — the browser must set the
    // multipart boundary automatically.
  });
}

/**
 * Generate an LLM-powered system prompt from selected conversations.
 *
 * The backend gathers conversation summaries, sends them to Qwen 2.5,
 * and returns a polished system prompt the user can paste into a new
 * ChatGPT session.
 *
 * @param {string[]} conversationIds – UUIDs of the selected conversations
 * @returns {Promise<{
 *   prompt: string,
 *   conversations_used: number,
 *   processing_time_ms: number
 * }>}
 * @throws {ApiError}
 */
export async function generatePrompt(conversationIds) {
  return request("/api/prompt/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_ids: conversationIds }),
  });
}

/**
 * Quick health check — useful for connectivity indicator.
 *
 * @returns {Promise<{
 *   status: string,
 *   ollama_connected: boolean,
 *   chroma_ready: boolean
 * }>}
 * @throws {ApiError}
 */
export async function healthCheck() {
  return request("/health");
}
