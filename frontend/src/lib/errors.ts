import { ApiError } from "./api";

/** A failure, said as what happened and what the person can do about it. */
export interface Explained {
  title: string;
  detail: string;
  /** Set when signing in again is the fix. */
  signIn?: boolean;
}

/**
 * Turns a thrown error into plain language. The backend's own message is kept
 * as the detail where it is already written for people (FastAPI `detail`).
 */
export function explainError(error: unknown): Explained {
  if (error instanceof ApiError) {
    if (error.status === 401) {
      return {
        title: "Your sign-in has expired",
        detail: "Sign in again to carry on. Your earlier queries are still saved.",
        signIn: true,
      };
    }
    if (error.status === 503) {
      return {
        title: "The server can’t do that right now",
        detail: error.message,
      };
    }
    if (error.status === 404 || error.status === 410) {
      return { title: "That’s no longer available", detail: error.message };
    }
    if (error.status === 413) {
      return { title: "That file is too large", detail: error.message };
    }
    if (error.status === 415 || error.status === 400 || error.status === 422) {
      return { title: "That file can’t be used", detail: error.message };
    }
    return { title: "Something went wrong on the server", detail: error.message };
  }
  // fetch() rejects with a TypeError when the server cannot be reached at all.
  if (error instanceof TypeError) {
    return {
      title: "Can’t reach the SatQuery server",
      detail:
        "Check your connection and try again. If it keeps happening, the server may be restarting or offline.",
    };
  }
  return {
    title: "Something went wrong",
    detail: error instanceof Error ? error.message : String(error),
  };
}
