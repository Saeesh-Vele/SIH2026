/** Where to land after signing in when nothing (valid) was asked for. */
export const DEFAULT_AFTER_SIGN_IN = "/console";

/**
 * The `?next=` target, only if it is a path on this site.
 *
 * Rejects absolute URLs and protocol-relative ones (`//evil.com`, `/\evil.com`,
 * which browsers treat the same way) so the sign-in page cannot be used to
 * bounce someone to another origin.
 */
export function safeNext(raw: string | null | undefined): string {
  if (!raw) return DEFAULT_AFTER_SIGN_IN;
  if (!raw.startsWith("/") || raw.startsWith("//") || raw.startsWith("/\\")) {
    return DEFAULT_AFTER_SIGN_IN;
  }
  // Control characters have no business in a path and can smuggle a scheme.
  if (/[\u0000-\u001f\u007f]/.test(raw)) return DEFAULT_AFTER_SIGN_IN;
  if (raw.startsWith("/sign-in")) return DEFAULT_AFTER_SIGN_IN;
  return raw;
}

export function signInHref(next: string): string {
  return `/sign-in?next=${encodeURIComponent(next)}`;
}
