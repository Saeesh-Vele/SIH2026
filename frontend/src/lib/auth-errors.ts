/**
 * Firebase error codes, said the way a person would.
 *
 * Wrong password, unknown email and a malformed credential deliberately share
 * one message: telling them apart would let anyone probe which emails have
 * accounts. Firebase itself returns `auth/invalid-credential` for all three
 * when email-enumeration protection is on.
 */

const MESSAGES: Record<string, string> = {
  "auth/invalid-credential": "That email and password don't match. Check both, or reset your password.",
  "auth/wrong-password": "That email and password don't match. Check both, or reset your password.",
  "auth/user-not-found": "That email and password don't match. Check both, or reset your password.",
  "auth/invalid-email": "That doesn't look like an email address.",
  "auth/missing-password": "Enter your password.",
  "auth/weak-password": "Choose a password with at least 6 characters.",
  "auth/email-already-in-use":
    "An account with that email already exists. Sign in instead, or reset the password.",
  "auth/popup-closed-by-user": "The Google window was closed before sign-in finished. Try again when ready.",
  "auth/cancelled-popup-request": "The Google window was closed before sign-in finished. Try again when ready.",
  "auth/popup-blocked":
    "Your browser blocked the Google sign-in window. Allow pop-ups for this site and try again.",
  "auth/unauthorized-domain":
    "Sign-in isn't enabled for this web address yet. The site owner needs to add it under Authorized domains in the Firebase console.",
  "auth/network-request-failed":
    "Couldn't reach the sign-in service. Check your connection and try again.",
  "auth/too-many-requests": "Too many attempts in a row. Wait a minute, then try again.",
  "auth/operation-not-allowed":
    "This sign-in method is switched off for this project. Try another option.",
  "auth/admin-restricted-operation":
    "This sign-in method is switched off for this project. Try another option.",
  "auth/user-disabled": "This account has been disabled.",
  "auth/credential-already-in-use":
    "That account already exists separately. Sign out and sign in to it directly — this guest session's history stays with the guest session.",
  "auth/provider-already-linked": "This account is already upgraded.",
  "auth/requires-recent-login": "For security, sign out and back in, then try again.",
};

export function authErrorMessage(error: unknown): string {
  const code =
    typeof error === "object" && error !== null && "code" in error
      ? String((error as { code: unknown }).code)
      : null;
  if (code && MESSAGES[code]) return MESSAGES[code];
  return "Sign-in didn't work. Try again, or pick another option.";
}

/** Closing the popup yourself is a choice, not a failure worth a red banner. */
export function isUserCancelled(error: unknown): boolean {
  const code = (error as { code?: string } | null)?.code;
  return code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request";
}
