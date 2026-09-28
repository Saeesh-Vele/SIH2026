"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { AlertTriangle, ArrowRight } from "lucide-react";
import { useAuth } from "@/components/auth/auth-provider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authErrorMessage, isUserCancelled } from "@/lib/auth-errors";
import { safeNext } from "@/lib/safe-next";
import { cn } from "@/lib/utils";

type Mode = "sign-in" | "sign-up" | "reset";

export default function SignInPage() {
  return (
    <main className="texture-graticule flex min-h-dvh items-center justify-center bg-void px-4 py-10">
      <Suspense fallback={null}>
        <SignInCard />
      </Suspense>
    </main>
  );
}

function SignInCard() {
  const auth = useAuth();
  const router = useRouter();
  const next = safeNext(useSearchParams().get("next"));

  const [mode, setMode] = useState<Mode>("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<{ text: string; soft: boolean } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Signed in — whether just now or from a restored session — go where asked.
  useEffect(() => {
    if (auth.user) router.replace(next);
  }, [auth.user, next, router]);

  async function attempt(key: string, action: () => Promise<void>) {
    setPending(key);
    setError(null);
    setNotice(null);
    try {
      await action();
    } catch (err) {
      setError({ text: authErrorMessage(err), soft: isUserCancelled(err) });
    } finally {
      setPending(null);
    }
  }

  function submitEmail(e: React.FormEvent) {
    e.preventDefault();
    if (mode === "reset") {
      void attempt("email", async () => {
        await auth.sendPasswordReset(email.trim());
        // Same reply whether or not the account exists.
        setNotice(`If an account exists for ${email.trim()}, a reset link is on its way.`);
      });
    } else if (mode === "sign-up") {
      void attempt("email", () => auth.signUpWithEmail(email.trim(), password));
    } else {
      void attempt("email", () => auth.signInWithEmail(email.trim(), password));
    }
  }

  if (!auth.configured) {
    return (
      <Card>
        <p className="flex items-start gap-2 text-[13px] leading-relaxed text-caution">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
          Sign-in isn&rsquo;t set up on this deployment.
        </p>
        <p className="font-mono text-[11px] leading-relaxed text-muted-foreground">
          Set the <code className="text-text-dim">NEXT_PUBLIC_FIREBASE_*</code> variables listed in{" "}
          <code className="text-text-dim">frontend/.env.example</code>, then rebuild.
        </p>
      </Card>
    );
  }

  const busy = pending !== null || auth.loading;

  return (
    <Card>
      <div className="space-y-1.5">
        <Link href="/" className="inline-flex items-baseline gap-2 rounded-sm">
          <span className="text-[15px] font-semibold tracking-tight text-foreground">SatQuery</span>
          <span className="font-mono text-[10px] text-signal">AI</span>
        </Link>
        <h1 className="text-xl font-medium text-foreground">Sign in to the console</h1>
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          Your queries and their traces are saved to your account, visible only to you.
        </p>
      </div>

      <div className="space-y-2">
        <Button
          type="button"
          disabled={busy}
          onClick={() => attempt("guest", auth.continueAsGuest)}
          className="h-10 w-full justify-between rounded-sm bg-signal px-4 text-[13px] font-medium text-void hover:bg-signal/85"
        >
          {pending === "guest" ? "Starting a guest session…" : "Continue as guest"}
          <ArrowRight className="size-4" aria-hidden />
        </Button>
        <p className="text-[12px] leading-relaxed text-muted-foreground">
          No account needed. You can turn it into a full account later and keep your history.
        </p>
        <Button
          type="button"
          variant="outline"
          disabled={busy}
          onClick={() => attempt("google", auth.signInWithGoogle)}
          className="h-10 w-full gap-2 rounded-sm border-rule-strong bg-raised text-[13px] text-foreground hover:bg-raised hover:text-signal"
        >
          <GoogleMark />
          {pending === "google" ? "Waiting for Google…" : "Continue with Google"}
        </Button>
      </div>

      <div className="flex items-center gap-3" aria-hidden>
        <span className="h-px flex-1 bg-rule" />
        <span className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground">OR EMAIL</span>
        <span className="h-px flex-1 bg-rule" />
      </div>

      <form onSubmit={submitEmail} className="space-y-3" noValidate>
        <div className="space-y-1.5">
          <Label htmlFor="email" className="text-[12px] text-text-dim">
            Email
          </Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="h-9 rounded-sm border-rule-strong bg-raised text-[13px]"
          />
        </div>

        {mode !== "reset" ? (
          <div className="space-y-1.5">
            <Label htmlFor="password" className="text-[12px] text-text-dim">
              Password
            </Label>
            <Input
              id="password"
              type="password"
              autoComplete={mode === "sign-up" ? "new-password" : "current-password"}
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              aria-describedby={mode === "sign-up" ? "password-hint" : undefined}
              className="h-9 rounded-sm border-rule-strong bg-raised text-[13px]"
            />
            {mode === "sign-up" ? (
              <p id="password-hint" className="text-[11px] text-muted-foreground">
                At least 6 characters.
              </p>
            ) : (
              <div className="text-right">
                <TextButton onClick={() => switchMode("reset")}>Forgot password?</TextButton>
              </div>
            )}
          </div>
        ) : (
          <p className="text-[12px] leading-relaxed text-muted-foreground">
            Enter your account email and we&rsquo;ll send a link to set a new password.
          </p>
        )}

        <Button
          type="submit"
          variant="outline"
          disabled={busy || !email.trim() || (mode !== "reset" && !password)}
          className="h-9 w-full rounded-sm border-rule-strong bg-transparent text-[13px] text-foreground hover:border-signal hover:bg-transparent hover:text-signal"
        >
          {pending === "email"
            ? "One moment…"
            : mode === "sign-up"
              ? "Create account"
              : mode === "reset"
                ? "Send reset link"
                : "Sign in"}
        </Button>

        <p className="text-center text-[12px] text-muted-foreground">
          {mode === "sign-in" ? (
            <>
              New here? <TextButton onClick={() => switchMode("sign-up")}>Create an account</TextButton>
            </>
          ) : (
            <TextButton onClick={() => switchMode("sign-in")}>Back to sign in</TextButton>
          )}
        </p>
      </form>

      <div aria-live="polite" className="empty:hidden">
        {error ? (
          <p
            className={cn(
              "flex items-start gap-2 border bg-raised p-2.5 text-[12.5px] leading-relaxed",
              error.soft ? "border-rule-strong text-text-dim" : "border-alarm text-alarm",
            )}
          >
            {error.soft ? null : <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />}
            {error.text}
          </p>
        ) : null}
        {notice ? (
          <p className="border border-signal-deep bg-raised p-2.5 text-[12.5px] leading-relaxed text-text-dim">
            {notice}
          </p>
        ) : null}
      </div>
    </Card>
  );

  function switchMode(nextMode: Mode) {
    setMode(nextMode);
    setError(null);
    setNotice(null);
  }
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full max-w-sm space-y-6 border border-rule bg-panel p-6 sm:p-8">{children}</div>
  );
}

function TextButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-sm text-[12px] text-signal underline-offset-4 hover:underline"
    >
      {children}
    </button>
  );
}

/** Google's "G", in its own colours, as their branding guidelines ask. */
function GoogleMark() {
  return (
    <svg viewBox="0 0 18 18" className="size-4" aria-hidden>
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62Z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18Z" />
      <path fill="#FBBC05" d="M3.97 10.72A5.4 5.4 0 0 1 3.68 9c0-.6.1-1.18.29-1.72V4.95H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.05l3.01-2.33Z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58Z" />
    </svg>
  );
}
