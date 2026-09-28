"use client";

import { useState } from "react";
import { AlertTriangle } from "lucide-react";
import { useAuth } from "@/components/auth/auth-provider";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authErrorMessage, isUserCancelled } from "@/lib/auth-errors";

/**
 * Links the guest session to Google or an email/password login. The uid does
 * not change, so the history the guest built up comes along unchanged.
 */
export function UpgradeGuestDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { upgradeWithGoogle, upgradeWithEmail } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState<"google" | "email" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function attempt(key: "google" | "email", action: () => Promise<void>) {
    setPending(key);
    setError(null);
    try {
      await action();
      onOpenChange(false);
    } catch (err) {
      // Closing the Google window here just means "not now".
      setError(isUserCancelled(err) ? null : authErrorMessage(err));
    } finally {
      setPending(null);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm rounded-sm border-rule bg-panel">
        <DialogHeader>
          <DialogTitle className="text-base font-medium">Keep your history</DialogTitle>
          <DialogDescription className="text-[13px] leading-relaxed text-muted-foreground">
            Turn this guest session into an account. Everything you&rsquo;ve run so far stays
            in your history, and you can sign in from any browser.
          </DialogDescription>
        </DialogHeader>

        <Button
          type="button"
          variant="outline"
          disabled={pending !== null}
          onClick={() => attempt("google", upgradeWithGoogle)}
          className="h-9 w-full rounded-sm border-rule-strong bg-raised text-[13px] hover:bg-raised hover:text-signal"
        >
          {pending === "google" ? "Waiting for Google…" : "Use my Google account"}
        </Button>

        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            void attempt("email", () => upgradeWithEmail(email.trim(), password));
          }}
        >
          <div className="space-y-1.5">
            <Label htmlFor="upgrade-email" className="text-[12px] text-text-dim">
              Email
            </Label>
            <Input
              id="upgrade-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="h-9 rounded-sm border-rule-strong bg-raised text-[13px]"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="upgrade-password" className="text-[12px] text-text-dim">
              Choose a password
            </Label>
            <Input
              id="upgrade-password"
              type="password"
              autoComplete="new-password"
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="h-9 rounded-sm border-rule-strong bg-raised text-[13px]"
            />
          </div>
          <Button
            type="submit"
            disabled={pending !== null || !email.trim() || password.length < 6}
            className="h-9 w-full rounded-sm bg-signal text-[13px] text-void hover:bg-signal/85"
          >
            {pending === "email" ? "Creating account…" : "Create account"}
          </Button>
        </form>

        <div aria-live="polite" className="empty:hidden">
          {error ? (
            <p className="flex items-start gap-2 border border-alarm bg-raised p-2.5 text-[12.5px] leading-relaxed text-alarm">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              {error}
            </p>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
