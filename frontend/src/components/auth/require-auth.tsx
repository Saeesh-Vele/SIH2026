"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/components/auth/auth-provider";
import { StatusDot } from "@/components/status-dot";
import { signInHref } from "@/lib/safe-next";

/**
 * Renders its children only for a signed-in user (guests count). Anyone else
 * goes to /sign-in with the page they wanted as `?next=`, so they come back to
 * it — sample selection included — once signed in.
 *
 * The check runs in the browser because that is where the Firebase session
 * lives. It is a convenience, not the security boundary: the API verifies the
 * token on every request regardless.
 */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading || user) return;
    router.replace(signInHref(`${pathname}${window.location.search}`));
  }, [loading, user, pathname, router]);

  if (!user) {
    return (
      <div className="grid h-dvh place-items-center bg-void" role="status" aria-live="polite">
        <p className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          <StatusDot tone="signal" pulse />
          {loading ? "Checking your sign-in…" : "Taking you to sign-in…"}
        </p>
      </div>
    );
  }

  return <>{children}</>;
}
