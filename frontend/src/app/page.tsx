import Link from "next/link";

/**
 * Placeholder landing page. Phase 2 replaces it with the full landing; until
 * then it only needs to get a visitor into the console.
 */
export default function LandingPage() {
  return (
    <main className="texture-graticule flex min-h-dvh flex-col items-center justify-center gap-6 bg-void px-4 text-center">
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-semibold tracking-tight text-foreground">SatQuery</span>
        <span className="font-mono text-xs text-signal">AI</span>
      </div>
      <p className="max-w-md text-[15px] leading-relaxed text-text-dim">
        Ask questions about satellite imagery in plain language, and see how each answer was
        produced.
      </p>
      <Link
        href="/console"
        className="rounded-sm bg-signal px-5 py-2.5 text-[13px] font-medium text-void transition-colors hover:bg-signal/85"
      >
        Launch console
      </Link>
    </main>
  );
}
