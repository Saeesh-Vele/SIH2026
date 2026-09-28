import Link from "next/link";
import { Wordmark } from "@/components/landing/wordmark";

export function SiteHeader() {
  return (
    <header className="relative z-20 mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-4 sm:px-6">
      <Link href="/" className="rounded-sm" aria-label="SatQuery AI home">
        <Wordmark />
      </Link>
      <nav aria-label="Primary" className="flex items-center gap-1 sm:gap-2">
        <a
          href="#how-it-works"
          className="hidden rounded-sm px-3 py-2 text-[13px] text-text-dim transition-colors hover:text-foreground sm:inline"
        >
          How it works
        </a>
        <Link
          href="/console"
          className="rounded-sm border border-rule-strong px-3 py-2 text-[13px] text-foreground transition-colors hover:border-signal hover:text-signal"
        >
          Launch console
        </Link>
      </nav>
    </header>
  );
}
