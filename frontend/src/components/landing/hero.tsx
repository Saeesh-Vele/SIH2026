import Link from "next/link";
import { HeroVisualLazy } from "@/components/landing/hero-visual-lazy";
import { sampleHref } from "@/lib/samples";

export function Hero() {
  return (
    <section className="relative isolate overflow-hidden">
      <HeroVisualLazy />

      <div className="relative z-10 mx-auto flex min-h-[min(88svh,820px)] w-full max-w-6xl flex-col justify-end px-4 pb-16 pt-[46vw] sm:px-6 md:justify-center md:pb-24 md:pt-10">
        <div className="max-w-[34rem]">
          {/* Painted immediately: it is the page's largest element, and holding
              it back for an entrance would hold back first paint with it. */}
          <h1
            className="text-balance text-[clamp(2.6rem,6.2vw,4.6rem)] font-semibold leading-[1.02] tracking-[-0.035em] text-foreground"
          >
            Ask a satellite image a question.
          </h1>
          <p
            className="animate-rise-in mt-6 max-w-[30rem] text-pretty text-[17px] leading-relaxed text-text-dim"
            style={{ animationDelay: "120ms" }}
          >
            Upload satellite imagery, ask in plain English, and get an answer along with a
            record of every step taken to reach it.
          </p>
          <div
            className="animate-rise-in mt-9 flex flex-wrap items-center gap-3"
            style={{ animationDelay: "240ms" }}
          >
            <Link
              href="/console"
              className="bg-accent-gradient rounded-sm px-5 py-3 text-[14px] font-semibold text-void shadow-glow-uplink transition-[filter] duration-(--dur-fast) hover:brightness-110"
            >
              Launch console
            </Link>
            <Link
              href={sampleHref()}
              className="rounded-sm border border-rule-strong bg-void/60 px-5 py-3 text-[14px] text-foreground backdrop-blur-sm transition-colors duration-(--dur-fast) hover:border-signal hover:text-signal"
            >
              Try a sample
            </Link>
          </div>
          <p
            className="animate-rise-in mt-4 text-[13px] text-muted-foreground"
            style={{ animationDelay: "320ms" }}
          >
            No account needed. You can continue as a guest.
          </p>
        </div>
      </div>
    </section>
  );
}
