"use client";

import { useEffect, useRef, useState } from "react";
import { useInView, useReducedMotion } from "motion/react";
import { AlertTriangle } from "lucide-react";
import { formatDuration } from "@/lib/format";
import { isRecorded, landingExample, type LandingExample } from "@/lib/landing-example";

/**
 * "Ask it anything": one recorded question replayed as it happened — the
 * question typed out, then the answer, then the trace line that produced it.
 * Everything shown comes from src/content/landing-example.json, which only
 * ever holds a real /api/query response. Until one is recorded the whole
 * section is left out; it appears on the next build after the file is filled.
 */
export function AskExample() {
  if (!isRecorded(landingExample)) return null;

  return (
    <section className="mx-auto grid w-full max-w-6xl gap-10 px-4 py-20 sm:px-6 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] md:items-center md:gap-16 md:py-28">
      <div className="max-w-md">
        <h2 className="text-balance text-[clamp(1.75rem,3.4vw,2.5rem)] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground">
          Ask it anything about an image
        </h2>
        <p className="mt-5 text-[16px] leading-relaxed text-text-dim">
          Type a question the way you would ask a colleague. SatQuery works out what you&rsquo;re
          asking, checks the imagery can answer it, runs the model that fits, and shows you each
          step it took.
        </p>
      </div>

      <Replay example={landingExample} />
    </section>
  );
}

function Terminal({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <figure
      aria-label={label}
      className="overflow-hidden rounded-(--radius-surface) border border-rule bg-panel"
    >
      <div className="flex h-9 items-center gap-2 border-b border-rule px-4" aria-hidden>
        <span className="size-2 rounded-full bg-rule-strong" />
        <span className="size-2 rounded-full bg-rule-strong" />
        <span className="size-2 rounded-full bg-rule-strong" />
      </div>
      <div className="min-h-64 space-y-5 p-5 font-mono text-[13px] leading-relaxed sm:p-6">
        {children}
      </div>
    </figure>
  );
}

type Stage = "idle" | "typing" | "answer" | "trace";

function Replay({
  example,
}: {
  example: LandingExample & { question: string; answer: string };
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-120px" });
  const reduceMotion = useReducedMotion();
  const [typed, setTyped] = useState(0);
  const [stage, setStage] = useState<Stage>("idle");

  // The first render is the same everywhere — the server cannot know the
  // motion preference — and reduced motion skips straight to the end once the
  // panel is in view.
  const done = stage === "trace";
  const shownQuestion = example.question.slice(0, typed);

  useEffect(() => {
    if (!inView) return;
    const timers: number[] = [];
    if (reduceMotion) {
      timers.push(
        window.setTimeout(() => {
          setTyped(example.question.length);
          setStage("trace");
        }, 0),
      );
      return () => timers.forEach(window.clearTimeout);
    }
    let i = 0;
    const tick = window.setInterval(() => {
      i += 1;
      setTyped(i);
      if (i >= example.question.length) {
        window.clearInterval(tick);
        timers.push(window.setTimeout(() => setStage("answer"), 450));
        timers.push(window.setTimeout(() => setStage("trace"), 1300));
      }
    }, 28);
    // Deferred so the first state change happens outside the effect body.
    timers.push(window.setTimeout(() => setStage("typing"), 0));
    return () => {
      window.clearInterval(tick);
      timers.forEach(window.clearTimeout);
    };
  }, [inView, reduceMotion, example.question]);

  const showAnswer = stage === "answer" || stage === "trace";
  const trace = example.trace_line;

  return (
    <div ref={ref}>
      <Terminal label="A recorded example run">
        {/* Screen readers get the whole exchange at once, not the typing. */}
        <div className="sr-only">
          <p>Question: {example.question}</p>
          <p>Answer: {example.answer}</p>
        </div>

        <p className="flex gap-3 text-foreground" aria-hidden>
          <span className="text-signal">›</span>
          <span>
            {shownQuestion}
            {!done && stage !== "answer" ? (
              <span className="animate-signal-pulse ml-0.5 inline-block h-4 w-2 translate-y-0.5 bg-signal" />
            ) : null}
          </span>
        </p>

        <div
          aria-hidden
          className="space-y-3 font-sans text-[15px] leading-relaxed text-text transition-opacity duration-(--dur-slow)"
          style={{ opacity: showAnswer ? 1 : 0 }}
        >
          {example.degraded ? (
            <span className="inline-flex items-center gap-1.5 rounded-sm border border-caution bg-caution/15 px-1.5 py-0.5 font-mono text-[11px] text-caution">
              <AlertTriangle className="size-3" aria-hidden />
              Degraded mode — CPU fallback
            </span>
          ) : null}
          <p>{example.answer}</p>
          <p className="font-mono text-[12px] text-muted-foreground">
            confidence{" "}
            <span className={example.confidence === null ? "text-caution" : "text-text-dim"}>
              {example.confidence === null ? "not measured" : example.confidence.toFixed(2)}
            </span>
          </p>
        </div>

        {trace ? (
          <div
            className="grid grid-cols-[auto_1fr_auto] items-baseline gap-x-3 border-t border-rule pt-4 text-[12px] transition-opacity duration-(--dur-slow)"
            style={{ opacity: done ? 1 : 0 }}
          >
            <span className="size-1.5 translate-y-[-1px] rounded-full bg-signal" aria-hidden />
            <span className="min-w-0">
              <span className="text-foreground">{trace.label}</span>
              {trace.detail ? (
                <span className="mt-0.5 block truncate text-muted-foreground">{trace.detail}</span>
              ) : null}
            </span>
            <span className="text-muted-foreground">
              {trace.duration_ms ? formatDuration(trace.duration_ms) : null}
            </span>
          </div>
        ) : null}
      </Terminal>
    </div>
  );
}
