"use client";

import { createRef, useRef, useState, type RefObject } from "react";
import { AlertTriangle } from "lucide-react";
import { AnimatedBeam } from "@/components/ui/animated-beam";
import { cn } from "@/lib/utils";

/**
 * The controller graph as it actually runs (backend/app/agent/graph.py):
 * intake -> intent_classifier -> input_validator -> task_router ->
 * one specialist -> output_combiner -> trace_logger.
 */
const STAGES = [
  { key: "intake", label: "Intake", explain: "Reads your files" },
  { key: "intent", label: "Intent", explain: "Works out what you’re asking" },
  { key: "validate", label: "Validate", explain: "Checks the imagery can answer it" },
  { key: "route", label: "Route", explain: "Picks the model for the job" },
] as const;

const SPECIALISTS = [
  { key: "vqa", label: "Q&A + captions" },
  { key: "change", label: "Change detection" },
  { key: "fusion", label: "Optical + SAR" },
] as const;

const CLOSING = [
  { key: "combine", label: "Combine", explain: "Shapes the answer" },
  { key: "trace", label: "Trace", explain: "Records every step" },
] as const;

type Key =
  | (typeof STAGES)[number]["key"]
  | (typeof SPECIALISTS)[number]["key"]
  | (typeof CLOSING)[number]["key"];

const EDGES: [Key, Key, number][] = [
  ["intake", "intent", 0],
  ["intent", "validate", 1],
  ["validate", "route", 2],
  ["route", "vqa", 3],
  ["route", "change", 3],
  ["route", "fusion", 3],
  ["vqa", "combine", 4],
  ["change", "combine", 4],
  ["fusion", "combine", 4],
  ["combine", "trace", 5],
];

/** One ref per node, created once. createRef objects are stable, so a map of
 *  them held in state is the same set every render. */
function useNodeRefs() {
  const [refs] = useState(
    () =>
      Object.fromEntries(
        [...STAGES, ...SPECIALISTS, ...CLOSING].map((node) => [node.key, createRef<HTMLDivElement>()]),
      ) as Record<Key, RefObject<HTMLDivElement | null>>,
  );
  return refs;
}

export function Pipeline() {
  const containerRef = useRef<HTMLDivElement>(null);
  const refs = useNodeRefs();

  return (
    <section
      id="how-it-works"
      className="mx-auto w-full max-w-6xl scroll-mt-8 px-4 py-20 sm:px-6 md:py-28"
    >
      <div className="max-w-2xl">
        <h2 className="text-balance text-[clamp(1.75rem,3.4vw,2.5rem)] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground">
          How it reaches an answer
        </h2>
        <p className="mt-5 text-[16px] leading-relaxed text-text-dim">
          Every answer comes with a trace you can audit step by step. If a run falls back to a
          smaller model, it is labelled{" "}
          <span className="inline-flex translate-y-[-1px] items-center gap-1 whitespace-nowrap rounded-sm border border-caution bg-caution/15 px-1.5 py-px font-mono text-[11.5px] text-caution">
            <AlertTriangle className="size-3" aria-hidden />
            Degraded
          </span>
          , never hidden.
        </p>
      </div>

      <div
        ref={containerRef}
        className="relative mt-14 flex flex-col items-center gap-6 md:flex-row md:items-center md:justify-between md:gap-3"
      >
        {EDGES.map(([from, to, stage]) => (
          <AnimatedBeam
            key={`${from}-${to}`}
            containerRef={containerRef}
            fromRef={refs[from]}
            toRef={refs[to]}
            pathColor="#2a3b49"
            pathOpacity={1}
            pathWidth={1.5}
            gradientStartColor="#4a8cff"
            gradientStopColor="#22c9c0"
            duration={3.2}
            delay={stage * 0.45}
            repeatDelay={1.2}
          />
        ))}

        <ol className="contents">
          {STAGES.map((stage) => (
            <li key={stage.key} className="contents">
              <Node nodeRef={refs[stage.key]} label={stage.label} explain={stage.explain} />
            </li>
          ))}

          <li className="relative z-10 flex flex-row gap-2 md:flex-col md:gap-3">
            <span className="sr-only">Then one specialist, chosen by Route:</span>
            {SPECIALISTS.map((specialist) => (
              <Node
                key={specialist.key}
                nodeRef={refs[specialist.key]}
                label={specialist.label}
                compact
              />
            ))}
          </li>

          {CLOSING.map((stage) => (
            <li key={stage.key} className="contents">
              <Node nodeRef={refs[stage.key]} label={stage.label} explain={stage.explain} />
            </li>
          ))}
        </ol>
      </div>

      <p className="mt-12 max-w-2xl text-[14px] leading-relaxed text-muted-foreground">
        If the imagery can&rsquo;t answer the question, say a change question with only one
        image, the run stops at Validate and tells you what is missing.
      </p>
    </section>
  );
}

function Node({
  nodeRef,
  label,
  explain,
  compact = false,
}: {
  nodeRef: RefObject<HTMLDivElement | null>;
  label: string;
  explain?: string;
  compact?: boolean;
}) {
  return (
    <div
      ref={nodeRef}
      className={cn(
        "relative z-10 rounded-sm border border-rule-strong bg-panel text-center",
        compact ? "w-[6.5rem] px-2 py-2 md:w-32" : "w-44 px-3 py-2.5 md:w-[7.5rem]",
      )}
    >
      <p className={cn("font-medium text-foreground", compact ? "text-[12px]" : "text-[13.5px]")}>
        {label}
      </p>
      {explain ? (
        <p className="mt-0.5 text-[11.5px] leading-snug text-muted-foreground">{explain}</p>
      ) : null}
    </div>
  );
}
