"use client";

import { Check, X } from "lucide-react";
import { STAGES, stageStates } from "@/lib/pipeline";
import { cn } from "@/lib/utils";
import type { TraceStep } from "@/lib/types";

/**
 * Which stage of the controller is working right now, from the streamed trace
 * steps. Tells a person the run is alive during the long model step, without
 * a spinner that says nothing.
 */
export function PipelineProgress({ steps, running }: { steps: TraceStep[]; running: boolean }) {
  const states = stageStates(steps, running);
  const active = STAGES.find((s) => states[s.key] === "active");

  return (
    <div>
      <p className="mb-2 text-[13px] text-text-dim" aria-live="polite">
        {active ? `${active.doing}…` : running ? "Starting…" : "Finished."}
      </p>
      <ol className="flex gap-1" aria-label="Progress through the pipeline">
        {STAGES.map((stage) => {
          const state = states[stage.key];
          return (
            <li key={stage.key} className="min-w-0 flex-1">
              <span
                className={cn(
                  "block h-1 rounded-full",
                  state === "done" && "bg-signal",
                  state === "active" && "animate-signal-pulse bg-uplink",
                  state === "failed" && "bg-alarm",
                  state === "skipped" && "bg-rule",
                  state === "waiting" && "bg-rule-strong/60",
                )}
              />
              <span
                className={cn(
                  "mt-1.5 flex items-center gap-1 truncate text-[10.5px]",
                  state === "active" ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {state === "done" ? <Check className="size-3 shrink-0 text-signal" aria-hidden /> : null}
                {state === "failed" ? <X className="size-3 shrink-0 text-alarm" aria-hidden /> : null}
                <span className="truncate">{stage.label}</span>
                <span className="sr-only">
                  {state === "skipped" ? " (skipped)" : state === "waiting" ? " (waiting)" : ` (${state})`}
                </span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
