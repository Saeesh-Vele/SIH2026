"use client";

import { useId } from "react";
import { Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SUGGESTED_QUESTIONS } from "@/lib/suggestions";
import { cn } from "@/lib/utils";
import type { UploadMode } from "@/lib/types";

/**
 * Step 2. The question is controlled from the page so a sample can fill it in;
 * suggestion chips fill it too, but only the Run button (or Enter) sends it.
 */
export function QuestionBox({
  value,
  onChange,
  onRun,
  mode,
  disabled,
  busy,
}: {
  value: string;
  onChange: (value: string) => void;
  onRun: (question: string) => void;
  /** The bound imagery's layout, which decides the suggestions. */
  mode: UploadMode;
  /** No imagery yet. */
  disabled: boolean;
  busy: boolean;
}) {
  const id = useId();
  const canRun = !disabled && !busy && value.trim().length > 0;

  function run() {
    if (canRun) onRun(value.trim());
  }

  return (
    <div className="space-y-3">
      <label htmlFor={id} className="sr-only">
        Your question
      </label>
      <textarea
        id={id}
        value={value}
        disabled={disabled}
        rows={2}
        placeholder={disabled ? "Choose imagery first" : "Ask about the imagery in plain English"}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            run();
          }
        }}
        className="block w-full resize-none rounded-sm border border-rule-strong bg-raised px-3 py-2.5 text-[14px] leading-relaxed text-foreground placeholder:text-muted-foreground focus:border-signal focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
      />

      {!disabled ? (
        <div>
          <p className="mb-1.5 text-[12px] text-muted-foreground">Suggestions</p>
          <ul className="flex flex-wrap gap-1.5">
            {SUGGESTED_QUESTIONS[mode].map((question) => (
              <li key={question}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onChange(question)}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-left text-[12px] transition-colors disabled:opacity-50",
                    value === question
                      ? "border-signal bg-signal-deep/50 text-signal"
                      : "border-rule-strong text-text-dim hover:border-signal hover:text-foreground",
                  )}
                >
                  {question}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="flex items-center gap-3">
        <Button
          type="button"
          data-tour="run"
          onClick={run}
          disabled={!canRun}
          className={cn(
            "h-9 gap-2 rounded-sm px-4 text-[13px] font-semibold disabled:opacity-100",
            canRun
              ? "bg-accent-gradient text-void hover:brightness-110"
              : "border border-rule-strong bg-raised text-text-dim",
          )}
        >
          <Play className="size-3.5 fill-current" aria-hidden />
          {busy ? "Running…" : "Run"}
        </Button>
        <span className="text-[12px] text-muted-foreground">
          {disabled ? "Step 1 comes first." : busy ? "Working on it." : "Enter also runs it."}
        </span>
      </div>
    </div>
  );
}
