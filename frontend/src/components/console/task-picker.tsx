"use client";

import { MODE_LABELS, TASK_OPTIONS, type TaskChoice } from "@/lib/tasks";
import { cn } from "@/lib/utils";
import type { UploadMode } from "@/lib/types";

const MODES: UploadMode[] = ["single", "bi_temporal", "cross_modal"];

/**
 * What the person wants to do. A specific task fixes the imagery layout; "Let
 * SatQuery decide" keeps the layout choice open underneath.
 */
export function TaskPicker({
  task,
  mode,
  onTaskChange,
  onModeChange,
}: {
  task: TaskChoice;
  mode: UploadMode;
  onTaskChange: (task: TaskChoice) => void;
  onModeChange: (mode: UploadMode) => void;
}) {
  return (
    <div className="space-y-4">
      <fieldset>
        <legend className="mb-2 text-[13px] text-text-dim">What do you want to do?</legend>
        <div className="grid gap-1.5 sm:grid-cols-2">
          {TASK_OPTIONS.map((option) => (
            <label
              key={option.id}
              className={cn(
                "flex cursor-pointer gap-2.5 rounded-sm border p-2.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-signal",
                option.id === "auto" && "sm:col-span-2",
                task === option.id
                  ? "border-signal bg-signal-deep/40"
                  : "border-rule hover:border-rule-strong",
              )}
            >
              <input
                type="radio"
                name="task"
                value={option.id}
                checked={task === option.id}
                onChange={() => onTaskChange(option.id)}
                className="sr-only"
              />
              <span
                aria-hidden
                className={cn(
                  "mt-0.5 grid size-3.5 shrink-0 place-items-center rounded-full border",
                  task === option.id ? "border-signal" : "border-rule-strong",
                )}
              >
                {task === option.id ? <span className="size-1.5 rounded-full bg-signal" /> : null}
              </span>
              <span>
                <span className="block text-[13px] font-medium text-foreground">{option.label}</span>
                <span className="block text-[12px] leading-snug text-muted-foreground">
                  {option.explain}
                </span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {task === "auto" ? (
        <fieldset>
          <legend className="mb-2 text-[13px] text-text-dim">How many images?</legend>
          <div className="inline-flex rounded-sm border border-rule bg-raised p-0.5" role="radiogroup">
            {MODES.map((m) => (
              <label
                key={m}
                className={cn(
                  "cursor-pointer rounded-[1px] px-3 py-1.5 text-[12.5px] transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-signal",
                  mode === m ? "bg-signal-deep text-signal" : "text-text-dim hover:text-foreground",
                )}
              >
                <input
                  type="radio"
                  name="mode"
                  value={m}
                  checked={mode === m}
                  onChange={() => onModeChange(m)}
                  className="sr-only"
                />
                {MODE_LABELS[m]}
              </label>
            ))}
          </div>
        </fieldset>
      ) : null}
    </div>
  );
}
