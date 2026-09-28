import type { TraceStep } from "./types";

/**
 * The controller's stages, in the order a run visits them, and the trace
 * labels each node writes (backend/app/agent/nodes/*). The specialist writes
 * one of several labels depending on the task, so it matches by prefix.
 * `label` fits under a seventh of the progress bar; `doing` is the sentence
 * shown above it while that stage runs.
 */
export const STAGES = [
  {
    key: "intake",
    label: "Files",
    doing: "Reading your files",
    match: (l: string) => l === "Intake",
  },
  {
    key: "intent",
    label: "Question",
    doing: "Working out what you’re asking",
    match: (l: string) => l === "Classify intent",
  },
  {
    key: "validate",
    label: "Check",
    doing: "Checking the imagery can answer it",
    match: (l: string) => l === "Validate inputs",
  },
  {
    key: "route",
    label: "Route",
    doing: "Picking the model",
    match: (l: string) => l === "Route task",
  },
  {
    key: "model",
    label: "Model",
    doing: "Running the model — the longest step",
    match: (l: string) => l.startsWith("Run "),
  },
  {
    key: "combine",
    label: "Answer",
    doing: "Shaping the answer",
    match: (l: string) => l === "Combine output",
  },
  {
    key: "trace",
    label: "Trace",
    doing: "Saving the trace",
    match: (l: string) => l === "Log trace",
  },
] as const;

export type StageKey = (typeof STAGES)[number]["key"];
export type StageState = "waiting" | "active" | "done" | "failed" | "skipped";

/**
 * Where each stage stands, from the steps streamed so far.
 *
 * Steps arrive as each node finishes, so the active stage is the first one
 * with no step yet. A stage passed over — the validator rejecting the imagery
 * sends the run straight to Combine — shows as skipped once a later stage has
 * reported.
 */
export function stageStates(steps: TraceStep[], running: boolean): Record<StageKey, StageState> {
  const reported = STAGES.map((stage) => {
    const step = [...steps].reverse().find((s) => stage.match(s.label));
    return step ? (step.status === "failed" ? "failed" : "done") : null;
  });
  const lastReported = reported.reduce<number>((last, r, i) => (r ? i : last), -1);

  const states = {} as Record<StageKey, StageState>;
  let activeAssigned = false;
  STAGES.forEach((stage, i) => {
    const r = reported[i];
    if (r) states[stage.key] = r;
    else if (i < lastReported) states[stage.key] = "skipped";
    else if (running && !activeAssigned) {
      states[stage.key] = "active";
      activeAssigned = true;
    } else states[stage.key] = "waiting";
  });
  return states;
}
