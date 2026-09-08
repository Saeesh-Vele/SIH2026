import type { TaskType, UploadMode } from "./types";

export const TASK_LABELS: Record<TaskType, string> = {
  vqa_grounding: "VQA + grounding",
  change_detection: "Change detection",
  optical_sar_fusion: "Optical–SAR fusion",
};

export const MODE_LABELS: Record<UploadMode, string> = {
  single: "Single image",
  cross_modal: "Optical + SAR",
  bi_temporal: "Bi-temporal",
};

export const MODE_HINTS: Record<UploadMode, string> = {
  single: "One scene. Ask questions about what it contains.",
  cross_modal: "Pair an optical scene with the SAR capture of the same footprint.",
  bi_temporal: "Two captures of one footprint, earliest first.",
};

/** Roles each mode expects, in slot order. Mirrors MODE_ROLES in the backend. */
export const MODE_ROLES: Record<UploadMode, { role: string; label: string }[]> = {
  single: [{ role: "primary", label: "Scene" }],
  cross_modal: [
    { role: "optical", label: "Optical" },
    { role: "sar", label: "SAR" },
  ],
  bi_temporal: [
    { role: "t0", label: "Earlier — T₀" },
    { role: "t1", label: "Later — T₁" },
  ],
};
