import type { Intent, TaskType, UploadMode } from "./types";

/** Task names — kept in lockstep with backend/model_config.yaml. */
export const TASK_LABELS: Record<TaskType, string> = {
  vqa_grounding: "Q&A and captioning",
  change_detection: "Change detection",
  optical_sar_fusion: "Optical + SAR fusion",
};

/**
 * What the person picks in step 1. A specific choice fixes both the imagery
 * layout and the intent sent with the question; "auto" leaves the intent to
 * the classifier and lets them choose the layout.
 */
export type TaskChoice = "auto" | "vqa" | "caption" | "change" | "fusion";

export interface TaskOption {
  id: TaskChoice;
  label: string;
  explain: string;
  /** Null for "auto": the person picks how many images. */
  mode: UploadMode | null;
  /** Sent as the forced intent; null lets the classifier decide. */
  intent: Intent | null;
}

export const TASK_OPTIONS: TaskOption[] = [
  {
    id: "auto",
    label: "Let SatQuery decide",
    explain: "Ask in your own words; SatQuery works out which kind of question it is.",
    mode: null,
    intent: null,
  },
  {
    id: "vqa",
    label: "Ask a question",
    explain: "A direct question about one image.",
    mode: "single",
    intent: "single_image_vqa",
  },
  {
    id: "caption",
    label: "Describe the scene",
    explain: "A plain-English description of one image.",
    mode: "single",
    intent: "single_image_captioning",
  },
  {
    id: "change",
    label: "Compare two dates",
    explain: "What changed between two images of the same place.",
    mode: "bi_temporal",
    intent: "change_vqa",
  },
  {
    id: "fusion",
    label: "Optical + radar",
    explain: "Read an ordinary image and a radar image of the same place together.",
    mode: "cross_modal",
    intent: "optical_sar_fusion",
  },
];

export const MODE_LABELS: Record<UploadMode, string> = {
  single: "One image",
  bi_temporal: "Before + after",
  cross_modal: "Optical + SAR",
};

/** One line under the upload slots, in plain language. */
export const MODE_HINTS: Record<UploadMode, string> = {
  single: "One satellite image. Ask what it shows.",
  bi_temporal:
    "Two images of the same area on different dates, earliest first. They must cover the same ground, or every difference looks like change.",
  cross_modal:
    "An ordinary (optical) image and a radar (SAR) image of the same area. SAR is radar: it sees through cloud and works at night, but shows texture rather than colour.",
};

/** Roles each mode expects, in slot order. Mirrors MODE_ROLES in the backend. */
export const MODE_ROLES: Record<
  UploadMode,
  { role: string; label: string; noun: string; hint: string }[]
> = {
  single: [{ role: "primary", label: "Image", noun: "an image", hint: "GeoTIFF, or PNG/JPEG" }],
  bi_temporal: [
    { role: "t0", label: "Before", noun: "the before image", hint: "The earlier date" },
    { role: "t1", label: "After", noun: "the after image", hint: "The later date" },
  ],
  cross_modal: [
    { role: "optical", label: "Optical", noun: "the optical image", hint: "Sentinel-2, 13 bands" },
    { role: "sar", label: "SAR (radar)", noun: "the SAR image", hint: "Sentinel-1, VV + VH" },
  ],
};
