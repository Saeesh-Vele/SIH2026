import type { Intent, UploadMode } from "./types";

/**
 * Starter questions per imagery layout. They fill the question box; nothing
 * is sent until the person runs it. Worded for what the models were tuned on —
 * land cover at Sentinel-2 resolution, where a pixel is 10 m — rather than
 * objects too small to see.
 */
export const SUGGESTED_QUESTIONS: Record<UploadMode, string[]> = {
  single: [
    "Describe this scene.",
    "Is this area mostly farmland or forest?",
    "Is there a river or lake in this image?",
    "Are there buildings or roads here?",
  ],
  bi_temporal: [
    "What changed between these two dates?",
    "Has any farmland become built-up land?",
    "Did the water area grow or shrink?",
  ],
  cross_modal: [
    "What does the radar show that the photo doesn't?",
    "Describe this area using both images.",
  ],
};

/** Short codes for the intents, for result badges. */
export const INTENT_CODES: Record<Intent, string> = {
  single_image_vqa: "VQA",
  single_image_captioning: "CAP",
  change_vqa: "CHG",
  optical_sar_fusion: "FUS",
};

export const INTENT_LABELS: Record<Intent, string> = {
  single_image_vqa: "Question",
  single_image_captioning: "Scene description",
  change_vqa: "Change",
  optical_sar_fusion: "Optical + SAR",
};
