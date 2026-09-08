import type { Intent } from "./types";

/**
 * Starter questions, one per intent, shown in the console's command list.
 * They are prompts for the person, not canned answers — every one of them goes
 * to the controller like anything else typed by hand.
 */
export const SUGGESTED_QUERIES: { text: string; intent: Intent }[] = [
  { text: "How many aircraft are parked on the apron?", intent: "single_image_vqa" },
  { text: "Describe what this scene contains.", intent: "single_image_vqa" },
  { text: "Where are the storage tanks in this scene?", intent: "single_image_grounding" },
  { text: "What changed between the two captures?", intent: "change_vqa" },
  {
    text: "The optical scene is cloudy — use SAR backscatter to confirm the vessels.",
    intent: "optical_sar_fusion",
  },
];

/** Short codes for the intents, for the console list and result badges. */
export const INTENT_CODES: Record<Intent, string> = {
  single_image_vqa: "VQA",
  single_image_grounding: "GND",
  change_vqa: "CHG",
  optical_sar_fusion: "FUS",
};

export const INTENT_LABELS: Record<Intent, string> = {
  single_image_vqa: "Single-image VQA",
  single_image_grounding: "Grounding",
  change_vqa: "Change VQA",
  optical_sar_fusion: "Optical–SAR fusion",
};
