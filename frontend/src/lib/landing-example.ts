import raw from "@/content/landing-example.json";

/**
 * The recorded run the landing page replays in "Ask it anything".
 *
 * Only real output goes here — written by the Colab recording script from an
 * actual /api/query response. While `status` is "pending" the page says so
 * instead of showing an answer.
 */
export interface LandingExample {
  status: "pending" | "recorded";
  sample_id: string | null;
  question: string | null;
  answer: string | null;
  /** Null when not measured (a degraded run). */
  confidence: number | null;
  degraded: boolean | null;
  task_selected: string | null;
  models_used: string[];
  /** The specialist's trace step, as the API returned it. */
  trace_line: { label: string; detail: string | null; duration_ms: number | null } | null;
  recorded_at: string | null;
}

export const landingExample = raw as LandingExample;

export function isRecorded(
  example: LandingExample,
): example is LandingExample & { status: "recorded"; question: string; answer: string } {
  return example.status === "recorded" && Boolean(example.question && example.answer);
}
