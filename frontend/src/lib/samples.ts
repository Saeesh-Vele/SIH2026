/**
 * The sample the landing page's "Try a sample" opens. Phase 3 builds the
 * gallery from public/samples/manifest.json; this id must match an entry there.
 */
export const LANDING_SAMPLE_ID = "eurosat-single";

export function sampleHref(id: string = LANDING_SAMPLE_ID): string {
  return `/console?sample=${encodeURIComponent(id)}`;
}
