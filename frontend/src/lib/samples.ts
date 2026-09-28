import type { AssetRole, StagedFile, UploadMode } from "./types";

/**
 * The sample gallery, driven by public/samples/manifest.json. A sample goes
 * through exactly the path a person's own upload does — fetched into File
 * objects, POSTed to /upload, run by the real models — so its result is as
 * real as any other.
 */
export interface SampleFile {
  role: AssetRole;
  path: string;
  label: string;
}

export interface Sample {
  id: string;
  /** "missing" marks a slot the gallery shows as not added yet. */
  status: "ready" | "missing";
  title: string;
  mode: UploadMode;
  files: SampleFile[];
  /** A true-colour PNG/JPEG of the sample, rendered as the models see it. */
  thumbnail: string | null;
  question: string;
  /** For a missing sample: what would need adding. */
  needs?: string;
  /** Acquisition dates (YYYY-MM-DD) of a before/after pair, earliest first. */
  dates?: { t0: string | null; t1: string | null };
  source: { name: string; url: string } | null;
  licence: string | null;
  attribution: string | null;
}

/** The sample the landing page's "Try a sample" opens. Must match a manifest id. */
export const LANDING_SAMPLE_ID = "eurosat-river";

export function sampleHref(id: string = LANDING_SAMPLE_ID): string {
  return `/console?sample=${encodeURIComponent(id)}`;
}

export async function loadSamples(signal?: AbortSignal): Promise<Sample[]> {
  const response = await fetch("/samples/manifest.json", { signal });
  if (!response.ok) throw new Error("The sample list could not be loaded.");
  const manifest = (await response.json()) as { samples: Sample[] };
  return manifest.samples;
}

/** PNG and JPEG need benchmark mode; GeoTIFF does not. */
export function needsBenchmarkMode(names: string[]): boolean {
  return names.some((name) => /\.(png|jpe?g)$/i.test(name));
}

/** The sample's files as staged uploads, in the role order the mode expects. */
export async function stageSample(sample: Sample, signal?: AbortSignal): Promise<StagedFile[]> {
  return Promise.all(
    sample.files.map(async (entry) => {
      const response = await fetch(entry.path, { signal });
      if (!response.ok) throw new Error(`Sample file ${entry.path} could not be loaded.`);
      const blob = await response.blob();
      const name = entry.path.split("/").pop() ?? `${entry.role}.tif`;
      const file = new File([blob], name, { type: blob.type });
      return { role: entry.role, name, sizeBytes: file.size, file };
    }),
  );
}
