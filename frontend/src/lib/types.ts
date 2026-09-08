/** Task names — kept in lockstep with backend/model_config.yaml. */
export type TaskType = "vqa_grounding" | "change_detection" | "optical_sar_fusion";

/** What the classifier decided the question is asking for. Mirrors app.models.schemas.Intent. */
export type Intent =
  | "single_image_vqa"
  | "single_image_captioning"
  | "change_vqa"
  | "optical_sar_fusion";

/** How a graph run ended. Mirrors app.models.schemas.QueryStatus. */
export type QueryStatus = "ok" | "rejected" | "unavailable" | "failed";

export type UploadMode = "single" | "cross_modal" | "bi_temporal";

export type AssetRole = "primary" | "optical" | "sar" | "t0" | "t1";

export type TraceStepStatus = "pending" | "running" | "complete" | "failed";

/** One controller step, exactly as the backend serialises it. */
export interface TraceStep {
  label: string;
  status: TraceStepStatus;
  detail: string | null;
  duration_ms: number | null;
  timestamp: string;
}

/** A box the canvas draws over the scene. Coordinates are 0–1 of scene extent. */
export interface BoxOverlay {
  kind: "box";
  id: string;
  label: string;
  confidence: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A change mask, expressed as normalised polygon rings. */
export interface MaskOverlay {
  kind: "mask";
  id: string;
  label: string;
  confidence: number;
  polygon: [number, number][];
  /** Share of the scene the region covers, 0–1. Set by change detection. */
  area?: number;
}

export type Overlay = BoxOverlay | MaskOverlay;

/**
 * What we actually know about a bound scene.
 *
 * Everything georeferenced is nullable and null today: /upload stores the file
 * but does not yet read GeoTIFF headers, so the readout shows a dash rather
 * than a number nobody measured.
 */
export interface SceneMeta {
  /** The filename, until a scene id can be read out of the header. */
  sceneId: string;
  role: AssetRole;
  sizeBytes: number;
  contentType: string | null;
  sensor: string | null;
  acquired: string | null;
  /** Decimal degrees, WGS84. */
  lat: number | null;
  lon: number | null;
  /** Ground sample distance, metres per pixel. */
  gsd: number | null;
  epsg: string | null;
  bands: string[] | null;
}

export interface QueryResult {
  queryId: string;
  traceId: string;
  status: QueryStatus;
  intent: Intent | null;
  taskSelected: TaskType;
  answer: string;
  confidence: number;
  overlays: Overlay[];
  modelsUsed: string[];
  steps: TraceStep[];
  /** Task-specific numbers rendered under the answer. */
  metrics: { label: string; value: string }[];
  /** Set when status is not "ok" — why the run produced no answer. */
  error: string | null;
}

export interface StagedFile {
  role: AssetRole;
  name: string;
  sizeBytes: number;
  /** Held so the bind step can POST the bytes and preview the image. */
  file: File;
}

/** One asset as the backend stored it. */
export interface UploadedAsset {
  asset_id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  role: AssetRole;
  stored_path: string;
}

export interface UploadResult {
  upload_id: string;
  mode: UploadMode;
  benchmark_mode: boolean;
  assets: UploadedAsset[];
  created_at: string;
}
