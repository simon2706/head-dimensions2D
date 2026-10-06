// TypeScript mirrors of the backend Pydantic schemas (backend/src/facial_measurement/models).

export type CheckStatus = 'pass' | 'warning' | 'fail' | 'skipped'
export type MeasurementType = 'direct' | 'surface_proxy' | 'experimental_proxy'
export type MeasurementQuality = 'good' | 'warning' | 'poor'

export interface LandmarkPoint {
  x: number
  y: number
  z: number
}

/** Partial overrides are accepted by the backend; omitted keys use defaults. */
export type MeasurementConfig = Record<string, number | string>

export interface ImageMetrics {
  sharpness_laplacian_var: number | null
  sharpness_crop_width_px: number | null
}

export interface MeasureRequest {
  image_width: number
  image_height: number
  landmarks: LandmarkPoint[]
  face_count: number
  /** 16 values, column-major. */
  facial_transformation_matrix: number[] | null
  blendshapes: Record<string, number>
  image_metrics: ImageMetrics | null
  /** MediaPipe Iris landmark model output; null if it did not run. */
  iris_model: IrisModelPayload | null
  config: MeasurementConfig
}

export interface IrisModelEyePayload {
  /** Iris centre + 4 contour points, normalized to the full image. */
  iris: LandmarkPoint[]
  roi: Record<string, number> | null
}

export interface IrisModelPayload {
  model: string
  right: IrisModelEyePayload
  left: IrisModelEyePayload
}

export interface ScaleResult {
  method: string
  assumed_iris_diameter_mm: number | null
  left_iris_diameter_px: number | null
  right_iris_diameter_px: number | null
  left_iris_vertical_diameter_px: number | null
  right_iris_vertical_diameter_px: number | null
  left_mm_per_px: number | null
  right_mm_per_px: number | null
  final_mm_per_px: number
  difference_percent: number | null
  landmarks: Record<string, number[]>
  /** Iris diameter endpoints in image px, e.g. 'right_horizontal'. */
  endpoints_px: Record<string, [number, number][]>
  iris_centers_px: Record<string, [number, number]>
  fallback_reason: string | null
}

export interface Pose {
  yaw_deg: number | null
  pitch_deg: number | null
  roll_deg: number | null
  source: string
  landmark_roll_deg: number
  camera_yaw_deg: number | null
  camera_pitch_deg: number | null
  camera_roll_deg: number | null
  translation: number[] | null
  line_of_sight_offset_deg: number | null
}

export interface QualityCheck {
  id: string
  label: string
  status: CheckStatus
  value: number | null
  unit: string
  message: string
  thresholds: Record<string, number>
}

export interface QualityReport {
  overall: CheckStatus
  checks: QualityCheck[]
}

export interface Measurement {
  id: string
  name: string
  type: MeasurementType
  value_px: number | null
  value_mm: number | null
  landmarks: number[]
  endpoints_px: [number, number][]
  quality: MeasurementQuality
  quality_notes: string[]
  definition: string
  formula: string
  level_y_face_px: number | null
  details: Record<string, unknown>
  error: string | null
}

export type Features = { feature_schema_version: string; scale_method: string } & Record<
  string,
  number | string | null
>

export interface Debug {
  image_width: number
  image_height: number
  face_bbox_px: Record<string, number>
  face_size_px: Record<string, number>
  face_frame: Record<string, unknown>
  eye_opening_ratios: Record<string, number>
  landmark_coordinates_px: Record<string, [number, number]>
  sharpness_laplacian_var: number | null
}

export interface MeasureResponse {
  engine_version: string
  scale: ScaleResult
  pose: Pose
  quality: QualityReport
  measurements: Measurement[]
  features: Features
  config: MeasurementConfig
  debug: Debug
}
