// Conversion of MediaPipe results into the backend request format. Pure
// functions so they can be unit-tested without MediaPipe.
import type { FaceLandmarkerResult, Matrix } from '@mediapipe/tasks-vision'
import type { ImageMetrics, IrisModelPayload, LandmarkPoint, MeasureRequest, MeasurementConfig } from '../types/api'
import type { IrisEyeResult, IrisModelResult } from './irisLandmark'

export const EXPECTED_LANDMARKS = 478

/** The subset of a detection that the measurement engine needs (and that is exported). */
export interface FaceDetection {
  imageWidth: number
  imageHeight: number
  faceCount: number
  landmarks: LandmarkPoint[]
  /** 16 values, column-major. */
  transformationMatrix: number[] | null
  blendshapes: Record<string, number>
  /** MediaPipe Iris landmark model output; null when it has not run or failed. */
  irisModel: IrisModelResult | null
}

/**
 * Return the 4x4 matrix as 16 column-major values.
 *
 * `Matrix.data` is documented only as "flattened". A rigid transform's last row
 * is (0, 0, 0, 1); its position in the flat array tells the layout apart.
 */
export function toColumnMajor(m: Pick<Matrix, 'rows' | 'columns' | 'data'>): number[] {
  const d = Array.from(m.data)
  if (m.rows !== 4 || m.columns !== 4 || d.length !== 16) {
    throw new Error(`Unexpected transformation matrix shape ${m.rows}x${m.columns}`)
  }
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-6
  const lastRowColumnMajor = near(d[3], 0) && near(d[7], 0) && near(d[11], 0) && near(d[15], 1)
  if (lastRowColumnMajor) return d
  const lastRowRowMajor = near(d[12], 0) && near(d[13], 0) && near(d[14], 0) && near(d[15], 1)
  if (lastRowRowMajor) {
    const out = new Array<number>(16)
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = d[r * 4 + c]
    return out
  }
  return d
}

export function extractDetection(
  result: FaceLandmarkerResult,
  imageWidth: number,
  imageHeight: number,
): FaceDetection | null {
  const faces = result.faceLandmarks ?? []
  if (faces.length === 0) return null
  const landmarks = faces[0].map(({ x, y, z }) => ({ x, y, z }))
  if (landmarks.length !== EXPECTED_LANDMARKS) {
    throw new Error(
      `Expected ${EXPECTED_LANDMARKS} landmarks (with iris), got ${landmarks.length}. ` +
        'Is the model a Face Landmarker model with iris refinement?',
    )
  }
  const matrix = result.facialTransformationMatrixes?.[0]
  const blendshapes: Record<string, number> = {}
  for (const c of result.faceBlendshapes?.[0]?.categories ?? []) blendshapes[c.categoryName] = c.score
  return {
    imageWidth,
    imageHeight,
    faceCount: faces.length,
    landmarks,
    transformationMatrix: matrix ? toColumnMajor(matrix) : null,
    blendshapes,
    irisModel: null,
  }
}

export function buildMeasureRequest(
  det: FaceDetection,
  config: MeasurementConfig,
  imageMetrics: ImageMetrics | null,
): MeasureRequest {
  return {
    image_width: det.imageWidth,
    image_height: det.imageHeight,
    landmarks: det.landmarks,
    face_count: det.faceCount,
    facial_transformation_matrix: det.transformationMatrix,
    blendshapes: det.blendshapes,
    image_metrics: imageMetrics,
    iris_model: det.irisModel ? irisModelPayload(det.irisModel) : null,
    config,
  }
}

/** Only the iris points (and the crop, for debugging) go to the backend. */
export function irisModelPayload(r: IrisModelResult): IrisModelPayload {
  const eye = (e: IrisEyeResult) => ({ iris: e.iris, roi: { ...e.roi } })
  return { model: 'iris_landmark.tflite', right: eye(r.right), left: eye(r.left) }
}

/** Normalized landmark -> image pixel coordinates (same convention as the backend). */
export function toPixel(p: { x: number; y: number }, width: number, height: number): [number, number] {
  return [p.x * width, p.y * height]
}

export function landmarkBBox(landmarks: LandmarkPoint[], width: number, height: number) {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const p of landmarks) {
    const [x, y] = toPixel(p, width, height)
    x0 = Math.min(x0, x)
    y0 = Math.min(y0, y)
    x1 = Math.max(x1, x)
    y1 = Math.max(y1, y)
  }
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}
