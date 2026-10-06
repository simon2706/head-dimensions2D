// Local, in-browser MediaPipe Iris landmark model (iris_landmark.tflite), run with
// LiteRT.js. Reproduces mediapipe/modules/iris_landmark: a rotated square crop
// around each eye (from two Face Landmarker eye corners) is resized to 64x64 and
// the model returns 5 iris + 71 eye-contour points, projected back to the image.
// Like the Face Landmarker, model and WASM are served from /public.
import { loadAndCompile, loadLiteRt, Tensor, type CompiledModel } from '@litertjs/core'
import type { LandmarkPoint } from '../types/api'

export const IRIS_MODEL_PATH = '/models/iris_landmark.tflite'
export const LITERT_WASM_PATH = '/litert/wasm/'
export const IRIS_MODEL_DESCRIPTION = 'iris_landmark.tflite (MediaPipe Iris landmark model, LiteRT.js)'

export const INPUT_SIZE = 64
export const NUM_IRIS = 5
export const NUM_EYE_CONTOUR = 71
/** iris_landmark_landmarks_to_roi.pbtxt: RectTransformationCalculator scale_x/scale_y. */
export const ROI_SCALE = 2.3
/**
 * Eye corners (start -> end) defining each ROI, from graphs/iris_tracking/iris_tracking_cpu.pbtxt
 * (see backend topology.py). The model was trained on the subject's right eye (image left);
 * the subject's left eye is flipped horizontally.
 */
interface EyeSpec {
  corners: readonly [number, number]
  flip: boolean
}
export const EYES: { right: EyeSpec; left: EyeSpec } = {
  right: { corners: [33, 133], flip: false },
  left: { corners: [362, 263], flip: true },
}

/** Rotated square crop in image pixels; `rotation` (rad) is the start->end corner angle, y down. */
export interface EyeRoi {
  cx: number
  cy: number
  size: number
  rotation: number
}

export interface IrisEyeResult {
  /** Iris centre + 4 contour points, normalized to the full image. */
  iris: LandmarkPoint[]
  /** 71 eye-contour (and brow) points, normalized to the full image. */
  eyeContour: LandmarkPoint[]
  roi: EyeRoi
}

export interface IrisModelResult {
  /** Subject's right eye. */
  right: IrisEyeResult
  /** Subject's left eye. */
  left: IrisEyeResult
}

/**
 * ROI as in MediaPipe's IrisLandmarkLandmarksToRoi: centre of the corners' bounding box,
 * rotation of the start->end vector (target angle 0), side = 2.3 x the long side of the
 * axis-aligned bounding box (square_long).
 */
export function eyeRoi(
  landmarks: { x: number; y: number }[],
  [startId, endId]: readonly [number, number],
  width: number,
  height: number,
): EyeRoi {
  const x0 = landmarks[startId].x * width
  const y0 = landmarks[startId].y * height
  const x1 = landmarks[endId].x * width
  const y1 = landmarks[endId].y * height
  return {
    cx: (x0 + x1) / 2,
    cy: (y0 + y1) / 2,
    size: ROI_SCALE * Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)),
    rotation: Math.atan2(y1 - y0, x1 - x0),
  }
}

/**
 * 2D affine [a, b, c, d, e, f] (CanvasRenderingContext2D.setTransform order) mapping image
 * pixels into the 64x64 model input, mirrored horizontally when `flip` is set.
 */
export function roiToInputTransform(roi: EyeRoi, flip: boolean): [number, number, number, number, number, number] {
  const k = INPUT_SIZE / roi.size
  const cos = Math.cos(roi.rotation)
  const sin = Math.sin(roi.rotation)
  const sx = flip ? -k : k
  // input = T(32,32) · S(sx,k) · R(-rotation) · T(-cx,-cy) · image
  const a = sx * cos
  const b = -k * sin
  const c = sx * sin
  const d = k * cos
  const half = INPUT_SIZE / 2
  return [a, b, c, d, half - a * roi.cx - c * roi.cy, half - b * roi.cx - d * roi.cy]
}

/**
 * Model output (x, y, z in 64-px input units) -> landmarks normalized to the full image,
 * as TfLiteTensorsToLandmarks (with flip) + LandmarkProjectionCalculator do.
 */
export function projectToImage(
  raw: ArrayLike<number>,
  count: number,
  roi: EyeRoi,
  flip: boolean,
  width: number,
  height: number,
): LandmarkPoint[] {
  const cos = Math.cos(roi.rotation)
  const sin = Math.sin(roi.rotation)
  const out: LandmarkPoint[] = []
  for (let i = 0; i < count; i++) {
    const rx = raw[3 * i] / INPUT_SIZE
    const u = (flip ? 1 - rx : rx) - 0.5
    const v = raw[3 * i + 1] / INPUT_SIZE - 0.5
    out.push({
      x: (roi.cx + roi.size * (cos * u - sin * v)) / width,
      y: (roi.cy + roi.size * (sin * u + cos * v)) / height,
      // Relative depth, scaled like x (not metric; never used for measurements).
      z: ((raw[3 * i + 2] / INPUT_SIZE) * roi.size) / width,
    })
  }
  return out
}

/** RGBA bytes -> RGB float32 in [0, 1] (TfLiteConverterCalculator with zero_center: false). */
export function rgbaToInput(rgba: Uint8ClampedArray): Float32Array {
  const n = rgba.length / 4
  const out = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    out[3 * i] = rgba[4 * i] / 255
    out[3 * i + 1] = rgba[4 * i + 1] / 255
    out[3 * i + 2] = rgba[4 * i + 2] / 255
  }
  return out
}

let cropCanvas: HTMLCanvasElement | null = null

/** Rotated, scaled (and optionally mirrored) 64x64 eye crop as model input. */
export function cropEye(source: CanvasImageSource, roi: EyeRoi, flip: boolean): Float32Array {
  cropCanvas ??= document.createElement('canvas')
  cropCanvas.width = INPUT_SIZE
  cropCanvas.height = INPUT_SIZE
  const ctx = cropCanvas.getContext('2d', { willReadFrequently: true })!
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  // Outside the image: black (MediaPipe replicates the border; only matters for clipped faces).
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.setTransform(...roiToInputTransform(roi, flip))
  ctx.drawImage(source, 0, 0)
  return rgbaToInput(ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data)
}

let modelPromise: Promise<CompiledModel> | null = null

/** Lazily loaded LiteRT runtime + compiled iris model (WASM/CPU; the model is tiny). */
export function getIrisModel(): Promise<CompiledModel> {
  if (!modelPromise) {
    modelPromise = loadLiteRt(LITERT_WASM_PATH).then(() =>
      loadAndCompile(IRIS_MODEL_PATH, { accelerator: 'wasm' }),
    )
    modelPromise.catch(() => {
      modelPromise = null
    })
  }
  return modelPromise
}

// Runs are serialized: the still-image and webcam paths share one compiled model.
let queue: Promise<unknown> = Promise.resolve()

async function runEye(model: CompiledModel, input: Float32Array): Promise<{ iris: Float32Array; contour: Float32Array }> {
  const tensor = new Tensor(input, [1, INPUT_SIZE, INPUT_SIZE, 3])
  let outputs: Tensor[] = []
  try {
    outputs = await model.run([tensor])
    const arrays = await Promise.all(outputs.map(async (t) => Float32Array.from((await t.data()) as Float32Array)))
    // Outputs are identified by size: output_iris [1,15], output_eyes_contours_and_brows [1,213].
    const iris = arrays.find((a) => a.length === NUM_IRIS * 3)
    const contour = arrays.find((a) => a.length === NUM_EYE_CONTOUR * 3)
    if (!iris || !contour) throw new Error(`Unexpected iris model outputs: ${arrays.map((a) => a.length)}`)
    return { iris, contour }
  } finally {
    tensor.delete()
    outputs.forEach((t) => t.delete())
  }
}

/** Run the Iris landmark model on both eyes, using Face Landmarker eye corners for the ROIs. */
export function detectIris(
  source: CanvasImageSource,
  faceLandmarks: { x: number; y: number }[],
  width: number,
  height: number,
): Promise<IrisModelResult> {
  const run = async (): Promise<IrisModelResult> => {
    const model = await getIrisModel()
    const eye = async ({ corners, flip }: EyeSpec): Promise<IrisEyeResult> => {
      const roi = eyeRoi(faceLandmarks, corners, width, height)
      const { iris, contour } = await runEye(model, cropEye(source, roi, flip))
      return {
        iris: projectToImage(iris, NUM_IRIS, roi, flip, width, height),
        eyeContour: projectToImage(contour, NUM_EYE_CONTOUR, roi, flip, width, height),
        roi,
      }
    }
    return { right: await eye(EYES.right), left: await eye(EYES.left) }
  }
  const result = queue.then(run, run)
  queue = result.catch(() => undefined)
  return result
}

/** ``detectIris`` that logs and returns null on failure (the backend then falls back and warns). */
export async function detectIrisSafe(
  source: CanvasImageSource,
  faceLandmarks: { x: number; y: number }[],
  width: number,
  height: number,
): Promise<IrisModelResult | null> {
  try {
    return await detectIris(source, faceLandmarks, width, height)
  } catch (e) {
    console.warn('[iris] Iris landmark model failed', e)
    return null
  }
}
