// Local, in-browser MediaPipe Face Landmarker. Model and WASM are served from
// /public, so no image or inference request ever leaves the machine.
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

export const MODEL_PATH = '/models/face_landmarker.task'
export const WASM_PATH = '/mediapipe/wasm'
export const MODEL_DESCRIPTION = 'face_landmarker.task (float16, MediaPipe Face Landmarker)'

type Mode = 'IMAGE' | 'VIDEO'
const cache = new Map<Mode, Promise<FaceLandmarker>>()

async function create(mode: Mode): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH)
  const options = {
    baseOptions: { modelAssetPath: MODEL_PATH },
    runningMode: mode,
    // 2 so that "exactly one face" can be checked; only the first face is measured.
    numFaces: 2,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  } as const
  try {
    return await FaceLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: 'GPU' },
    })
  } catch (err) {
    console.warn('[mediapipe] GPU delegate unavailable, falling back to CPU', err)
    return FaceLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: 'CPU' },
    })
  }
}

/** Lazily created landmarker per running mode (IMAGE for stills, VIDEO for the webcam). */
export function getFaceLandmarker(mode: Mode): Promise<FaceLandmarker> {
  let p = cache.get(mode)
  if (!p) {
    p = create(mode)
    p.catch(() => cache.delete(mode))
    cache.set(mode, p)
  }
  return p
}
