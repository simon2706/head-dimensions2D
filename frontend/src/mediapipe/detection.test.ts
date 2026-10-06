import type { FaceLandmarkerResult } from '@mediapipe/tasks-vision'
import { describe, expect, it } from 'vitest'
import { buildMeasureRequest, extractDetection, landmarkBBox, toColumnMajor, toPixel } from './detection'

// Rotation about z by 90° with translation (1, 2, -50).
const ROW_MAJOR = [0, -1, 0, 1, 1, 0, 0, 2, 0, 0, 1, -50, 0, 0, 0, 1]
const COLUMN_MAJOR = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 2, -50, 1]

function fakeResult(faces: number, n = 478): FaceLandmarkerResult {
  const face = Array.from({ length: n }, (_, i) => ({ x: i / n, y: 0.5, z: 0.01, visibility: 0 }))
  return {
    faceLandmarks: Array.from({ length: faces }, () => face),
    faceBlendshapes: [
      { categories: [{ categoryName: 'eyeBlinkLeft', score: 0.1, index: 0, displayName: '' }], headIndex: 0, headName: '' },
    ],
    facialTransformationMatrixes: [{ rows: 4, columns: 4, data: COLUMN_MAJOR }],
  }
}

describe('toColumnMajor', () => {
  it('keeps column-major data unchanged', () => {
    expect(toColumnMajor({ rows: 4, columns: 4, data: COLUMN_MAJOR })).toEqual(COLUMN_MAJOR)
  })
  it('transposes row-major data', () => {
    expect(toColumnMajor({ rows: 4, columns: 4, data: ROW_MAJOR })).toEqual(COLUMN_MAJOR)
  })
  it('rejects wrong shapes', () => {
    expect(() => toColumnMajor({ rows: 3, columns: 3, data: [1, 0, 0, 0, 1, 0, 0, 0, 1] })).toThrow()
  })
})

describe('extractDetection', () => {
  it('returns null without faces', () => {
    expect(extractDetection(fakeResult(0), 100, 100)).toBeNull()
  })
  it('keeps the first face, counts faces and flattens blendshapes', () => {
    const det = extractDetection(fakeResult(2), 640, 480)!
    expect(det.faceCount).toBe(2)
    expect(det.landmarks).toHaveLength(478)
    expect(det.landmarks[0]).toEqual({ x: 0, y: 0.5, z: 0.01 })
    expect(det.blendshapes).toEqual({ eyeBlinkLeft: 0.1 })
    expect(det.transformationMatrix).toEqual(COLUMN_MAJOR)
  })
  it('requires iris landmarks', () => {
    expect(() => extractDetection(fakeResult(1, 468), 100, 100)).toThrow(/478/)
  })
})

describe('request building and pixel helpers', () => {
  it('builds the backend request', () => {
    const det = extractDetection(fakeResult(1), 1920, 1080)!
    const req = buildMeasureRequest(det, { assumed_iris_diameter_mm: 12 }, null)
    expect(req.image_width).toBe(1920)
    expect(req.face_count).toBe(1)
    expect(req.config).toEqual({ assumed_iris_diameter_mm: 12 })
    expect(req.facial_transformation_matrix).toHaveLength(16)
    expect(req.iris_model).toBeNull()
  })
  it('sends iris-model points and ROIs but not the eye contour', () => {
    const det = extractDetection(fakeResult(1), 1920, 1080)!
    const eye = (cx: number) => ({
      iris: Array.from({ length: 5 }, (_, i) => ({ x: cx + i / 1000, y: 0.4, z: 0 })),
      eyeContour: Array.from({ length: 71 }, () => ({ x: cx, y: 0.4, z: 0 })),
      roi: { cx: cx * 1920, cy: 432, size: 150, rotation: 0.01 },
    })
    const req = buildMeasureRequest({ ...det, irisModel: { right: eye(0.45), left: eye(0.55) } }, {}, null)
    expect(req.iris_model?.model).toBe('iris_landmark.tflite')
    expect(req.iris_model?.right.iris).toHaveLength(5)
    expect(req.iris_model?.left.iris[0].x).toBeCloseTo(0.55)
    expect(req.iris_model?.right.roi).toEqual({ cx: 0.45 * 1920, cy: 432, size: 150, rotation: 0.01 })
    expect(req.iris_model?.right).not.toHaveProperty('eyeContour')
  })
  it('converts normalized landmarks to pixels', () => {
    expect(toPixel({ x: 0.5, y: 0.25 }, 1920, 1080)).toEqual([960, 270])
    expect(landmarkBBox([{ x: 0.1, y: 0.2, z: 0 }, { x: 0.3, y: 0.6, z: 0 }], 100, 200)).toEqual({
      x: 10,
      y: 40,
      width: 20,
      height: 80,
    })
  })
})
