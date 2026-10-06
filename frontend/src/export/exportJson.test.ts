import { describe, expect, it } from 'vitest'
import type { FaceDetection } from '../mediapipe/detection'
import type { MeasureResponse } from '../types/api'
import { buildExport, exportFileName } from './exportJson'

describe('buildExport', () => {
  it('includes landmarks, matrix, blendshapes and results but no image data', () => {
    const det: FaceDetection = {
      imageWidth: 10,
      imageHeight: 20,
      faceCount: 1,
      landmarks: [{ x: 0.1, y: 0.2, z: 0 }],
      transformationMatrix: new Array(16).fill(0),
      blendshapes: { eyeBlinkLeft: 0.2 },
      irisModel: null,
    }
    const result = { engine_version: '0.1.0', config: {}, features: { feature_schema_version: '1.0' } } as unknown as MeasureResponse
    const out = buildExport(det, null, result, { kind: 'upload', fileName: 'a.jpg' }, 'model', new Date(0))
    expect(out.image).toEqual({ width: 10, height: 20 })
    expect(out.mediapipe.landmarks_normalized).toHaveLength(1)
    expect(out.mediapipe.blendshapes).toEqual({ eyeBlinkLeft: 0.2 })
    expect(out.iris_model).toBeNull()
    const pt = { x: 0.4, y: 0.4, z: 0 }
    const eye = { iris: [pt, pt, pt, pt, pt], eyeContour: new Array(71).fill(pt), roi: { cx: 4, cy: 8, size: 3, rotation: 0 } }
    const withIris = buildExport({ ...det, irisModel: { right: eye, left: eye } }, null, result, { kind: 'webcam' }, 'm')
    expect(withIris.iris_model?.right.iris_normalized).toHaveLength(5)
    expect(withIris.iris_model?.left.eye_contour_normalized).toHaveLength(71)
    expect(out.created_at).toBe('1970-01-01T00:00:00.000Z')
    expect(JSON.stringify(out)).not.toMatch(/data:image|base64/)
  })
  it('creates filesystem-safe file names', () => {
    expect(exportFileName(new Date(0))).toBe('facial-measurement-1970-01-01T00-00-00-000Z.json')
  })
})
