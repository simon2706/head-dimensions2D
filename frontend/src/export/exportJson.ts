// Offline-dataset export. Contains landmarks, metadata and results — never the image.
import type { FaceDetection } from '../mediapipe/detection'
import { IRIS_MODEL_DESCRIPTION, type IrisEyeResult } from '../mediapipe/irisLandmark'
import type { ImageMetrics, MeasureResponse } from '../types/api'

export const EXPORT_SCHEMA_VERSION = '1.1' // 1.1: added iris_model

export interface CaptureSource {
  kind: 'upload' | 'webcam'
  fileName?: string
}

export function buildExport(
  detection: FaceDetection,
  imageMetrics: ImageMetrics | null,
  result: MeasureResponse,
  source: CaptureSource,
  modelDescription: string,
  createdAt: Date = new Date(),
) {
  return {
    export_schema_version: EXPORT_SCHEMA_VERSION,
    created_at: createdAt.toISOString(),
    disclaimer:
      'Experimental research prototype. Values are iris-scaled estimates and proxies, not medical ' +
      'or anthropometric measurements.',
    source: { kind: source.kind, file_name: source.fileName ?? null },
    image: { width: detection.imageWidth, height: detection.imageHeight },
    mediapipe: {
      model: modelDescription,
      face_count: detection.faceCount,
      landmark_count: detection.landmarks.length,
      landmarks_normalized: detection.landmarks,
      facial_transformation_matrix_column_major: detection.transformationMatrix,
      blendshapes: detection.blendshapes,
    },
    iris_model: detection.irisModel
      ? {
          model: IRIS_MODEL_DESCRIPTION,
          right: eyeExport(detection.irisModel.right),
          left: eyeExport(detection.irisModel.left),
        }
      : null,
    image_metrics: imageMetrics,
    engine_version: result.engine_version,
    config: result.config,
    scale: result.scale,
    pose: result.pose,
    quality: result.quality,
    measurements: result.measurements,
    features: result.features,
    debug: result.debug,
  }
}

function eyeExport(e: IrisEyeResult) {
  return { iris_normalized: e.iris, eye_contour_normalized: e.eyeContour, roi_px: e.roi }
}

export function downloadJson(data: unknown, fileName: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = fileName
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function exportFileName(date: Date = new Date()): string {
  return `facial-measurement-${date.toISOString().replace(/[:.]/g, '-')}.json`
}
