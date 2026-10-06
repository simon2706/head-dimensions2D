// Simple blur diagnostic: variance of the 4-neighbour Laplacian of the face crop,
// resampled to a fixed width so values are roughly comparable across resolutions.
import type { ImageMetrics } from '../types/api'

export const SHARPNESS_CROP_WIDTH = 256

export function rgbaToGray(rgba: Uint8ClampedArray, width: number, height: number): Float32Array {
  const gray = new Float32Array(width * height)
  for (let i = 0; i < width * height; i++) {
    gray[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]
  }
  return gray
}

export function laplacianVariance(gray: Float32Array, width: number, height: number): number {
  if (width < 3 || height < 3) return 0
  let sum = 0
  let sumSq = 0
  let n = 0
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x
      const v = gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width] - 4 * gray[i]
      sum += v
      sumSq += v * v
      n++
    }
  }
  const mean = sum / n
  return sumSq / n - mean * mean
}

export function faceSharpness(
  source: CanvasImageSource,
  bbox: { x: number; y: number; width: number; height: number },
): ImageMetrics {
  const w = SHARPNESS_CROP_WIDTH
  const h = Math.max(3, Math.round((bbox.height / Math.max(bbox.width, 1)) * w))
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return { sharpness_laplacian_var: null, sharpness_crop_width_px: null }
  ctx.drawImage(source, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, w, h)
  const { data } = ctx.getImageData(0, 0, w, h)
  return {
    sharpness_laplacian_var: laplacianVariance(rgbaToGray(data, w, h), w, h),
    sharpness_crop_width_px: w,
  }
}
