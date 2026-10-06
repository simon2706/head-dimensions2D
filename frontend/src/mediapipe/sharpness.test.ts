import { describe, expect, it } from 'vitest'
import { laplacianVariance, rgbaToGray } from './sharpness'

function image(w: number, h: number, f: (x: number, y: number) => number): Float32Array {
  const g = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) g[y * w + x] = f(x, y)
  return g
}

describe('laplacianVariance', () => {
  it('is zero for flat and linear-gradient images', () => {
    expect(laplacianVariance(image(16, 16, () => 128), 16, 16)).toBe(0)
    expect(laplacianVariance(image(16, 16, (x) => 4 * x), 16, 16)).toBeCloseTo(0)
  })
  it('is larger for a sharp checkerboard than for a smooth one', () => {
    const sharp = laplacianVariance(image(32, 32, (x, y) => ((x + y) % 2) * 255), 32, 32)
    const smooth = laplacianVariance(image(32, 32, (x, y) => 128 + 20 * Math.sin((x + y) / 4)), 32, 32)
    expect(sharp).toBeGreaterThan(smooth * 100)
  })
})

describe('rgbaToGray', () => {
  it('uses luma weights', () => {
    const g = rgbaToGray(new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 255]), 2, 1)
    expect(g[0]).toBeCloseTo(76.245)
    expect(g[1]).toBeCloseTo(29.07)
  })
})
