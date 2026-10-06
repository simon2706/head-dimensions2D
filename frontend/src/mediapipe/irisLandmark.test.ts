import { describe, expect, it } from 'vitest'
import { eyeRoi, INPUT_SIZE, projectToImage, rgbaToInput, roiToInputTransform, ROI_SCALE, type EyeRoi } from './irisLandmark'

const W = 1000
const H = 800

function landmarksWith(points: Record<number, [number, number]>) {
  const lms = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }))
  for (const [id, [x, y]] of Object.entries(points)) lms[Number(id)] = { x: x / W, y: y / H }
  return lms
}

/** Apply a setTransform-style affine to an image point. */
function apply([a, b, c, d, e, f]: number[], [x, y]: [number, number]): [number, number] {
  return [a * x + c * y + e, b * x + d * y + f]
}

/** Image px -> raw model output (64-px units), via the crop transform. */
function toRaw(roi: EyeRoi, flip: boolean, pts: [number, number][]): number[] {
  const t = roiToInputTransform(roi, flip)
  return pts.flatMap((p) => [...apply(t, p), 0])
}

describe('eyeRoi', () => {
  it('centres on the corners and scales the long side by 2.3', () => {
    const roi = eyeRoi(landmarksWith({ 33: [400, 300], 133: [460, 300] }), [33, 133], W, H)
    expect(roi.cx).toBeCloseTo(430)
    expect(roi.cy).toBeCloseTo(300)
    expect(roi.size).toBeCloseTo(ROI_SCALE * 60)
    expect(roi.rotation).toBeCloseTo(0)
  })

  it('follows in-plane roll of the corner line (y down)', () => {
    const a = (15 * Math.PI) / 180
    const lms = landmarksWith({ 362: [500, 300], 263: [500 + 60 * Math.cos(a), 300 + 60 * Math.sin(a)] })
    const roi = eyeRoi(lms, [362, 263], W, H)
    expect(roi.rotation).toBeCloseTo(a)
    // square_long uses the axis-aligned bounding box of the two corners.
    expect(roi.size).toBeCloseTo(ROI_SCALE * 60 * Math.cos(a))
  })
})

describe('roiToInputTransform', () => {
  const roi: EyeRoi = { cx: 430, cy: 300, size: 128, rotation: 0.3 }

  it('maps the ROI centre to the input centre', () => {
    for (const flip of [false, true]) {
      const [x, y] = apply(roiToInputTransform(roi, flip), [roi.cx, roi.cy])
      expect(x).toBeCloseTo(INPUT_SIZE / 2)
      expect(y).toBeCloseTo(INPUT_SIZE / 2)
    }
  })

  it('maps a point along the corner direction to the input x axis, mirrored when flipped', () => {
    const p: [number, number] = [roi.cx + 32 * Math.cos(roi.rotation), roi.cy + 32 * Math.sin(roi.rotation)]
    const [x, y] = apply(roiToInputTransform(roi, false), p)
    expect(x).toBeCloseTo(32 + 16) // 32 px of a 128 px ROI = quarter of 64
    expect(y).toBeCloseTo(32)
    const [xf] = apply(roiToInputTransform(roi, true), p)
    expect(xf).toBeCloseTo(32 - 16)
  })
})

describe('projectToImage', () => {
  it('maps the input centre back to the ROI centre (normalized)', () => {
    const roi: EyeRoi = { cx: 430, cy: 300, size: 128, rotation: 0 }
    const [p] = projectToImage([32, 32, 0], 1, roi, false, W, H)
    expect(p.x).toBeCloseTo(430 / W)
    expect(p.y).toBeCloseTo(300 / H)
  })

  it('inverts the crop transform for rotated and flipped ROIs', () => {
    const pts: [number, number][] = [
      [420, 290],
      [445, 310],
      [430, 300],
    ]
    for (const flip of [false, true]) {
      const roi: EyeRoi = { cx: 430, cy: 300, size: 140, rotation: -0.25 }
      const back = projectToImage(toRaw(roi, flip, pts), pts.length, roi, flip, W, H)
      back.forEach((p, i) => {
        expect(p.x * W).toBeCloseTo(pts[i][0])
        expect(p.y * H).toBeCloseTo(pts[i][1])
      })
    }
  })

  it('un-flips x for the mirrored eye', () => {
    const roi: EyeRoi = { cx: 500, cy: 300, size: 64, rotation: 0 }
    const [p] = projectToImage([48, 32, 0], 1, roi, true, W, H)
    expect(p.x * W).toBeCloseTo(500 - 16)
  })
})

describe('rgbaToInput', () => {
  it('drops alpha and scales to [0, 1]', () => {
    const out = rgbaToInput(new Uint8ClampedArray([255, 0, 51, 255, 0, 255, 0, 0]))
    expect(Array.from(out)).toEqual(Array.from(new Float32Array([1, 0, 0.2, 0, 1, 0])))
  })
})
