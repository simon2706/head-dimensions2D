import { describe, expect, it } from 'vitest'
import { formatValue } from './format'
import { placeLabels } from './labels'

describe('placeLabels', () => {
  it('stacks labels that would overlap and keeps others at their anchor height', () => {
    const placed = placeLabels(
      [
        { id: 'b', anchor: [100, 205] },
        { id: 'a', anchor: [100, 200] },
        { id: 'c', anchor: [100, 400] },
      ],
      150,
      20,
    )
    expect(placed.map((p) => [p.id, p.y])).toEqual([
      ['a', 200],
      ['b', 220],
      ['c', 400],
    ])
    expect(placed.every((p) => p.x === 150)).toBe(true)
  })
})

describe('formatValue', () => {
  it('formats mm and px, and missing values', () => {
    expect(formatValue({ value_mm: 96.43, value_px: 342.4 }, 'mm')).toBe('96.4 mm')
    expect(formatValue({ value_mm: 96.43, value_px: 342.4 }, 'px')).toBe('342.4 px')
    expect(formatValue({ value_mm: null, value_px: null }, 'mm')).toBe('—')
  })
})
