// Placement of measurement labels to the right of the face so that labels of
// near-horizontal lines at similar heights do not overlap.

export interface LabelInput {
  id: string
  /** Anchor point (right-most endpoint of the measurement line). */
  anchor: [number, number]
}

export interface PlacedLabel extends LabelInput {
  x: number
  y: number
}

export function placeLabels(items: LabelInput[], columnX: number, lineHeight: number): PlacedLabel[] {
  const sorted = [...items].sort((a, b) => a.anchor[1] - b.anchor[1])
  const placed: PlacedLabel[] = []
  let lastY = -Infinity
  for (const item of sorted) {
    const y = Math.max(item.anchor[1], lastY + lineHeight)
    placed.push({ ...item, x: columnX, y })
    lastY = y
  }
  return placed
}
