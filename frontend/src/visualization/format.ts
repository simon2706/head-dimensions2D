import type { Measurement } from '../types/api'

export type Unit = 'mm' | 'px'

export function formatValue(m: Pick<Measurement, 'value_mm' | 'value_px'>, unit: Unit): string {
  const v = unit === 'mm' ? m.value_mm : m.value_px
  if (v === null || v === undefined) return '—'
  return `${v.toFixed(1)} ${unit}`
}

export function fmt(v: number | null | undefined, digits = 2, suffix = ''): string {
  return v === null || v === undefined ? '—' : `${v.toFixed(digits)}${suffix}`
}
