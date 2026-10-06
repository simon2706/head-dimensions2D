import type { CheckStatus, MeasurementQuality, MeasurementType } from '../types/api'

export const TYPE_STYLE: Record<MeasurementType, { color: string; dash: string; label: string }> = {
  direct: { color: '#14b8a6', dash: '', label: 'direct' },
  surface_proxy: { color: '#f59e0b', dash: '9 5', label: 'surface proxy' },
  experimental_proxy: { color: '#e879f9', dash: '2 4', label: 'experimental proxy' },
}

export const STATUS_ICON: Record<CheckStatus, string> = {
  pass: '✓',
  warning: '⚠',
  fail: '✗',
  skipped: '–',
}

export const QUALITY_ICON: Record<MeasurementQuality, string> = {
  good: '✓',
  warning: '⚠',
  poor: '✗',
}

export const LAYER_COLORS = {
  mesh: 'rgba(200, 220, 255, 0.35)',
  eyes: '#60a5fa',
  iris: '#38bdf8',
  irisModel: '#c084fc',
  eyebrows: '#a3e635',
  contour: '#f87171',
  ids: '#fde68a',
}
