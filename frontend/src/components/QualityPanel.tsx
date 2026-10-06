import type { Pose, QualityReport } from '../types/api'
import { fmt } from '../visualization/format'
import { STATUS_ICON } from '../visualization/style'

const OVERALL_TEXT = { pass: 'GOOD', warning: 'WARNING', fail: 'FAIL', skipped: '—' } as const

export function QualityPanel({ quality, pose }: { quality: QualityReport; pose: Pose }) {
  return (
    <section className="panel">
      <h2>Capture quality</h2>
      <ul className="checks">
        {quality.checks.map((c) => (
          <li key={c.id} className={`status-${c.status}`} title={c.message || undefined}>
            <span className="icon">{STATUS_ICON[c.status]}</span>
            <span className="check-label">{c.label}</span>
            <span className="check-value">
              {c.value !== null && c.id !== 'face_count' ? `${fmt(c.value, c.unit === '°' || c.unit === '%' ? 1 : 2)}${c.unit === '°' || c.unit === '%' ? c.unit : ' ' + c.unit}` : ''}
            </span>
            {c.message && c.status !== 'pass' && <span className="check-message">{c.message}</span>}
          </li>
        ))}
      </ul>
      <div className={`overall status-${quality.overall}`}>
        Overall: <strong>{OVERALL_TEXT[quality.overall]}</strong>
      </div>
      {quality.overall === 'fail' && (
        <p className="warning-text">
          This capture failed at least one quality check. Measurements are shown for inspection only
          and are marked as poor — retake the photo.
        </p>
      )}
      <p className="muted small">
        Pose source: {pose.source}. Landmark roll cross-check: {fmt(pose.landmark_roll_deg, 1, '°')}
      </p>
    </section>
  )
}
