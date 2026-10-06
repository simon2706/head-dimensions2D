import type { Measurement, MeasurementType } from '../types/api'
import { formatValue, type Unit } from '../visualization/format'
import { QUALITY_ICON, TYPE_STYLE } from '../visualization/style'

interface Props {
  measurements: Measurement[]
  unit: Unit
  onUnitChange: (u: Unit) => void
  selectedId: string | null
  onSelect: (id: string | null) => void
}

const GROUPS: { type: MeasurementType; title: string }[] = [
  { type: 'direct', title: 'Direct' },
  { type: 'surface_proxy', title: 'Surface proxies' },
  { type: 'experimental_proxy', title: 'Experimental proxies' },
]

export function MeasurementsPanel({ measurements, unit, onUnitChange, selectedId, onSelect }: Props) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Measurements</h2>
        <div className="segmented">
          {(['mm', 'px'] as Unit[]).map((u) => (
            <button key={u} className={u === unit ? 'active' : ''} onClick={() => onUnitChange(u)}>
              {u}
            </button>
          ))}
        </div>
      </div>
      {GROUPS.map(({ type, title }) => {
        const rows = measurements.filter((m) => m.type === type)
        if (rows.length === 0) return null
        const style = TYPE_STYLE[type]
        return (
          <div key={type} className="measure-group">
            <h3>
              <svg width="28" height="8" aria-hidden>
                <line x1="0" y1="4" x2="28" y2="4" stroke={style.color} strokeWidth="2" strokeDasharray={style.dash} />
              </svg>
              {title}
            </h3>
            <ul className="measurements">
              {rows.map((m) => {
                const sel = m.id === selectedId
                return (
                  <li key={m.id} className={sel ? 'selected' : ''}>
                    <button className="measure-row" onClick={() => onSelect(sel ? null : m.id)}>
                      <span className={`q q-${m.quality}`} title={`quality: ${m.quality}`}>
                        {QUALITY_ICON[m.quality]}
                      </span>
                      <span className="measure-name">{m.name}</span>
                      <span className="measure-value">{formatValue(m, unit)}</span>
                    </button>
                    {sel && (
                      <div className="measure-detail">
                        <div>
                          <code>{m.id}</code> · {TYPE_STYLE[m.type].label}
                        </div>
                        <p>{m.definition}</p>
                        <div>
                          <span className="muted">Formula:</span> <code>{m.formula}</code>
                        </div>
                        <div>
                          <span className="muted">Landmarks:</span> <code>{m.landmarks.join(', ')}</code>
                        </div>
                        <div>
                          <span className="muted">Value:</span> {formatValue(m, 'px')} / {formatValue(m, 'mm')}
                        </div>
                        {m.endpoints_px.length === 2 && (
                          <div>
                            <span className="muted">Endpoints (px):</span>{' '}
                            <code>
                              {m.endpoints_px.map(([x, y]) => `(${x.toFixed(1)}, ${y.toFixed(1)})`).join(' → ')}
                            </code>
                          </div>
                        )}
                        {m.error && <div className="warning-text">{m.error}</div>}
                        {m.quality_notes.length > 0 && (
                          <ul className="notes">
                            {m.quality_notes.map((n) => (
                              <li key={n}>{n}</li>
                            ))}
                          </ul>
                        )}
                        {Object.keys(m.details).length > 0 && (
                          <pre className="details">{JSON.stringify(m.details, null, 1)}</pre>
                        )}
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          </div>
        )
      })}
      <p className="muted small">
        Research prototype. Values are iris-scaled estimates, not medical or anthropometric
        measurements. Proxies are not anatomical breadths.
      </p>
    </section>
  )
}
