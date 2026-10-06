import type { MeasurementConfig } from '../types/api'

interface Props {
  defaults: MeasurementConfig
  config: MeasurementConfig
  onChange: (c: MeasurementConfig) => void
}

const PRIMARY = ['assumed_iris_diameter_mm', 'scale_source']
/** Config keys that take one of a fixed set of string values (backend Literal types). */
const CHOICES: Record<string, string[]> = {
  scale_source: ['iris_model', 'face_landmarker'],
}

/** Editable view of the backend MeasurementConfig. Changes re-run the measurement only (no re-detection). */
export function ConfigPanel({ defaults, config, onChange }: Props) {
  const keys = Object.keys(defaults)
  const ordered = [...PRIMARY.filter((k) => keys.includes(k)), ...keys.filter((k) => !PRIMARY.includes(k))]
  const set = (key: string, raw: string) => {
    const v = Number(raw)
    if (raw.trim() !== '' && Number.isFinite(v)) onChange({ ...config, [key]: v })
  }
  return (
    <details className="panel config">
      <summary>Measurement configuration</summary>
      <div className="config-grid">
        {ordered.map((key) => (
          <label key={key} className={config[key] !== defaults[key] ? 'changed' : ''}>
            <span>{key}</span>
            {CHOICES[key] ? (
              <select value={String(config[key])} onChange={(e) => onChange({ ...config, [key]: e.target.value })}>
                {CHOICES[key].map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                step="any"
                defaultValue={config[key]}
                key={`${key}-${config[key]}`}
                onBlur={(e) => set(key, e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && set(key, (e.target as HTMLInputElement).value)}
              />
            )}
          </label>
        ))}
      </div>
      <button onClick={() => onChange({ ...defaults })}>Reset to defaults</button>
    </details>
  )
}
