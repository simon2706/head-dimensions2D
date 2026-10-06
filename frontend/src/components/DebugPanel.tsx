import type { ReactNode } from 'react'
import type { IrisModelResult } from '../mediapipe/irisLandmark'
import type { MeasureResponse } from '../types/api'
import { fmt } from '../visualization/format'

function Row({ k, v }: { k: string; v: ReactNode }) {
  return (
    <tr>
      <th>{k}</th>
      <td>{v}</td>
    </tr>
  )
}

export function DebugPanel({ result, irisModel }: { result: MeasureResponse; irisModel: IrisModelResult | null }) {
  const { scale, pose, debug } = result
  const bb = debug.face_bbox_px
  return (
    <details className="panel debug">
      <summary>Debug information</summary>
      <table className="kv">
        <tbody>
          <Row k="Image resolution" v={`${debug.image_width} × ${debug.image_height}`} />
          <Row k="Face bounding box" v={`x ${fmt(bb.x, 0)}, y ${fmt(bb.y, 0)}, ${fmt(bb.width, 0)} × ${fmt(bb.height, 0)} px`} />
          <Row k="Face size (bbox diagonal)" v={`${fmt(debug.face_size_px.bbox_diagonal, 0)} px`} />
          <Row k="Scale method" v={scale.method} />
          {scale.fallback_reason && <Row k="Scale fallback" v={scale.fallback_reason} />}
          <Row k="Left iris diameter" v={`${fmt(scale.left_iris_diameter_px)} px (vertical ${fmt(scale.left_iris_vertical_diameter_px)})`} />
          <Row k="Right iris diameter" v={`${fmt(scale.right_iris_diameter_px)} px (vertical ${fmt(scale.right_iris_vertical_diameter_px)})`} />
          <Row k="Assumed iris diameter" v={`${fmt(scale.assumed_iris_diameter_mm)} mm`} />
          <Row k="Left scale" v={`${fmt(scale.left_mm_per_px, 4)} mm/px`} />
          <Row k="Right scale" v={`${fmt(scale.right_mm_per_px, 4)} mm/px`} />
          <Row k="Final scale" v={`${fmt(scale.final_mm_per_px, 4)} mm/px`} />
          <Row k="Scale mismatch" v={`${fmt(scale.difference_percent)} %`} />
          {Object.keys(scale.landmarks).length > 0 && (
            <Row k="Iris landmark pairs" v={<code>{JSON.stringify(scale.landmarks)}</code>} />
          )}
          {irisModel && (
            <Row
              k="Iris model eye crops (px)"
              v={(['right', 'left'] as const).map((side) => {
                const r = irisModel[side].roi
                return (
                  <div key={side}>
                    {side}: centre {fmt(r.cx, 0)}, {fmt(r.cy, 0)}; side {fmt(r.size, 0)}; rotation{' '}
                    {fmt((r.rotation * 180) / Math.PI, 1, '°')}
                  </div>
                )
              })}
            />
          )}
          <Row k="Head yaw (line of sight)" v={fmt(pose.yaw_deg, 2, '°')} />
          <Row k="Head pitch (line of sight)" v={fmt(pose.pitch_deg, 2, '°')} />
          <Row k="Head roll (line of sight)" v={`${fmt(pose.roll_deg, 2, '°')} (landmark ${fmt(pose.landmark_roll_deg, 2, '°')})`} />
          <Row
            k="Camera-axis yaw / pitch / roll"
            v={`${fmt(pose.camera_yaw_deg, 2, '°')} / ${fmt(pose.camera_pitch_deg, 2, '°')} / ${fmt(pose.camera_roll_deg, 2, '°')}`}
          />
          <Row k="Line-of-sight offset" v={fmt(pose.line_of_sight_offset_deg, 2, '°')} />
          <Row k="Face translation (MediaPipe cm)" v={pose.translation ? pose.translation.map((x) => x.toFixed(1)).join(', ') : '—'} />
          <Row k="Face frame" v={<code>{JSON.stringify(debug.face_frame)}</code>} />
          <Row k="Eye opening (lid / iris)" v={`R ${fmt(debug.eye_opening_ratios.right)}, L ${fmt(debug.eye_opening_ratios.left)}`} />
          <Row k="Sharpness (Laplacian var.)" v={fmt(debug.sharpness_laplacian_var, 1)} />
          <Row k="Engine version" v={result.engine_version} />
        </tbody>
      </table>

      <h3>Measurements</h3>
      <table className="grid">
        <thead>
          <tr>
            <th>id</th>
            <th>px</th>
            <th>mm</th>
            <th>landmarks</th>
            <th>formula</th>
          </tr>
        </thead>
        <tbody>
          {result.measurements.map((m) => (
            <tr key={m.id}>
              <td>
                <code>{m.id}</code>
              </td>
              <td>{fmt(m.value_px, 1)}</td>
              <td>{fmt(m.value_mm, 1)}</td>
              <td>
                <code>{m.landmarks.join(', ')}</code>
              </td>
              <td>
                <code>{m.formula}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Relevant landmarks (image px)</h3>
      <table className="grid">
        <thead>
          <tr>
            <th>id</th>
            <th>x</th>
            <th>y</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(debug.landmark_coordinates_px).map(([id, [x, y]]) => (
            <tr key={id}>
              <td>{id}</td>
              <td>{x.toFixed(1)}</td>
              <td>{y.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Feature vector</h3>
      <pre className="details">{JSON.stringify(result.features, null, 2)}</pre>
    </details>
  )
}
