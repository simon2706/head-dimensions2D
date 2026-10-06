import { FaceLandmarker } from '@mediapipe/tasks-vision'
import { useMemo } from 'react'
import { toPixel } from '../mediapipe/detection'
import type { IrisModelResult } from '../mediapipe/irisLandmark'
import type { LandmarkPoint, MeasureResponse } from '../types/api'
import { formatValue, type Unit } from './format'
import { placeLabels } from './labels'
import { LAYER_COLORS, TYPE_STYLE } from './style'
import type { OverlayToggles } from './toggles'

interface Props {
  width: number
  height: number
  /** Rendered width in CSS px, used to keep strokes/text a constant on-screen size. */
  displayWidth: number
  landmarks: LandmarkPoint[]
  /** MediaPipe Iris landmark model output, drawn on the iris layer when present. */
  irisModel?: IrisModelResult | null
  result: MeasureResponse | null
  toggles: OverlayToggles
  unit: Unit
  selectedId: string | null
  onSelect?: (id: string | null) => void
  /** When given, the photo is drawn inside the SVG so it crops/zooms together with the overlay. */
  imageUrl?: string
  /** Visible region in image pixels [x, y, w, h]; defaults to the whole image. */
  viewBox?: [number, number, number, number]
}

type Conn = { start: number; end: number }

function connectionsPath(conns: Conn[], pts: [number, number][]): string {
  let d = ''
  for (const { start, end } of conns) {
    const a = pts[start]
    const b = pts[end]
    if (a && b) d += `M${a[0].toFixed(1)} ${a[1].toFixed(1)}L${b[0].toFixed(1)} ${b[1].toFixed(1)}`
  }
  return d
}

export function Overlay({
  width,
  height,
  displayWidth,
  landmarks,
  irisModel = null,
  result,
  toggles,
  unit,
  selectedId,
  onSelect,
  imageUrl,
  viewBox,
}: Props) {
  const [vx, vy, vw, vh] = viewBox ?? [0, 0, width, height]
  const k = vw / Math.max(displayWidth, 1) // image px per screen px
  const pts = useMemo(() => landmarks.map((p) => toPixel(p, width, height)), [landmarks, width, height])

  const paths = useMemo(
    () => ({
      mesh: connectionsPath(FaceLandmarker.FACE_LANDMARKS_TESSELATION, pts),
      eyes: connectionsPath(
        [...FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, ...FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE],
        pts,
      ),
      iris: connectionsPath(
        [...FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, ...FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS],
        pts,
      ),
      eyebrows: connectionsPath(
        [...FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, ...FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW],
        pts,
      ),
      contour: connectionsPath(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, pts),
    }),
    [pts],
  )

  const measurements = result?.measurements.filter((m) => m.endpoints_px.length === 2) ?? []
  const selected = measurements.find((m) => m.id === selectedId) ?? null
  const fontSize = 12 * k
  const xs = measurements.flatMap((m) => m.endpoints_px.map((p) => p[0]))
  const faceRight = Math.max(...xs, vx)
  const faceLeft = Math.min(...xs, vx + vw)
  // Labels go in a column beside the face: right if there is room, else left.
  const approxLabelWidth = 22 * fontSize
  const roomRight = vx + vw - faceRight
  const labelsOnRight = roomRight >= 24 * k + approxLabelWidth || roomRight >= faceLeft - vx
  const labels = placeLabels(
    measurements.map((m) => ({
      id: m.id,
      anchor: m.endpoints_px.reduce((a, b) =>
        labelsOnRight ? (b[0] > a[0] ? b : a) : b[0] < a[0] ? b : a,
      ) as [number, number],
    })),
    labelsOnRight ? faceRight + 24 * k : faceLeft - 24 * k,
    fontSize * 1.35,
  )
  const leaderGap = (labelsOnRight ? 1 : -1) * 4 * k
  const byId = new Map(measurements.map((m) => [m.id, m]))

  // Iris horizontal diameters actually used for the scale, as reported by the backend.
  const irisDiameters = result
    ? [result.scale.endpoints_px.right_horizontal, result.scale.endpoints_px.left_horizontal].filter(Boolean)
    : []
  // Iris-model points per eye: centre + 4 contour points, and a circle through the contour.
  const irisModelEyes = irisModel
    ? [irisModel.right, irisModel.left].map((eye) => {
        const p = eye.iris.map((q) => toPixel(q, width, height))
        const r = p.slice(1).reduce((acc, q) => acc + Math.hypot(q[0] - p[0][0], q[1] - p[0][1]), 0) / 4
        return { p, r }
      })
    : []

  const idSet = new Set<number>()
  if (toggles.ids && result) {
    for (const id of Object.keys(result.debug.landmark_coordinates_px)) idSet.add(Number(id))
  }
  if (selected) selected.landmarks.forEach((id) => idSet.add(id))

  const stroke = (w: number) => ({ strokeWidth: w, vectorEffect: 'non-scaling-stroke' as const })

  return (
    <svg
      className={imageUrl ? 'photo-svg' : 'overlay'}
      viewBox={`${vx} ${vy} ${vw} ${vh}`}
      preserveAspectRatio="xMidYMid meet"
      style={imageUrl ? { aspectRatio: `${vw} / ${vh}`, width: `min(100%, calc((100vh - 150px) * ${vw / vh}))` } : undefined}
      onClick={() => onSelect?.(null)}
    >
      {imageUrl && <image href={imageUrl} x={0} y={0} width={width} height={height} />}
      {toggles.mesh && <path d={paths.mesh} stroke={LAYER_COLORS.mesh} fill="none" {...stroke(0.5)} />}
      {toggles.contour && <path d={paths.contour} stroke={LAYER_COLORS.contour} fill="none" {...stroke(1.2)} />}
      {toggles.eyebrows && <path d={paths.eyebrows} stroke={LAYER_COLORS.eyebrows} fill="none" {...stroke(1.2)} />}
      {toggles.eyes && <path d={paths.eyes} stroke={LAYER_COLORS.eyes} fill="none" {...stroke(1)} />}
      {toggles.iris && (
        <g>
          {/* Face Landmarker iris (468-477); faint when the Iris model is shown alongside. */}
          <g opacity={irisModel ? 0.45 : 1}>
            <path d={paths.iris} stroke={LAYER_COLORS.iris} fill="none" {...stroke(1)} />
            {[468, 473].map((i) =>
              pts[i] ? <circle key={i} cx={pts[i][0]} cy={pts[i][1]} r={1.5 * k} fill={LAYER_COLORS.iris} /> : null,
            )}
          </g>
          {irisModelEyes.map(({ p, r }, e) => (
            <g key={e}>
              <circle cx={p[0][0]} cy={p[0][1]} r={r} stroke={LAYER_COLORS.irisModel} fill="none" {...stroke(1)} />
              {p.map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r={1.5 * k} fill={LAYER_COLORS.irisModel} />
              ))}
            </g>
          ))}
          {irisDiameters.map(([a, b], i) => (
            <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#ffffff" {...stroke(1)} />
          ))}
        </g>
      )}

      {toggles.measurements &&
        measurements.map((m) => {
          const [[x1, y1], [x2, y2]] = m.endpoints_px
          const style = TYPE_STYLE[m.type]
          const isSel = m.id === selectedId
          const dim = selectedId !== null && !isSel
          return (
            <g
              key={m.id}
              className="measurement"
              opacity={dim ? 0.3 : 1}
              onClick={(e) => {
                e.stopPropagation()
                onSelect?.(isSel ? null : m.id)
              }}
            >
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" {...stroke(12)} />
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={style.color}
                strokeDasharray={style.dash}
                {...stroke(isSel ? 3 : 1.6)}
              />
              {[
                [x1, y1],
                [x2, y2],
              ].map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r={(isSel ? 4 : 3) * k} fill={style.color} stroke="#000" {...stroke(0.8)} />
              ))}
            </g>
          )
        })}

      {toggles.measurements &&
        labels.map((l) => {
          const m = byId.get(l.id)!
          const style = TYPE_STYLE[m.type]
          const isSel = m.id === selectedId
          const dim = selectedId !== null && !isSel
          return (
            <g
              key={`label-${l.id}`}
              opacity={dim ? 0.3 : 1}
              className="measurement"
              onClick={(e) => {
                e.stopPropagation()
                onSelect?.(isSel ? null : m.id)
              }}
            >
              <line
                x1={l.anchor[0] + leaderGap}
                y1={l.anchor[1]}
                x2={l.x - leaderGap}
                y2={l.y - fontSize * 0.35}
                stroke={style.color}
                strokeOpacity={0.6}
                {...stroke(0.7)}
              />
              <text
                x={l.x}
                y={l.y}
                fontSize={fontSize}
                fill={style.color}
                textAnchor={labelsOnRight ? 'start' : 'end'}
                fontWeight={isSel ? 700 : 500}
                className="label"
              >
                {formatValue(m, unit)} · {m.name}
              </text>
            </g>
          )
        })}

      {idSet.size > 0 &&
        [...idSet].map((id) =>
          pts[id] ? (
            <g key={`id-${id}`} pointerEvents="none">
              <circle cx={pts[id][0]} cy={pts[id][1]} r={1.8 * k} fill={LAYER_COLORS.ids} />
              <text
                x={pts[id][0] + 3 * k}
                y={pts[id][1] - 3 * k}
                fontSize={8.5 * k}
                fill={LAYER_COLORS.ids}
                className="label"
              >
                {id}
              </text>
            </g>
          ) : null,
        )}
    </svg>
  )
}
