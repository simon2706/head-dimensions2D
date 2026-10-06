import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDefaultConfig, measure } from './api/client'
import { CaptureInstructions } from './components/CaptureInstructions'
import { ConfigPanel } from './components/ConfigPanel'
import { DebugPanel } from './components/DebugPanel'
import { MeasurementsPanel } from './components/MeasurementsPanel'
import { QualityPanel } from './components/QualityPanel'
import { WebcamCapture } from './components/WebcamCapture'
import { buildExport, downloadJson, exportFileName, type CaptureSource } from './export/exportJson'
import { buildMeasureRequest, extractDetection, landmarkBBox, type FaceDetection } from './mediapipe/detection'
import { getFaceLandmarker, MODEL_DESCRIPTION } from './mediapipe/faceLandmarker'
import { detectIrisSafe, getIrisModel } from './mediapipe/irisLandmark'
import { faceSharpness } from './mediapipe/sharpness'
import type { ImageMetrics, MeasureResponse, MeasurementConfig } from './types/api'
import type { Unit } from './visualization/format'
import { Overlay } from './visualization/Overlay'
import { DEFAULT_TOGGLES, type OverlayToggles } from './visualization/toggles'

interface Capture {
  url: string
  width: number
  height: number
  source: CaptureSource
}

const TOGGLE_LABELS: [keyof OverlayToggles, string][] = [
  ['mesh', 'Full mesh'],
  ['eyes', 'Eyes'],
  ['iris', 'Iris'],
  ['eyebrows', 'Eyebrows'],
  ['contour', 'Face contour'],
  ['measurements', 'Measurements'],
  ['ids', 'Landmark IDs'],
]

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not decode image'))
    img.src = url
  })
}

/** Draw any image source into a canvas at its native resolution (EXIF orientation applied). */
function toCanvas(src: HTMLImageElement): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = src.naturalWidth
  c.height = src.naturalHeight
  c.getContext('2d')!.drawImage(src, 0, 0)
  return c
}

export default function App() {
  const [defaults, setDefaults] = useState<MeasurementConfig | null>(null)
  const [config, setConfig] = useState<MeasurementConfig>({})
  const [capture, setCapture] = useState<Capture | null>(null)
  const [detection, setDetection] = useState<FaceDetection | null>(null)
  const [imageMetrics, setImageMetrics] = useState<ImageMetrics | null>(null)
  const [result, setResult] = useState<MeasureResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [camera, setCamera] = useState(false)
  const [toggles, setToggles] = useState<OverlayToggles>(DEFAULT_TOGGLES)
  const [unit, setUnit] = useState<Unit>('mm')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [zoomToFace, setZoomToFace] = useState(true)
  const [displayWidth, setDisplayWidth] = useState(800)
  const stageRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const configRef = useRef(config)
  useEffect(() => {
    configRef.current = config
  }, [config])

  useEffect(() => {
    fetchDefaultConfig()
      .then((c) => {
        setDefaults(c)
        setConfig(c)
      })
      .catch((e) => setError(`Cannot reach the measurement backend (${e.message}). Is it running on :8000?`))
    // Warm up the models so the first photo is processed quickly.
    getFaceLandmarker('IMAGE').catch((e) => setError(`Failed to load MediaPipe: ${e.message}`))
    // A failure here is not fatal: the backend falls back to the Face Landmarker iris and warns.
    getIrisModel().catch((e) => console.warn('[iris] failed to load the Iris landmark model', e))
  }, [])

  useEffect(() => {
    const el = stageRef.current?.querySelector('svg')
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setDisplayWidth(entry.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [capture, camera])

  const measureCtrl = useRef<AbortController | null>(null)
  const runMeasure = useCallback(
    async (det: FaceDetection, cfg: MeasurementConfig, metrics: ImageMetrics | null) => {
      measureCtrl.current?.abort()
      const ctrl = new AbortController()
      measureCtrl.current = ctrl
      setBusy('Measuring…')
      try {
        setResult(await measure(buildMeasureRequest(det, cfg, metrics), ctrl.signal))
        setError(null)
      } catch (e) {
        if (!(e instanceof DOMException && e.name === 'AbortError')) {
          setError(e instanceof Error ? e.message : String(e))
        }
      } finally {
        if (measureCtrl.current === ctrl) setBusy(null)
      }
    },
    [],
  )

  /** Config edits re-run only the measurement engine on the stored landmarks. */
  const onConfigChange = (c: MeasurementConfig) => {
    setConfig(c)
    if (detection) runMeasure(detection, c, imageMetrics)
  }

  const processCanvas = useCallback(async (canvas: HTMLCanvasElement, url: string, source: CaptureSource) => {
    setBusy('Detecting landmarks…')
    setError(null)
    setResult(null)
    setDetection(null)
    setSelectedId(null)
    setCapture((prev) => {
      if (prev && prev.url !== url) URL.revokeObjectURL(prev.url)
      return { url, width: canvas.width, height: canvas.height, source }
    })
    try {
      const lm = await getFaceLandmarker('IMAGE')
      const face = extractDetection(lm.detect(canvas), canvas.width, canvas.height)
      if (!face) {
        setError('No face detected. Use a well-lit frontal photo with the whole face visible.')
        setBusy(null)
        return
      }
      const det = { ...face, irisModel: await detectIrisSafe(canvas, face.landmarks, canvas.width, canvas.height) }
      const metrics = faceSharpness(canvas, landmarkBBox(det.landmarks, canvas.width, canvas.height))
      setImageMetrics(metrics)
      setDetection(det)
      await runMeasure(det, configRef.current, metrics)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(null)
    }
  }, [runMeasure])

  const onFile = async (file: File) => {
    const url = URL.createObjectURL(file)
    try {
      const img = await loadImage(url)
      await processCanvas(toCanvas(img), url, { kind: 'upload', fileName: file.name })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const onWebcamCapture = (canvas: HTMLCanvasElement) => {
    setCamera(false)
    canvas.toBlob(
      (blob) => {
        if (blob) processCanvas(canvas, URL.createObjectURL(blob), { kind: 'webcam' })
      },
      'image/png',
    )
  }

  const onExport = () => {
    if (!detection || !result || !capture) return
    downloadJson(buildExport(detection, imageMetrics, result, capture.source, MODEL_DESCRIPTION), exportFileName())
  }

  return (
    <div className="app">
      <header>
        <h1>Facial Measurement Prototype</h1>
        <span className="disclaimer">
          Experimental research tool — not a medical or anthropometric measurement device
        </span>
      </header>

      <main>
        <div className="viewer">
          {camera ? (
            <WebcamCapture config={config} onCapture={onWebcamCapture} onClose={() => setCamera(false)} />
          ) : capture ? (
            <div className="stage" ref={stageRef}>
              <Overlay
                width={capture.width}
                height={capture.height}
                displayWidth={displayWidth}
                landmarks={detection?.landmarks ?? []}
                irisModel={detection?.irisModel ?? null}
                result={result}
                toggles={toggles}
                unit={unit}
                selectedId={selectedId}
                onSelect={setSelectedId}
                imageUrl={capture.url}
                viewBox={zoomToFace && detection ? faceViewBox(detection) : undefined}
              />
            </div>
          ) : (
            <CaptureInstructions />
          )}
        </div>

        <aside>
          {busy && <p className="busy">{busy}</p>}
          {error && <p className="error">{error}</p>}
          {result && (
            <>
              <QualityPanel quality={result.quality} pose={result.pose} />
              <MeasurementsPanel
                measurements={result.measurements}
                unit={unit}
                onUnitChange={setUnit}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
              <DebugPanel result={result} irisModel={detection?.irisModel ?? null} />
            </>
          )}
          {defaults && <ConfigPanel defaults={defaults} config={config} onChange={onConfigChange} />}
          {!result && capture && !busy && !error && <p className="muted">No result.</p>}
          {capture && !camera && <CaptureInstructionsHint />}
        </aside>
      </main>

      <footer className="toolbar">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onFile(f)
            e.target.value = ''
          }}
        />
        <button className="primary" onClick={() => fileRef.current?.click()}>
          Upload image
        </button>
        <button onClick={() => setCamera((c) => !c)}>{camera ? 'Close camera' : 'Take photo'}</button>
        <div className="toggles">
          {TOGGLE_LABELS.map(([key, label]) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={toggles[key]}
                onChange={(e) => setToggles({ ...toggles, [key]: e.target.checked })}
              />
              {label}
            </label>
          ))}
        </div>
        <label className="zoom-toggle">
          <input type="checkbox" checked={zoomToFace} onChange={(e) => setZoomToFace(e.target.checked)} />
          Zoom to face
        </label>
        <button onClick={onExport} disabled={!result}>
          Export JSON
        </button>
      </footer>
    </div>
  )
}

/** Face bounding box padded for context, widened on the right for the label column. */
function faceViewBox(det: FaceDetection): [number, number, number, number] {
  const bb = landmarkBBox(det.landmarks, det.imageWidth, det.imageHeight)
  const padX = 0.2 * bb.width
  const padY = 0.15 * bb.height
  const left = bb.x - padX
  const contentW = bb.width + 2 * padX
  const w = contentW / 0.68 // ~32 % of the width reserved for labels
  return [left, bb.y - padY, w, bb.height + 2 * padY]
}

function CaptureInstructionsHint() {
  return (
    <details className="panel">
      <summary>Capture instructions</summary>
      <CaptureInstructions />
    </details>
  )
}
