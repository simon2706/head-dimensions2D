import { useEffect, useRef, useState } from 'react'
import { measure } from '../api/client'
import { buildMeasureRequest, extractDetection, type FaceDetection } from '../mediapipe/detection'
import { getFaceLandmarker } from '../mediapipe/faceLandmarker'
import { detectIrisSafe } from '../mediapipe/irisLandmark'
import type { MeasureResponse, MeasurementConfig } from '../types/api'
import { Overlay } from '../visualization/Overlay'
import { DEFAULT_TOGGLES } from '../visualization/toggles'
import { STATUS_ICON } from '../visualization/style'

interface Props {
  config: MeasurementConfig
  /** Receives an unmirrored still frame at full camera resolution. */
  onCapture: (frame: HTMLCanvasElement) => void
  onClose: () => void
}

const LIVE_CHECK_INTERVAL_MS = 330
const LIVE_CHECKS = ['face_count', 'head_roll', 'head_yaw', 'head_pitch', 'iris_agreement', 'eyes_open', 'face_not_clipped']
const ALIGNMENT_CHECKS = ['face_count', 'head_roll', 'head_yaw', 'head_pitch']

/**
 * Live preview with lightweight alignment feedback. The <video> and its overlay
 * are mirrored together with CSS for natural UX; landmarks and the captured
 * frame stay in the camera's (unmirrored) coordinate system.
 */
export function WebcamCapture({ config, onCapture, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [detection, setDetection] = useState<FaceDetection | null>(null)
  const [live, setLive] = useState<MeasureResponse | null>(null)
  const [displayWidth, setDisplayWidth] = useState(640)
  const configRef = useRef(config)
  useEffect(() => {
    configRef.current = config
  }, [config])

  useEffect(() => {
    let stream: MediaStream | null = null
    let raf = 0
    let stopped = false
    let lastCheck = 0
    let inflight = false
    let lastTs = -1

    const loop = async () => {
      const video = videoRef.current
      if (stopped || !video) return
      if (video.readyState >= 2 && video.videoWidth > 0) try {
        setDisplayWidth(video.clientWidth)
        const lm = await getFaceLandmarker('VIDEO')
        const ts = Math.max(performance.now(), lastTs + 1)
        lastTs = ts
        const det = extractDetection(lm.detectForVideo(video, ts), video.videoWidth, video.videoHeight)
        setDetection(det)
        if (!det) setLive(null)
        const now = performance.now()
        if (det && !inflight && now - lastCheck > LIVE_CHECK_INTERVAL_MS) {
          inflight = true
          lastCheck = now
          // The Iris model runs only on these throttled checks, not on every frame.
          detectIrisSafe(video, det.landmarks, det.imageWidth, det.imageHeight)
            .then((irisModel) => measure(buildMeasureRequest({ ...det, irisModel }, configRef.current, null)))
            .then((r) => !stopped && setLive(r))
            .catch(() => !stopped && setLive(null))
            .finally(() => (inflight = false))
        }
      } catch (e) {
        console.error('[webcam] detection failed', e)
      }
      raf = requestAnimationFrame(loop)
    }

    navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' }, audio: false })
      .then(async (s) => {
        stream = s
        if (stopped || !videoRef.current) return
        videoRef.current.srcObject = s
        await videoRef.current.play()
        raf = requestAnimationFrame(loop)
      })
      .catch((e) => setError(`Camera unavailable: ${e instanceof Error ? e.message : String(e)}`))

    return () => {
      stopped = true
      cancelAnimationFrame(raf)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const capture = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')!.drawImage(video, 0, 0) // unmirrored
    onCapture(canvas)
  }

  const checks = live?.quality.checks.filter((c) => LIVE_CHECKS.includes(c.id)) ?? []
  const aligned =
    detection !== null &&
    checks.length > 0 &&
    checks.filter((c) => ALIGNMENT_CHECKS.includes(c.id)).every((c) => c.status !== 'fail')

  return (
    <div className="webcam">
      {error && <p className="error">{error}</p>}
      <div className="stage mirrored">
        <video ref={videoRef} playsInline muted />
        {detection && (
          <Overlay
            width={detection.imageWidth}
            height={detection.imageHeight}
            displayWidth={displayWidth}
            landmarks={detection.landmarks}
            result={null}
            toggles={{ ...DEFAULT_TOGGLES, ids: false, measurements: false, eyebrows: false }}
            unit="mm"
            selectedId={null}
          />
        )}
      </div>
      <div className="webcam-bar">
        <ul className="live-checks">
          {!detection && <li className="status-fail">✗ No face</li>}
          {checks.map((c) => (
            <li key={c.id} className={`status-${c.status}`}>
              {STATUS_ICON[c.status]} {c.label}
              {c.unit === '°' && c.value !== null ? ` ${c.value.toFixed(1)}°` : ''}
            </li>
          ))}
        </ul>
        <div className="webcam-actions">
          <button className="primary" onClick={capture} disabled={!detection}>
            {aligned ? 'Capture' : 'Capture anyway'}
          </button>
          <button onClick={onClose}>Close camera</button>
        </div>
      </div>
      <p className="muted small">
        Preview is mirrored like a mirror; the captured photo and all landmark coordinates are not.
      </p>
    </div>
  )
}
