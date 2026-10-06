import type { MeasureRequest, MeasureResponse, MeasurementConfig } from '../types/api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = JSON.stringify((await res.json()).detail ?? detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`Backend ${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export async function fetchDefaultConfig(): Promise<MeasurementConfig> {
  return json(await fetch('/api/config'))
}

export async function measure(req: MeasureRequest, signal?: AbortSignal): Promise<MeasureResponse> {
  return json(
    await fetch('/api/measure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal,
    }),
  )
}
