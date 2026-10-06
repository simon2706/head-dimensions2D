export interface OverlayToggles {
  mesh: boolean
  eyes: boolean
  iris: boolean
  eyebrows: boolean
  contour: boolean
  measurements: boolean
  ids: boolean
}

export const DEFAULT_TOGGLES: OverlayToggles = {
  mesh: false,
  eyes: true,
  iris: true,
  eyebrows: true,
  contour: true,
  measurements: true,
  ids: true,
}
