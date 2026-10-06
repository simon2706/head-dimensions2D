// Copies the MediaPipe and LiteRT.js WASM runtimes from node_modules into public/
// so the browser loads them from this app instead of a CDN. Runs on `npm install`.
import { cpSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const runtimes = [
  ['MediaPipe', '@mediapipe/tasks-vision/wasm', 'public/mediapipe/wasm'],
  ['LiteRT.js', '@litertjs/core/wasm', 'public/litert/wasm'],
]
const models = ['public/models/face_landmarker.task', 'public/models/iris_landmark.tflite']

for (const [name, pkgPath, destPath] of runtimes) {
  const src = resolve(root, 'node_modules', pkgPath)
  const dest = resolve(root, destPath)
  if (!existsSync(src)) {
    console.error(`[setup-assets] ${pkgPath} is not installed; run npm install`)
    process.exit(1)
  }
  mkdirSync(dest, { recursive: true })
  cpSync(src, dest, { recursive: true })
  console.log(`[setup-assets] copied ${name} WASM runtime -> ${dest}`)
}
for (const model of models) {
  if (!existsSync(resolve(root, model))) {
    console.warn(`[setup-assets] ${model} missing: run ../scripts/fetch-model.sh`)
  }
}
