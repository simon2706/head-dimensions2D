export function CaptureInstructions() {
  return (
    <section className="instructions">
      <h2>For best measurement accuracy</h2>
      <ul>
        <li>Use a neutral expression</li>
        <li>Keep both eyes fully open</li>
        <li>Look directly toward the camera</li>
        <li>Keep your head straight</li>
        <li>Place the camera approximately at eye height</li>
        <li>Avoid a very close selfie — ideally place the camera around 1–1.5 m away</li>
        <li>Use good, even lighting</li>
        <li>Keep hair away from the side of the face where possible</li>
      </ul>
      <p className="muted small">
        Images are processed locally in your browser. Only landmark coordinates are sent to the
        local measurement backend; the photo itself is never uploaded.
      </p>
    </section>
  )
}
