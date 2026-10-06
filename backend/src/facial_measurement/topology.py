"""MediaPipe Face Landmarker landmark topology used by the measurement engine.

All landmark IDs are taken from the official MediaPipe Face Landmarker topology
(478 points: 468 face-mesh points + 10 iris points) and its canonical face model
(``mediapipe/modules/face_geometry/data/canonical_face_model.obj``).

Naming convention: "left" and "right" always refer to the **subject's** left and
right, as in MediaPipe's own connection sets (``FACEMESH_LEFT_EYE`` etc.). In a
non-mirrored frontal photo the subject's right side appears on the image's left.

Canonical-model coordinates quoted below are (x, y) in cm with y pointing up;
the outer canthi sit at y = 2.66.
"""

NUM_LANDMARKS = 478

# --- Eye corners (canthi) -----------------------------------------------------
RIGHT_EYE_OUTER = 33  # subject's right lateral canthus
RIGHT_EYE_INNER = 133  # subject's right medial canthus
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362

# --- Eyelids (mid upper / mid lower), used for eye-opening checks ---------------
RIGHT_EYE_UPPER_LID = 159
RIGHT_EYE_LOWER_LID = 145
LEFT_EYE_UPPER_LID = 386
LEFT_EYE_LOWER_LID = 374

# --- Iris (refined landmarks 468-477) --------------------------------------------
# Each iris has a centre followed by four contour points. The horizontal
# boundary pair is used for the iris diameter (the vertical pair is reported for
# debugging only because the eyelids frequently occlude the iris vertically).
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_CONTOUR = (469, 470, 471, 472)
RIGHT_IRIS_HORIZONTAL = (469, 471)
RIGHT_IRIS_VERTICAL = (470, 472)

LEFT_IRIS_CENTER = 473
LEFT_IRIS_CONTOUR = (474, 475, 476, 477)
LEFT_IRIS_HORIZONTAL = (474, 476)
LEFT_IRIS_VERTICAL = (475, 477)

# --- MediaPipe Iris landmark model (iris_landmark.tflite) -------------------------
# A separate model from Face Landmarker. It runs on a 64x64 crop around each eye and
# returns 71 eye-contour points and 5 iris points (centre + 4 contour points). The
# eye crop is built from two eye corners, exactly as in MediaPipe's
# graphs/iris_tracking/iris_tracking_cpu.pbtxt ("left eye" there = image left =
# subject's right eye, which the model sees unflipped; the other eye is flipped).
IRIS_MODEL_NUM_IRIS = 5
IRIS_MODEL_NUM_EYE_CONTOUR = 71
IRIS_MODEL_CENTER = 0
# The 4 contour points go around the iris (verified on real output: 1 and 3 are
# the horizontal extremes, 2 and 4 the vertical ones; same order as 469-472), so
# opposite points are (1, 3) and (2, 4). Note: MediaPipe's own
# iris_to_depth_calculator.cc pairs (1, 2) and (3, 4), which are *adjacent*
# points (~sqrt(2) x radius, not a diameter); that pairing is not used here.
# Which pair is horizontal is still decided per image in the face frame.
IRIS_MODEL_OPPOSITE_PAIRS = ((1, 3), (2, 4))
RIGHT_EYE_ROI_CORNERS = (RIGHT_EYE_OUTER, RIGHT_EYE_INNER)  # 33 -> 133
LEFT_EYE_ROI_CORNERS = (LEFT_EYE_INNER, LEFT_EYE_OUTER)  # 362 -> 263

# --- Eyebrows (MediaPipe FACEMESH_RIGHT_EYEBROW / FACEMESH_LEFT_EYEBROW) --------
# The lateral-most point of each contour is selected per image; in practice it is
# 70/300 (upper tail, canonical |x| = 5.72) or 46/276 (lower tail, |x| = 5.25).
RIGHT_EYEBROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
LEFT_EYEBROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)

# --- Face oval (MediaPipe FACEMESH_FACE_OVAL, ordered as a closed loop) ----------
# Starts at the top of the forehead (10), runs down the subject's LEFT side
# (image right) to the chin (152), then up the subject's RIGHT side back to 10.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)  # fmt: skip
FOREHEAD_TOP = 10
CHIN = 152

# --- Upper temporal mesh width ---------------------------------------------------
# 162 / 389 are the face-oval vertices at canonical y = 4.11, i.e. at the level of
# the eyebrow tail (46/70 are at y = 3.88-4.25) and on the lateral silhouette
# (|x| = 7.56). This is approximately where glasses temples pass the face.
# Rejected neighbours: 127/356 (y = 2.36, eye level - duplicates the eye-level
# width) and 21/251 (y = 5.43, upper forehead).
RIGHT_UPPER_TEMPORAL = 162
LEFT_UPPER_TEMPORAL = 389

# --- Midface level anchor --------------------------------------------------------
# Landmark 2 is the lowest midline point of the nose (approximately subnasale;
# canonical y = -2.09). It is less sensitive to pitch than the nose tip (1).
SUBNASALE = 2
NOSE_TIP = 1
