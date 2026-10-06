"""Experimental facial/head dimension measurement engine.

The engine consumes MediaPipe Face Landmarker output (478 normalized landmarks,
facial transformation matrix and blendshapes) and produces pixel- and
millimetre-space measurements plus capture-quality diagnostics. It never
receives the photograph itself.
"""

__version__ = "0.1.0"
