"""OBS recording subsystem — OPTIONAL at runtime.

Editing works without it; this subsystem adds screen+camera capture (with the
camera recorded to a SEPARATE clean file via the Source Record plugin) plus the
camera-over-screen compositor and the GPU cut-synced camera mix.

Every heavy/optional dependency is imported LAZILY inside the function that needs
it, each with a clear ``pip install dopest-clip[<extra>]`` hint:

    dopest-clip[obs]       websocket-client  (WS v5 client / recording control)
    dopest-clip[graphics]  resvg-py          (SVG overlay rasterization)
    dopest-clip[matting]   moviepy/torch/cv2 (compositor + GPU camera mix)

So ``import dopest_clip.obs.*`` costs nothing and the light test venv can import
every module here. The pure pieces (timeline math, SVG string builders, keyframe
params) carry no heavy deps at all and are unit-testable directly.
"""
