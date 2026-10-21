"""360 video (Insta360 X3) support: intake, proxy analysis, director and renderer.

Modules:
    x3        clip discovery and pairing (``_00_`` / ``_10_`` / ``.lrv``), modes,
              trailer metadata (lens calibration, IMU) via telemetry-parser
    geometry  ffmpeg ``v360`` conventions, IMU -> camera axes, horizon-levelled
              orientation track, dual-fisheye sampling on the GPU
    views     ``.lrv`` proxy analysis: 8 horizon views + the downward selfie view
              at ~1 fps, CLIP / person detector / aesthetic, per-direction scores
    director  smooth virtual-camera path (yaw / pitch / FOV) per segment
    render    flat MP4 rendering with ``v360`` + ``sendcmd`` from an SDK-stitched
              equirect, an Insta360 Studio export, or the raw dual fisheye
    stitch    Insta360 Media SDK (``MediaSDKTest.exe``) wrapper
    project   ``<project>/pano.json`` and source-mode resolution
"""
