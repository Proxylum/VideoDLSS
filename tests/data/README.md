# Test clips

Stage 0 tests synthesise their own clips (`tests/integration/TestClips.cpp`) and need no files here.
`tiny.onnx` is the TensorRT smoke model of the stage 2 tests.

## The five acceptance clips (ТЗ §9, TASK-0010)

`python tests/data/fetch.py` downloads the sources (sha256-checked) into `cache/`, cuts the clips into `clips/`
and writes `clips/manifest.json` (source, licence, cut point, ffprobe of the result). Nothing here is committed
(decision of the 2026-09-18 briefing: no clips in git, no LFS); `--check` verifies what is on disk.
Cutting needs the ffmpeg / ffprobe CLI on PATH (`--ffmpeg DIR` or `FFMPEG_BIN` otherwise). The cache is ~940 MB
(Tears of Steel 1080p and Big Buck Bunny 1080p60 come as ZIPs that are extracted and deleted).

| Clip | Content (ТЗ) | Source, cut | Format |
|---|---|---|---|
| `static.mp4` | static camera: the round window of the lab, an airship passes outside | Tears of Steel 1080p, 103.0–113.0 s | 1920×800, 24 fps, 240 frames |
| `pan.mp4` | lateral tracking shot along the canal, parallax on trees and railings, a car passes | Tears of Steel 1080p, 274.5–284.5 s | 1920×800, 24 fps, 240 frames |
| `face.mp4` | three real faces in a dialogue, medium close-up, static camera, one take | Tears of Steel 1080p, 239.0–249.0 s | 1920×800, 24 fps, 240 frames |
| `foliage_hair.mp4` | grass and moss in the foreground, the bunny's fur as he climbs out of the burrow | Big Buck Bunny 1080p 60 fps (2013 render), 41.6–51.6 s | 1920×1080, 60 fps, 600 frames |
| `lowbitrate_720.mp4` | 720p at 1 Mbit/s with visible compression artefacts, as published | test-videos.co.uk encode of Big Buck Bunny, whole file | 1280×720, 30 fps, 300 frames |

Cuts are re-encoded with x264 (`crf 14`, `preset slow`, BT.709 tags, AAC audio kept) so every clip is a
constant-frame-rate MP4 the pipeline reads without a demuxer surprise; the low-bitrate clip is copied as is
because its artefacts are the point. `foliage_hair` keeps 60 fps: dropping every other frame gives a ground
truth for frame generation (`bench_clips.py --sections fg`).

Why these shots: contact sheets of the sources were inspected for takes without a cut. The open movies have no
continuous 10-second *fast* pan (trailers are cut every 1–3 s), so the canal tracking shot is the closest real
footage with a moving camera; real close-ups in Tears of Steel are shot/reverse-shot with cuts every 1–2 s, the
longest single close-up is 6.4 s, hence the three-face take for `face`.

### Licences and credits

| Source | Licence | Credit required by the licence |
|---|---|---|
| Tears of Steel (2012), `download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip` | CC BY 3.0 | (CC) Blender Foundation — mango.blender.org |
| Big Buck Bunny (2008; 2013 re-render 1080p 60 fps), `download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_60fps_normal.mp4.zip` | CC BY 3.0 | (c) copyright 2008, Blender Foundation — www.bigbuckbunny.org |
| Big Buck Bunny 720p 10 s 1 MB, `test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4` | CC BY 3.0 (the film); the re-encode is published by test-videos.co.uk for testing | (c) copyright 2008, Blender Foundation — www.bigbuckbunny.org; encode: test-videos.co.uk |

The Tears of Steel soundtrack is CC BY-ND 3.0 ((C) Joram Letwory — www.tearsofsteel.org): the clips keep the
audio only to exercise the pipeline's audio copy; they are test fixtures, not a redistribution.

## Benchmarks on the clips

`python tests/data/bench_clips.py` runs the stage 2 / 3 / 5 / 7 measurements on every clip
(OFA vs SEA-RAFT warp-PSNR, DA3 vs VDA scale and warped TAE, upscale A/B, frame-generation ground truth on the
60 fps clip) and prints the markdown tables that live in `docs/benchmarks.md`; results also go to
`build/release/clipbench/results.json`. It needs an NVIDIA GPU and the models in `models/cache/`.
