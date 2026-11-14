# X2 realistic test media (worktree screw-it-wt/media, branch ws/media)

Read: briefs/common.md, AGENTS.md, SPEC.md 2.2, 3.3 (upload validation), 6.1 steps 1 and 3.

You own only scripts/make_test_media.py, fixtures/media/, backend/tests/media/. Create nothing elsewhere.

Purpose: P2 (frames, scale) and P5 (media validator) need test files that look like what real phones send. Build a deterministic generator and commit its outputs, so every team can test against the same files.

Build scripts/make_test_media.py producing fixtures/media/ with a manifest.json describing each file (name, container, codec, pixel format, colour transfer, duration, resolution, rotation, expected validator verdict, expected keyframe count range, and for board videos the ground-truth board pose per frame and the known object size):
1. A synthetic orbit scene rendered on the CPU: a ChArUco board (5x7, 30 mm squares, OpenCV DICT_5X5_100 or what P2 uses: check backend/screwit/recon/board.py in /local/home/brancli/.kiro/crew/workspace/screw-it-wt/recon if it exists, else choose and record) lying flat, plus a 100 mm matte test cube and a 38.1 mm OD cylinder, camera orbiting at 30-40 degrees elevation, mild motion blur on some frames. Render with a numpy rasterizer or whatever works headless here. Save ground-truth cameras (world_to_camera row-major, OpenCV, mm) and K.
2. Encode that scene as: (a) iPhone-like MOV, HEVC Main10, yuv420p10le, BT.2020 + HLG transfer tags, 1920x1080, 30 fps, 20 s, with a rotation side-data/display-matrix of 90 degrees (portrait); (b) Android-like MP4 H.264 yuv420p 1920x1080 30 fps 25 s; (c) WebM VP9 1280x720 15 s; (d) a 6 s clip (too short); (e) a truncated copy of (b); (f) an MP4 with an MPEG-4 Part 2 or other non-allowlisted codec; (g) a text file renamed .mp4. Also a photos set: 24 JPEGs of the scene, some with EXIF orientation 6 and 8, mixed portrait/landscape.
3. Use the imageio-ffmpeg binary. Check which encoders it has (`-encoders`): if libx265 or 10-bit HEVC is missing, try a static ffmpeg build downloaded into $KIROCREW_SCRATCH (verify its checksum from the official source page), and if that is impossible, produce the closest available file, mark it in the manifest as a substitute, and say so plainly in your report. Never claim a property the file does not have: verify every file afterwards by probing it and record the probe output in the manifest.
4. Keep fixtures/media/ under 60 MB total (use reasonable bitrates).
5. backend/tests/media/test_media.py: re-probe every file and assert it matches the manifest; assert the generator is deterministic (rerun into a temp dir, compare hashes of the non-encoded artifacts and probe properties of the encoded ones).

Report also: exactly which real-phone properties are reproduced and which are not.
