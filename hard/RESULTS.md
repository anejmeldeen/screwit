# Hard-object results (partial: stopped by the user)

The evaluation was stopped after one scene. Rerun the rest with
`scripts/hard_eval.py --all` (see `hard/README.md`).

| Scene | Board | Scale (key dimensions) | Object mask IoU | Mesh accuracy | Verdict |
|---|---|---|---|---|---|
| `gso_real_glass` (Google Scanned Objects glass on a tabletop) | aruco | see `hard/gso_real_glass.json` | 0.173 | 133.6 mm (median distance to the true surface) | **fails** |
| `gso_real_mug`, `gso_real_shoe`, `gso_real_toy` | not run | | | | |
| `under_sink`, `under_sink_portrait` | not run | | | | |
| `knob_panel` | not run | | | | |

## What the one result says

The board was found (fit 1.09 mm, tilt 1.0 deg), so scale and frame were right, but the object mask
was wrong: the scanned object is 423 x 286 x 229 mm against a true 205 x 38 x 205 mm glass, so the
mask took in a large area of tabletop around it (mask IoU 0.17), and the mesh is far from the true
surface. The likely cause is the model-free
CPU object mask (`backend/screwit/perception/object_mask.py`): it keeps clusters above the support
plane near the point the cameras orbit, and a thin, low-texture or see-through object on a textured
tabletop gives the depth model little to work with, so the clusters are wrong. This is unconfirmed;
the next step is to view the mask overlay for this scene and to run the remaining scenes, especially
the under-sink ones.

Until the remaining scenes run, there is no evidence that a real under-sink scan works.
