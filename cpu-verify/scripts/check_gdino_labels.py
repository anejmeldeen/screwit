"""Record the type of results['labels'] from the HF Grounding DINO post-processor in the pinned transformers."""
from common import ROOT, save
from PIL import Image
import transformers
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

p = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
m = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
im = Image.open(ROOT / "data/sam_frames/00000.jpg").convert("RGB")
i = p(images=im, text="bed. pillow.", return_tensors="pt")
r = p.post_process_grounded_object_detection(m(**i), i.input_ids, box_threshold=0.35, text_threshold=0.25,
                                             target_sizes=[im.size[::-1]])[0]
out = {"transformers": transformers.__version__, "labels_type": str(type(r["labels"][0])),
       "labels": [str(x) for x in r["labels"]], "text_labels": [str(x) for x in r["text_labels"]]}
save("gdino_labels.json", out)
print(out)
