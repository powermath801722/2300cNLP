"""Fetch the GLUE SST-2 validation split (872 labelled sentences) for the calibration and adversarial modules.

    pip install pyarrow
    python training/calibration_build.py lab/calibration/data/sst2.json
    node training/calibration_logits.mjs http://localhost:8765 lab/calibration/data/sst2.json   # adds DistilBERT logits

SST-2 (Socher et al. 2013) as distributed in GLUE (Wang et al. 2018), from huggingface.co/datasets/nyu-mll/glue.
"""
import io, json, sys, urllib.request
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/sst2/validation-00000-of-00001.parquet"
out = sys.argv[1]
t = pq.read_table(io.BytesIO(urllib.request.urlopen(URL).read())).to_pydict()
data = {
    "source": "GLUE SST-2 validation split (Socher et al. 2013; Wang et al. 2018), via huggingface.co/datasets/nyu-mll/glue",
    "sentence": [s.strip() for s in t["sentence"]],
    "label": [int(x) for x in t["label"]],
}
json.dump(data, open(out, "w"), separators=(",", ":"))
print(len(data["sentence"]), "sentences,", sum(data["label"]), "positive")
