"""Sample texts for the NLP Lab TextRank module.

Two U.S. presidential inaugural addresses (public domain, NLTK 'inaugural' corpus) and one short
article written for this lab.

Usage: python training/textrank_samples.py path/to/inaugural_dir lab/textrank/data/samples.json
"""
import json, os, re, sys

src, out = sys.argv[1], sys.argv[2]

ARTICLE = """How River Deltas Are Built

A river delta forms where a river meets a body of slower water and drops the sediment it has been carrying. As the current slows, the heaviest grains settle first, near the river mouth, while fine silt and clay travel farther before they sink. Over centuries these deposits build a fan of new land that pushes out into the sea or lake.

Deltas grow by a process called avulsion. When a channel fills with sediment, the river eventually breaks through its banks during a flood and finds a shorter, steeper path to the sea. The abandoned channel slowly subsides, and the new channel begins building land in a different place. Seen from above, a delta is a record of many such switches stacked on top of one another.

The shape of a delta depends on which force is strongest at the coast. River-dominated deltas, such as the Mississippi, extend long fingers of land because waves and tides are too weak to rework the sediment. Wave-dominated deltas, such as the Nile, are smoothed into a gentle arc by waves that spread sediment along the shore. Tide-dominated deltas, such as the Ganges–Brahmaputra, are cut by many channels that tides keep open as they flow in and out twice a day.

Deltas are among the most fertile and densely populated places on Earth. Their soils are renewed by floods, their channels make natural transport routes, and their wetlands shelter fish, birds and shellfish. Hundreds of millions of people live on deltas, often in large port cities.

Many deltas are now shrinking. Dams upstream trap the sediment that once reached the coast, levees stop floods from spreading new soil across the land, and pumping groundwater makes the ground sink. At the same time, sea level is rising. Engineers are testing controlled diversions that let a river flood selected areas again, so that sediment can rebuild marshes faster than the sea can drown them.
"""

def address(name):
    raw = open(os.path.join(src, name), encoding="utf-8", errors="replace").read()
    return re.sub(r"[ \t]+", " ", raw).strip()

samples = [
    {"id": "deltas", "title": "How River Deltas Are Built (article written for this lab)", "text": ARTICLE.strip()},
    {"id": "lincoln", "title": "Abraham Lincoln, Second Inaugural Address (1865)", "text": address("1865-Lincoln.txt")},
    {"id": "kennedy", "title": "John F. Kennedy, Inaugural Address (1961)", "text": address("1961-Kennedy.txt")},
]
json.dump({"note": "Inaugural addresses are U.S. government works in the public domain.", "samples": samples},
          open(out, "w"), ensure_ascii=False, indent=0)
print([(s["id"], len(s["text"].split())) for s in samples])
