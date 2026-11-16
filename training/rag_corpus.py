"""Build the knowledge base for /lab/rag/: short passages about Vellmouth, a FICTIONAL delta town
(invented so a language model cannot answer from memory), plus passages about Vellbridge, a fictional
upstream town with similar-sounding facts that serve as distractors. Also writes labelled test questions.

    python3 training/rag_corpus.py          -> lab/rag/data/corpus.json
Then embed with:  node training/rag_embed.mjs [http://localhost:8765]
"""
import json, os

V = 'Vellmouth'
P = []  # (topic, text)

def add(topic, *texts):
    for t in texts:
        P.append((topic, t))

add('Geography',
    'Vellmouth is a fictional town at the mouth of the Sorrel River, where the river splits into seven channels before reaching the Grey Sea.',
    'The Tamsin Delta, on which Vellmouth stands, covers about 410 square kilometres of marsh, reed beds and tidal islands.',
    'The Sorrel River rises in the Hollin Hills, 212 kilometres inland, and carries fine silt that builds the delta outward by roughly 20 metres a year.',
    'Vellmouth is built on three low islands joined by causeways: Saltholm in the west, Mid Isle in the centre and Rushey in the east.',
    'The highest point in Vellmouth is Beacon Knoll on Mid Isle, only 9 metres above mean sea level.',
    'The widest of the seven delta channels is the Lantern Reach, which ships use to reach the Vellmouth harbour.',
    'The tidal range at Vellmouth is about 2.4 metres, and spring tides flood the outer reed beds twice a month.',
    'Vellmouth lies 38 kilometres downstream of the town of Vellbridge, the first point where the Sorrel can be crossed on foot at low water.',
    'The Grey Sea off Vellmouth is shallow for several kilometres, so large ships must wait for high tide before entering the Lantern Reach.',
    'Most of the delta is protected as the Tamsin Delta Reserve, except the three town islands and a strip of farmland on Rushey.',
)
add('History',
    'Vellmouth was founded in 1342 by salt makers who boiled seawater over peat fires on Saltholm.',
    'The town received its market charter in 1407, which allowed a weekly market every Thursday on Mid Isle.',
    'In 1588 a storm surge known locally as the Black Tide drowned the first harbour and forced the town to rebuild its quays on higher ground.',
    'The causeway between Saltholm and Mid Isle was completed in 1702 after nineteen years of work by the Causeway Guild.',
    'During the 1820s Vellmouth became known for reed paper, made from delta reeds pressed in water-powered mills on Rushey.',
    'The railway reached Vellmouth in 1871, ending at a terminus on Mid Isle called Vellmouth Quay station.',
    'The Great Flood of 1953 breached the Rushey dyke and covered two thirds of the town, but no lives were lost because the bells of St Brannoc rang a warning.',
    'After the 1953 flood the town built the Sorrel Barrier, a set of tidal gates that closes when a surge is forecast.',
    'The Sorrel Barrier was finished in 1961 and has closed 143 times since then.',
    'The old salt works on Saltholm closed in 1934 and the buildings now house the Vellmouth Salt Museum.',
    'Vellmouth was twinned with the fictional port of Ostervik in 1979.',
    'The town celebrated its 650th anniversary in 1992 with a regatta of 650 small boats on the Lantern Reach.',
)
add('Government',
    'Vellmouth is governed by a town council of 15 members elected every four years.',
    'The current mayor of Vellmouth is Ilse Marrable, first elected in 2021 and re-elected in 2025.',
    'Before Ilse Marrable, the mayor of Vellmouth was Tobias Wren, who served from 2009 to 2021.',
    'The town hall of Vellmouth is the Tidehouse, a brick building on Mid Isle that once served as the customs house.',
    'Vellmouth had a population of 18,400 at the last census in 2021.',
    'About 61 percent of Vellmouth residents live on Mid Isle, 24 percent on Saltholm and 15 percent on Rushey.',
    'The council meets on the first Tuesday of each month in the Long Room of the Tidehouse, and meetings are open to the public.',
    'The Vellmouth Harbour Board, separate from the council, manages moorings, dredging and pilotage on the Lantern Reach.',
    'The town motto of Vellmouth is "Salt and tide endure", carved above the door of the Tidehouse.',
    'Vellmouth\'s coat of arms shows three silver islands on a blue field with a gold salt pan above them.',
)
add('Economy',
    'Fishing is still important in Vellmouth: a fleet of about 40 boats lands mostly sole, plaice and brown shrimp.',
    'Vellmouth brown shrimp are boiled on board in seawater and sold peeled at the Thursday market.',
    'Salt making returned to Vellmouth in 2014 when a small company, Saltholm Flake, began producing hand-raked sea salt.',
    'Reed cutting for thatch is a winter trade in the delta; cutters work from December to March when the reeds are dry.',
    'Tourism brings about 260,000 visitors to Vellmouth each year, mostly in summer and during the Lantern Festival.',
    'The largest employer in Vellmouth is the St Brannoc Hospital on Mid Isle, with about 900 staff.',
    'A small boatyard on Rushey, Wren & Daughters, builds wooden delta punts with flat bottoms for shallow channels.',
    'The Thursday market on Mid Isle has about 70 stalls and has been held almost without interruption since 1407.',
    'Vellmouth harbour handles around 300 cargo ships a year, mainly grain and timber.',
    'Unemployment in Vellmouth was 4.1 percent in 2024, slightly below the regional average.',
)
add('Transport',
    'Three bridges cross the delta channels inside Vellmouth: the Causeway Bridge, the Rushey Swing Bridge and the footbridge known as the Needle.',
    'The Rushey Swing Bridge opens for ships about eight times a day, stopping road traffic for roughly ten minutes each time.',
    'The Needle is a pedestrian bridge opened in 2008 between Mid Isle and Rushey; it is 140 metres long.',
    'A passenger ferry, the Little Heron, runs every 30 minutes between Saltholm and the reserve visitor centre on Gull Island.',
    'The last Little Heron ferry leaves Gull Island at 18:30 in summer and 16:00 in winter.',
    'Trains run hourly from Vellmouth Quay station to Vellbridge and onward to the regional capital; the trip to Vellbridge takes 34 minutes.',
    'Cars are banned from the old streets of Saltholm; residents use bicycles and hand carts.',
    'Vellmouth has no airport; the nearest is 95 kilometres away.',
    'A bus route called the Island Loop connects all three islands every 15 minutes during the day.',
    'Bicycle hire is available at Vellmouth Quay station and at the Gull Island visitor centre.',
)
add('Culture',
    'The Lantern Festival is held in Vellmouth on the last Saturday of October, when hundreds of paper lanterns are floated down the Lantern Reach.',
    'Lanterns for the festival are traditionally made from Vellmouth reed paper and lit with beeswax candles.',
    'The Lantern Festival commemorates the night of the 1588 Black Tide, when survivors used lanterns to find each other on the water.',
    'Vellmouth\'s signature dish is shrimp kedge, brown shrimp cooked with rice, butter, nutmeg and samphire.',
    'Samphire, a salty marsh plant, is gathered on the delta flats in July and August and pickled for winter.',
    'The Vellmouth Brass Band was founded in 1889 and plays on the quay every Sunday afternoon in summer.',
    'St Brannoc\'s Church on Mid Isle has a bell tower 41 metres tall, the tallest structure in the town.',
    'The Salt Museum on Saltholm displays boiling pans, peat cutting tools and a reconstructed salt maker\'s hut.',
    'Every June the town holds the Punt Race, in which crews pole flat-bottomed punts 3 kilometres along the Needle Channel.',
    'The record time for the Punt Race is 11 minutes 52 seconds, set in 2019 by a crew from Rushey.',
    'Vellmouth\'s local newspaper is the Delta Courier, published every Friday since 1858.',
    'A traditional Vellmouth greeting between boat crews is "fair tide", answered with "and a soft keel".',
)
add('Education',
    'Vellmouth has four primary schools and one secondary school, Sorrel Academy, which has about 1,100 pupils.',
    'The Delta Observatory on Beacon Knoll was founded in 1896 to record tides and weather; it still publishes daily tide tables.',
    'The Delta Observatory has kept an unbroken daily tide record since 1 January 1897.',
    'The Vellmouth Library in the Tidehouse holds the town archive, including the 1407 market charter.',
    'The Tamsin Delta Field Station, run by a university, studies silt deposition and marsh birds; it opened in 1974.',
    'Researchers at the field station measured that the delta front advanced 1.9 kilometres between 1900 and 2000.',
    'Sorrel Academy runs a sailing programme in which every pupil learns to handle a dinghy before age 14.',
    'The Vellmouth Night School offers evening classes in reed thatching, net mending and the local dialect.',
)
add('Wildlife',
    'The Tamsin Delta Reserve is a breeding site for about 1,200 pairs of avocets, the reserve\'s emblem bird.',
    'Grey seals haul out on the sandbanks off Gull Island, where up to 600 can be counted in winter.',
    'The bittern, a secretive heron, returned to breed in the delta reed beds in 2003 after an absence of 60 years.',
    'Every autumn about 40,000 brent geese arrive from the Arctic to feed on eelgrass in the delta.',
    'Otters are found in all seven delta channels and are most often seen at dawn near the Needle.',
    'The reserve visitor centre on Gull Island has hides overlooking the main avocet colony.',
    'Fishing is banned in the reserve\'s inner lagoons between April and July to protect nesting birds.',
    'European eels migrate through the delta; a fish pass was added to the Sorrel Barrier in 2016 to help them.',
    'The marsh fritillary butterfly survives on the dry grassland of Beacon Knoll.',
    'Wardens count the avocet colony every May; the 2024 count was 1,214 pairs.',
)
add('Climate',
    'Vellmouth has a mild maritime climate; the average July temperature is 18 degrees Celsius and the average January temperature is 4 degrees.',
    'Annual rainfall in Vellmouth averages 640 millimetres, with the wettest month usually November.',
    'Sea fog, locally called the haar, rolls in from the Grey Sea on spring mornings and often lifts by noon.',
    'Sea level at Vellmouth has risen about 21 centimetres since the Delta Observatory began recording in 1897.',
    'The town\'s Climate Plan of 2022 commits to raising the Rushey dyke by 60 centimetres by 2035.',
    'The Sorrel Barrier closed a record 11 times during the winter of 2023 to 2024.',
    'Snow is rare in Vellmouth; the Delta Observatory records an average of three days with snow cover a year.',
    'Prevailing winds at Vellmouth blow from the south-west, and the strongest gales usually arrive between November and February.',
)
# Distractors: Vellbridge, a different fictional town upstream with look-alike facts.
add('Vellbridge (distractor)',
    'Vellbridge is a fictional market town 38 kilometres upstream of Vellmouth, at the first ford across the Sorrel River.',
    'Vellbridge was founded in 1290, about fifty years before Vellmouth, around a wooden bridge that gave the town its name.',
    'The mayor of Vellbridge is Corin Hask, elected in 2023.',
    'Vellbridge had a population of 26,900 at the 2021 census, making it larger than Vellmouth.',
    'Vellbridge holds a market every Saturday in the square in front of its stone bridge.',
    'The Vellbridge Lantern Walk takes place in early December and is unrelated to the Lantern Festival downstream.',
    'Vellbridge\'s signature dish is river trout baked in clay, quite different from the shrimp dishes of the coast.',
    'The railway reached Vellbridge in 1863, eight years before the line was extended to the coast.',
    'Vellbridge is home to the Sorrel Paper Mill, which has made paper from wood pulp since 1910.',
    'The town hall of Vellbridge is the Bridgehouse, built in 1756 beside the river crossing.',
)

docs = []
for i, (topic, text) in enumerate(P):
    docs.append({'id': f'P{i + 1:03d}', 'topic': topic, 'text': text})
by_text = {d['text']: d['id'] for d in docs}
def ids(*frags):
    out = []
    for f in frags:
        hits = [d['id'] for d in docs if f in d['text']]
        assert len(hits) == 1, (f, hits)
        out.append(hits[0])
    return out

queries = [
    {'q': 'Who is the mayor of Vellmouth?', 'answer': 'Ilse Marrable', 'gold': ids('The current mayor of Vellmouth')},
    {'q': 'When was Vellmouth founded?', 'answer': '1342', 'gold': ids('Vellmouth was founded in 1342')},
    {'q': 'What is floated down the Lantern Reach during the Lantern Festival?', 'answer': 'paper lanterns', 'gold': ids('The Lantern Festival is held')},
    {'q': 'How long does the train from Vellmouth to Vellbridge take?', 'answer': '34 minutes', 'gold': ids('Trains run hourly')},
    {'q': 'What warned people during the Great Flood of 1953?', 'answer': 'the bells of St Brannoc', 'gold': ids('The Great Flood of 1953')},
    {'q': 'What is the emblem bird of the Tamsin Delta Reserve?', 'answer': 'the avocet', 'gold': ids('breeding site for about 1,200 pairs')},
    {'q': 'What was the population of Vellmouth in 2021?', 'answer': '18,400', 'gold': ids('Vellmouth had a population of 18,400')},
    {'q': 'What is shrimp kedge made of?', 'answer': 'brown shrimp, rice, butter, nutmeg and samphire', 'gold': ids("Vellmouth's signature dish")},
    {'q': 'When does the last Little Heron ferry leave Gull Island in winter?', 'answer': '16:00', 'gold': ids('The last Little Heron ferry')},
    {'q': 'How many times has the Sorrel Barrier closed since 1961?', 'answer': '143', 'gold': ids('The Sorrel Barrier was finished in 1961')},
    {'q': 'What is the town motto of Vellmouth?', 'answer': 'Salt and tide endure', 'gold': ids('The town motto of Vellmouth')},
    {'q': 'What is the record time for the Punt Race?', 'answer': '11 minutes 52 seconds', 'gold': ids('The record time for the Punt Race')},
]

out = {
    'model': 'Xenova/all-MiniLM-L6-v2',
    'note': 'Vellmouth, Vellbridge, the Sorrel River and the Tamsin Delta are fictional; every passage was written for this demo.',
    'docs': docs,
    'queries': queries,
}
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(root, 'lab/rag/data'), exist_ok=True)
with open(os.path.join(root, 'lab/rag/data/corpus.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
print(len(docs), 'passages,', len(queries), 'queries')
