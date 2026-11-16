// Sample documents for Text Studio. Lincoln's address is loaded from the TextRank module's public-domain samples.
export const SAMPLES = [
  {
    id: 'news', label: 'News report', note: 'Written for Text Studio; the town, people and project are fictional.',
    text: `Council approves flood barrier for Riverside district

Riverside, Nov. 12 — The city council voted 7 to 2 on Tuesday to build a movable flood barrier along the lower Braid River, ending a debate that began after the 2024 floods damaged more than 300 homes.

The barrier will be designed by the U.S. Army Corps of Engineers and is expected to cost $48 million, according to Mayor Elena Ruiz. Construction is scheduled to begin next spring and should be completed by 2028.

"This protects families who were told for years that nothing could be done," Ruiz said after the vote. Two council members, Daniel Okafor and Priya Natarajan, voted against the plan, arguing that the money would be better spent on wetland restoration upstream.

Residents at the meeting were divided. Some applauded the decision, while others worried that the barrier would push floodwater toward neighboring towns. Farmers in Delta County said they had not been consulted. The council said an independent review by Delta State University will be published before construction begins, and that the full engineering plans will be posted online next month.`,
  },
  {
    id: 'review', label: 'Product review', note: 'Written for Text Studio, with three deliberate typos.',
    text: `I bought the Riverline kettle three weeks ago and I really wanted to love it. The design is gorgeous, the brushed steel looks great on the counter, and it boils a full litre in under four minutes.

Unfortunately the lid hinge broke on day nine. Customer service was polite but slow, and I had to wait twelve days to recieve a replacement part. The new hinge works fine so far.

The temperature presets are definately the best feature: green tea at eighty degrees is perfect every time. However, the base gets noticeably hot and the batery in the little display died after a week.

Would I buy it again? Probably not at this price. If they fix the build quality, it could be excellent.`,
  },
  {
    id: 'walden', label: 'Thoreau, Walden (1854)', note: 'From “Where I Lived, and What I Lived For”. Public domain.',
    text: `I went to the woods because I wished to live deliberately, to front only the essential facts of life, and see if I could not learn what it had to teach, and not, when I came to die, discover that I had not lived. I did not wish to live what was not life, living is so dear; nor did I wish to practise resignation, unless it was quite necessary. I wanted to live deep and suck out all the marrow of life, to live so sturdily and Spartan-like as to put to rout all that was not life, to cut a broad swath and shave close, to drive life into a corner, and reduce it to its lowest terms, and, if it proved to be mean, why then to get the whole and genuine meanness of it, and publish its meanness to the world; or if it were sublime, to know it by experience, and be able to give a true account of it in my next excursion.`,
  },
  { id: 'lincoln', label: 'Lincoln, Second Inaugural (1865)', note: 'U.S. government work, public domain.', fromLab: '/lab/textrank/data/samples.json' },
];
