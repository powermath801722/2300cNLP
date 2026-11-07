// Rule-based lemmatizer in the style of WordNet's morphy: detach suffixes by part of speech,
// then keep only candidates found in a lexicon (here: the 30k-word Word2Vec vocabulary with corpus counts).

export const IRREGULAR = {
  verb: { am: 'be', is: 'be', are: 'be', was: 'be', were: 'be', been: 'be', being: 'be', has: 'have', had: 'have', having: 'have',
    did: 'do', done: 'do', does: 'do', went: 'go', gone: 'go', goes: 'go', saw: 'see', seen: 'see', ate: 'eat', eaten: 'eat',
    ran: 'run', came: 'come', took: 'take', taken: 'take', gave: 'give', given: 'give', made: 'make', said: 'say', knew: 'know',
    known: 'know', thought: 'think', brought: 'bring', bought: 'buy', caught: 'catch', taught: 'teach', found: 'find', told: 'tell',
    wrote: 'write', written: 'write', spoke: 'speak', spoken: 'speak', began: 'begin', begun: 'begin', sang: 'sing', sung: 'sing',
    swam: 'swim', swum: 'swim', drove: 'drive', driven: 'drive', rode: 'ride', ridden: 'ride', flew: 'fly', flown: 'fly',
    fell: 'fall', fallen: 'fall', felt: 'feel', kept: 'keep', left: 'leave', lost: 'lose', met: 'meet', paid: 'pay', sat: 'sit',
    stood: 'stand', understood: 'understand', won: 'win', wore: 'wear', worn: 'wear', got: 'get', gotten: 'get', held: 'hold',
    led: 'lead', meant: 'mean', sent: 'send', spent: 'spend', built: 'build', grew: 'grow', grown: 'grow', drew: 'draw', drawn: 'draw',
    chose: 'choose', chosen: 'choose', broke: 'break', broken: 'break', froze: 'freeze', frozen: 'freeze', lay: 'lie', lain: 'lie',
    rose: 'rise', risen: 'rise', shook: 'shake', shaken: 'shake', stole: 'steal', stolen: 'steal', threw: 'throw', thrown: 'throw',
    woke: 'wake', woken: 'wake', fought: 'fight', sought: 'seek', became: 'become', forgot: 'forget', forgotten: 'forget' },
  noun: { men: 'man', women: 'woman', children: 'child', mice: 'mouse', geese: 'goose', feet: 'foot', teeth: 'tooth', people: 'person',
    oxen: 'ox', data: 'datum', criteria: 'criterion', phenomena: 'phenomenon', analyses: 'analysis', theses: 'thesis', crises: 'crisis',
    matrices: 'matrix', indices: 'index', vertices: 'vertex', alumni: 'alumnus', cacti: 'cactus', fungi: 'fungus', radii: 'radius',
    wolves: 'wolf', knives: 'knife', wives: 'wife', lives: 'life', leaves: 'leaf', halves: 'half', selves: 'self', loaves: 'loaf' },
  adj: { better: 'good', best: 'good', worse: 'bad', worst: 'bad', more: 'many', most: 'many', less: 'little', least: 'little',
    further: 'far', furthest: 'far', farther: 'far', farthest: 'far', elder: 'old', eldest: 'old' },
};
// Words that look inflected but are their own lemma.
export const INVARIANT = new Set(['news', 'series', 'species', 'physics', 'mathematics', 'economics', 'politics', 'ethics', 'linguistics',
  'means', 'thus', 'bus', 'gas', 'plus', 'this', 'his', 'is', 'was', 'has', 'does', 'always', 'perhaps', 'yes', 'less', 'unless',
  'business', 'process', 'access', 'success', 'analysis', 'basis', 'crisis', 'thesis', 'status', 'virus', 'campus', 'focus', 'bonus',
  'during', 'nothing', 'something', 'anything', 'everything', 'thing', 'king', 'ring', 'spring', 'string', 'morning', 'evening',
  'ceiling', 'bed', 'red', 'need', 'seed', 'speed', 'feed', 'hundred', 'sacred', 'naked', 'wicked']);

// morphy-style detachment rules: [suffix, replacement]
const RULES = {
  noun: [['ies', 'y'], ['ses', 's'], ['xes', 'x'], ['zes', 'z'], ['ches', 'ch'], ['shes', 'sh'], ['men', 'man'], ['s', '']],
  verb: [['ying', 'ie'], ['ies', 'y'], ['es', 'e'], ['es', ''], ['s', ''], ['ied', 'y'], ['ed', 'e'], ['ed', ''], ['ing', 'e'], ['ing', '']],
  adj: [['ier', 'y'], ['iest', 'y'], ['er', ''], ['est', ''], ['er', 'e'], ['est', 'e']],
};
// Spelling rule: a single final consonant after a single vowel (hop-ing, car-ed, us-ed) usually means a silent e was dropped.
const cvcEnd = (s) => /(^|[^aeiou])[aeiou][b-df-hj-np-tv-z]$/.test(s);
const undouble = (s) => (/([b-df-hj-np-tv-z])\1$/.test(s) && !/(ll|ss|zz|ff)$/.test(s) ? s.slice(0, -1) : null);

/**
 * Lemmatize one word. lexicon: Map(word → count). pos: 'noun' | 'verb' | 'adj' | 'auto'.
 * Returns { lemma, pos, how, candidates:[{form, rule, count}] }
 */
export function lemmatize(word, lexicon, pos = 'auto') {
  const w = word.toLowerCase();
  const order = pos === 'auto' ? guessOrder(w) : [pos];
  const cands = [];
  for (const p of order) {
    if (IRREGULAR[p][w]) return { lemma: IRREGULAR[p][w], pos: p, how: 'irregular form (exception list)', candidates: [] };
  }
  if (INVARIANT.has(w)) return { lemma: w, pos: order[0], how: 'base form (exception list)', candidates: [] };
  for (const p of order) {
    for (const [suf, rep] of RULES[p]) {
      if (!w.endsWith(suf) || w.length - suf.length < (suf === 'ying' ? 1 : 2)) continue;
      const base = w.slice(0, w.length - suf.length);
      for (const f of [base + rep, rep === '' ? undouble(base) : null]) {
        if (!f || f === w) continue;
        if (p === 'noun' && suf === 's' && /(ss|us|is)$/.test(w)) continue;
        const count = lexicon.get(f) || 0;
        const eRestore = p === 'verb' && rep === 'e' && (suf === 'ing' || suf === 'ed') && cvcEnd(base) && count > 0;
        cands.push({ form: f, rule: `-${suf} → ${rep ? '-' + rep : '∅'}${f !== base + rep ? ' (undouble)' : ''}`, pos: p, count, eRestore });
      }
    }
    const known = cands.filter((c) => c.pos === p && c.count > 0);
    const eForm = known.find((c) => c.eRestore);
    if (eForm) return { lemma: eForm.form, pos: p, how: `${eForm.rule}; silent e restored (single consonant after a single vowel)`, candidates: cands };
    if (known.length) {
      known.sort((a, b) => b.count - a.count);
      return { lemma: known[0].form, pos: p, how: `${known[0].rule}; most frequent candidate in the lexicon`, candidates: cands };
    }
  }
  return { lemma: w, pos: order[0], how: cands.length ? 'no candidate found in the lexicon; left unchanged' : 'no rule applies; already a base form', candidates: cands };
}

function guessOrder(w) {
  if (/(ing|ed)$/.test(w)) return ['verb', 'adj', 'noun'];
  if (/(er|est)$/.test(w)) return ['adj', 'noun', 'verb'];
  return ['noun', 'verb', 'adj'];
}
