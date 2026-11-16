// Dependency-free text helpers shared by the attribution, calibration and adversarial modules.

/** Word-level segmentation shared by the modules: words are features; punctuation stays fixed. */
export function segment(text) {
  const parts = text.match(/[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]/g) || [];
  return parts.map((t) => ({ t, word: /[A-Za-z0-9]/.test(t) }));
}
/** Rebuild text keeping parts where keep(i) is true (punctuation always kept). */
export function rebuild(parts, keep) {
  let s = '';
  parts.forEach((p, i) => {
    if (p.word && !keep(i)) return;
    s += (s && !/^[.,!?;:)'"]$/.test(p.t) ? ' ' : '') + p.t;
  });
  return s;
}

/** Small seeded RNG (mulberry32). */
export function rng(seed = 42) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
