// RAG: retrieval finds the gold passage; the grounded answer is correct and supported; closed-book is not grounded.
export default async ({ page, check, warn }) => {
  const top = await page.evaluate(() => window.__lab.retrieve('Who is the mayor of Vellmouth?', 'dense', 3));
  check(top.includes('P024'), `dense retrieval puts the mayor passage P024 in the top 3 (got ${top.join(', ')})`);
  const bm = await page.evaluate(() => window.__lab.retrieve('What is the town motto of Vellmouth?', 'bm25', 1));
  check(bm[0] === 'P032', `BM25 top-1 for the motto question is P032 (got ${bm[0]})`);
  await page.fill('#q', 'Who is the mayor of Vellmouth?');
  await page.evaluate(() => { window.__lab.last = null; });
  await page.click('#run');
  await page.waitForFunction(() => window.__lab.last || window.__lab.runError, null, { timeout: 300000 });
  const r = await page.evaluate(() => window.__lab.last || { error: window.__lab.runError });
  check(!r.error, 'RAG run completes' + (r.error ? ': ' + r.error : ''));
  check(/marrable/i.test(r.answer || ''), `grounded answer names Ilse Marrable (got “${r.answer}”)`);
  check((r.verdicts || []).includes('supported'), `answer claim verified as supported (${(r.verdicts || []).join(', ')})`);
  check(!/marrable/i.test(r.closedBook || ''), `closed-book answer cannot know the fictional mayor (got “${r.closedBook}”)`);
  warn((r.closedVerdicts || []).every((v) => v !== 'supported'), 'closed-book claims are not supported by the passages');
  check(await page.locator('#passages li.used').count() >= 3, 'retrieved passages listed and marked as used');
  check((await page.textContent('#prompt')).includes('[P024]'), 'assembled prompt shows passage ids');
};
