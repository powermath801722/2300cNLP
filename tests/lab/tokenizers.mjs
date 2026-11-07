// Tokenizers module: BPE learns the textbook merges; pretrained tokenizers produce sensible counts.
export default async ({ page, check, warn }) => {
  const b0 = await page.evaluate(() => window.__lab.bpe);
  check(b0 && b0.merges >= 5, `BPE learns merges on the toy corpus (${b0 && b0.merges})`);
  const first = await page.$$eval('#mergetab tbody tr', (rows) => rows.slice(0, 3).map((r) => r.children[2].textContent));
  // Sennrich toy corpus {low:5, lower:2, newest:6, widest:3}: first merges are e+s, es+t, est+</w>
  check(first[0] === 'es' && first[1] === 'est' && first[2] === 'est</w>', `first merges match Sennrich et al. (${first.join(', ')})`);
  await page.fill('#k', '0'); await page.dispatchEvent('#k', 'input');
  const chars = await page.evaluate(() => window.__lab.bpe.held.length);
  await page.evaluate(() => { const k = document.getElementById('k'); k.value = k.max; k.dispatchEvent(new Event('input')); });
  const full = await page.evaluate(() => window.__lab.bpe);
  check(full.held.length < chars, `held-out "lowest" gets shorter with all merges (${chars} → ${full.held.length}: ${full.held.join(' ')})`);
  await page.click('#step');
  await page.click('#c-prose');
  const prose = await page.evaluate(() => window.__lab.bpe);
  check(prose.merges > 50, `prose corpus learns many merges (${prose.merges})`);

  await page.click('#cmpgo');
  await page.waitForFunction(() => window.__lab.compare, null, { timeout: 180000 });
  const c = await page.evaluate(() => window.__lab.compare);
  check(c['GPT-2'].n > 5 && c['BERT (uncased)'].n > 5 && c['T5'].n > 5, `all three tokenizers produce tokens (${c['GPT-2'].n}/${c['BERT (uncased)'].n}/${c['T5'].n})`);
  check(c['GPT-2'].unk === 0, 'GPT-2 byte-level BPE has no unknown tokens');
  await page.click('#cmpchips button:has-text("Emoji")');
  await page.waitForFunction(() => window.__lab.compare && window.__lab.compare['GPT-2'].pieces.join('').length > 0 && /NLP|NL/.test(window.__lab.compare['GPT-2'].pieces.join('')), null, { timeout: 60000 });
  const e = await page.evaluate(() => window.__lab.compare);
  warn(e['BERT (uncased)'].unk > 0 || e['T5'].unk > 0, `emoji produce unknown tokens in BERT or T5 (BERT ${e['BERT (uncased)'].unk}, T5 ${e['T5'].unk})`);
  check(await page.locator('#cmpout .panel').count() === 3, 'three tokenizer cards rendered');
};
