// Attribution: exact recovery on the linear models, redundancy behaviour, and a DistilBERT run.
export default async ({ page, check, warn }) => {
  await page.waitForFunction(() => window.__lab.runs >= 1, null, { timeout: 60000 });
  check(await page.locator('#heat .hrow').count() === 4, 'logistic regression shows exact + three method rows');
  const vs = await page.locator('#vsexact tbody tr').allInnerTexts();
  const rho = vs.map((r) => parseFloat(r.split('\t')[1]));
  check(rho[0] > 0.99 && rho[2] > 0.99, `occlusion and KernelSHAP match the exact LR attributions (ρ ${rho.join(', ')})`);
  check(rho[1] > 0.9, `LIME is close to exact (ρ ${rho[1]})`);
  check(await page.locator('#randtbl tbody tr').count() === 3, 'model-randomization sanity check is reported');

  // Redundancy: occlusion gives duplicated words 0 under binary features, SHAP splits the weight
  const r = await page.evaluate(async () => {
    const res = await window.__attr.explainAll('lr', 'good good film', { samples: 64, seed: 1, status: () => {} });
    return { occ: res.attr.occ, shap: res.attr.shap, exact: res.exact };
  });
  check(Math.abs(r.occ[0]) < 1e-9 && Math.abs(r.occ[1]) < 1e-9, 'occlusion assigns 0 to each copy of a duplicated word (redundancy)');
  check(Math.abs(r.shap[0] - r.exact[0]) < 1e-6 && Math.abs(r.shap[0] - r.shap[1]) < 1e-6, 'exact KernelSHAP splits the duplicated word’s weight equally');

  // Naive Bayes is additive: all methods recover the truth
  await page.selectOption('#model', 'nb');
  await page.click('#go');
  await page.waitForFunction(() => window.__lab.runs >= 2, null, { timeout: 60000 });
  const nb = (await page.locator('#vsexact tbody tr').allInnerTexts()).map((x) => parseFloat(x.split('\t')[2]));
  check(nb[0] < 1e-6 && nb[2] < 1e-6, `Naive Bayes: occlusion and SHAP errors vs exact are 0 (${nb.join(', ')})`);

  // DistilBERT end to end
  await page.selectOption('#model', 'bert');
  await page.selectOption('#budget', '64');
  await page.fill('#txt', 'A warm and funny film.');
  await page.click('#go');
  await page.waitForFunction(() => window.__lab.runs >= 3, null, { timeout: 300000 });
  check(/DistilBERT/.test(await page.textContent('#pred')), 'DistilBERT explanation rendered');
  const aopc = (await page.locator('#faith tbody tr').allInnerTexts()).map((x) => parseFloat(x.split('\t')[1]));
  check(aopc.every(Number.isFinite), 'faithfulness metrics are finite');
  warn(Math.max(...aopc.slice(0, 3)) > aopc[aopc.length - 1], `at least one method beats random ordering on AOPC (${aopc.join(', ')})`);
};
