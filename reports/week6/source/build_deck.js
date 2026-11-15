// Week 6 deck generator. Numbers are read from reports/week6/p4_validation/presentation_claims.json (independently verified).
// Usage: NODE_PATH=<dir with pptxgenjs> node build_deck.js <repo_root> <apply_theme.js path>
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');
const ROOT = process.argv[2];
const { applyTheme } = require(process.argv[3]);
const W6 = path.join(ROOT, 'reports/week6');
const FIG = path.join(W6, 'figures');
const OUTDIR = path.join(W6, 'presentation');
fs.mkdirSync(OUTDIR, { recursive: true });
const C = JSON.parse(fs.readFileSync(path.join(W6, 'p4_validation/presentation_claims.json')));
const f3 = (x) => Number(x).toFixed(3);

// ---- style (Week 4 reference: dark title slide, green accent, light grey cards, Calibri)
const GREEN = '1F8A3B', DARK = '2B2B2B', DARK2 = '3A3A3A', INK = '1A1A1A', MUTED = '5F5F5F', CARD = 'F1F4F1', LINE = 'D9DED9', WHITE = 'FFFFFF', TINT = 'E3F1E6';
const THEME = { name: 'Week6 Green', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: INK, lt1: WHITE, dk2: DARK, lt2: CARD, accent1: GREEN, accent2: '2A78D6', accent3: 'EB6834', accent4: 'EDA100', accent5: '6250D6', accent6: '5F5F5F', hlink: GREEN, folHlink: '5F5F5F' } };
const FOOTER = 'Clustering meets Vision & Language · Topic 1 · Week 6 (16.11.2026)';
const W = 11.69, H = 8.27, MX = 0.55;

const pres = new pptxgen();
pres.defineLayout({ name: 'A4_LANDSCAPE', width: W, height: H });
pres.layout = 'A4_LANDSCAPE';
pres.theme = { headFontFace: 'Calibri', bodyFontFace: 'Calibri' };
pres.title = 'Week 6 status update: Visual Form vs. Acoustic Function in Instrument Clustering';
pres.author = 'Tim Henninger, Christian Krengel, Aiman Alkafahje';

pres.defineSlideMaster({ title: 'CONTENT', background: { color: WHITE },
  objects: [{ text: { text: FOOTER, options: { x: MX, y: H - 0.42, w: 8, h: 0.25, fontSize: 9, color: MUTED, margin: 0 } } }],
  slideNumber: { x: W - MX - 0.5, y: H - 0.42, w: 0.5, h: 0.25, fontSize: 9, color: MUTED, align: 'right' },
  placeholders: [] });
pres.defineSlideMaster({ title: 'DARK', background: { color: DARK }, objects: [] });

// pixel sizes for aspect-correct placement
function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function img(slide, name, x, y, maxW, maxH, align = 'left') {
  const p = path.join(FIG, name); const s = pngSize(p); const r = s.w / s.h;
  let w = maxW, h = w / r; if (h > maxH) { h = maxH; w = h * r; }
  const xx = align === 'center' ? x + (maxW - w) / 2 : x;
  slide.addImage({ path: p, x: xx, y, w, h, altText: name.replace('.png', '').replace(/_/g, ' ') });
  return { x: xx, y, w, h };
}
function title(slide, t) { slide.addText(t, { x: MX, y: 0.35, w: W - 2 * MX, h: 0.6, fontSize: 28, bold: true, color: INK, margin: 0, isTextBox: true, valign: 'middle' }); }
function caption(slide, t, y = H - 0.8) { slide.addText(t, { x: MX, y, w: W - 2 * MX, h: 0.36, fontSize: 10, color: MUTED, margin: 0, isTextBox: true, valign: 'top' }); }
function card(slide, x, y, w, h, fill = CARD) { slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.06 }); }
function statCard(slide, x, y, w, h, big, text, bigSize = 26) {
  card(slide, x, y, w, h);
  slide.addText(big, { x: x + 0.18, y: y + 0.1, w: w - 0.36, h: 0.55, fontSize: bigSize, bold: true, color: GREEN, margin: 0, isTextBox: true, valign: 'middle' });
  slide.addText(text, { x: x + 0.18, y: y + 0.68, w: w - 0.36, h: h - 0.78, fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: 'top' });
}
function textCard(slide, x, y, w, h, head, body, fill = CARD, bodySize = 12) {
  card(slide, x, y, w, h, fill);
  slide.addText(head, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: 0.32, fontSize: 14, bold: true, color: GREEN, margin: 0, isTextBox: true, valign: 'top' });
  slide.addText(body, { x: x + 0.18, y: y + 0.5, w: w - 0.36, h: h - 0.6, fontSize: bodySize, color: INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 3 });
}
const sections = ['Overview', 'Results', 'Outlook', 'Backup'];

// =============== 1 Title
pres.addSection({ title: 'Overview' });
let s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Overview' });
s.addText('Topic 1 · Week 6 Update · 16.11.2026', { x: 0.9, y: 1.15, w: 9, h: 0.35, fontSize: 14, bold: true, color: '3DBB5E', margin: 0, isTextBox: true });
s.addText('Visual Form vs. Acoustic Function in Instrument Clustering', { x: 0.9, y: 1.6, w: 9.9, h: 1.3, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true, valign: 'top' });
s.addText('Topic 1: Benchmarking Clustering Methods on VLM Embeddings', { x: 0.9, y: 2.95, w: 9.9, h: 0.35, fontSize: 15, color: 'D0D0D0', margin: 0, isTextBox: true });
card(s, 0.9, 3.6, 9.9, 1.75, DARK2);
s.addText([
  { text: 'Main research question', options: { fontSize: 12, color: 'BDBDBD', breakLine: true } },
  { text: 'Which unsupervised clustering method and preprocessing pipeline best recover the expert Hornbostel–Sachs classification of musical instruments from frozen Vision/VLM embeddings, and can the number of clusters be chosen without labels?', options: { fontSize: 15, color: WHITE } }],
  { x: 1.15, y: 3.75, w: 9.4, h: 1.5, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 4 });
s.addText('Today: a verified development pilot (2,000 images): methods, backbones, blind k, preprocessing.', { x: 0.9, y: 5.55, w: 9.9, h: 0.35, fontSize: 13, color: '3DBB5E', margin: 0, isTextBox: true });
s.addText([{ text: 'Tim Henninger · Christian Krengel · Aiman Alkafahje', options: { breakLine: true } }, { text: 'Clustering meets Vision and Language · Dr. Udo Schlegel · LMU Munich' }],
  { x: 0.9, y: 6.4, w: 9.9, h: 0.7, fontSize: 12, color: 'BDBDBD', margin: 0, isTextBox: true, paraSpaceAfter: 4 });
s.addNotes(`(~45 s) Week 4 we presented the plan; today we show a verified working benchmark. Our question: can unsupervised clustering of frozen image embeddings recover the expert Hornbostel–Sachs (HS) taxonomy, which classifies instruments by how they produce sound, while images mainly show form. Today's results are all on a 2,000-image development pilot drawn from MIMO/Europeana. They are development results, not held-out estimates: I will say this several times because it matters for how to read every number.`);

// =============== 2 Progress
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
title(s, '1 · What we accomplished since Week 4');
const items = [
  ['Foundation verified', '816 / 816 earlier scores reproduced exactly from saved assignments'],
  ['Pilot dataset audited', `${C.dataset.images.toLocaleString('en')} images · ${C.dataset.museums} museums · ${C.dataset.hs3_n.toLocaleString('en')} HS3 labels (${C.dataset.hs3_classes} classes)`],
  ['Two frozen backbones', 'CLIP ViT-B/32 (512-d) and SigLIP ViT-B/16 (768-d)'],
  ['Five clustering families', 'k-Means · Agglomerative · GMM · Spectral · HDBSCAN'],
  ['Blind k', 'Silhouette, elbow, BIC, eigengap: labels only for scoring'],
  ['Robustness & preprocessing', '5 × 80% subsets; 5 representations × 3 methods'],
];
items.forEach(([h, b], i) => {
  const y = 1.25 + i * 0.86;
  s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.05, w: 0.36, h: 0.36, fill: { color: GREEN }, line: { color: GREEN } });
  s.addText('✓', { x: MX, y: y + 0.05, w: 0.36, h: 0.36, fontSize: 14, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText([{ text: h, options: { bold: true, fontSize: 15, color: INK, breakLine: true } }, { text: b, options: { fontSize: 13, color: MUTED } }],
    { x: MX + 0.55, y, w: 5.6, h: 0.75, margin: 0, isTextBox: true, valign: 'top' });
});
const sx = 7.05, sw = 2.0, sh = 1.75;
statCard(s, sx, 1.25, sw, sh, '1,556', 'clustering fits (P2 + P3), all assignments saved');
statCard(s, sx + sw + 0.15, 1.25, sw, sh, '299 / 299', 'independent checks of the numbers in this talk', 22);
statCard(s, sx, 1.25 + sh + 0.15, sw, sh, '0', 'HS labels used to fit clusters or choose k');
statCard(s, sx + sw + 0.15, 1.25 + sh + 0.15, sw, sh, `${C.siglip_vs_clip.wins} / ${C.siglip_vs_clip.cells}`, 'matched cells where SigLIP beats CLIP (HS3 AMI)', 24);
textCard(s, sx, 5.05, 2 * sw + 0.15, 1.15, 'Status', 'Development pilot, already explored: not a held-out test. DINOv2 and the 27k benchmark are not yet run.', TINT, 12);
caption(s, 'Sources: reports/week6/P1_FOUNDATION_AUDIT.md, P2_RQ1_FINDINGS.md, P3_RQ2_FINDINGS.md, p4_validation/checks.csv.');
s.addNotes(`(~1 min) Since Week 4 we first audited what already existed. Every earlier score, 816 rows, was recomputed exactly from the saved cluster assignments, and the dataset counts were confirmed: 2,000 images from 216 museums, 1,549 with an HS level-3 label, 22 classes. On top of that we ran P2 (five clustering families, two backbones, blind k) and P3 (subset robustness and a preprocessing ablation): 1,556 fits in total. Labels were never used to fit clusters or to choose k; they enter only in a separate scoring step, after the selection files are written. Finally, a separate script re-derived every number in this talk from the raw assignment files: 299 of 299 checks pass. What is not done: DINOv2 and the large 27k benchmark.`);

// =============== 3 Pipeline
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
title(s, '2 · Our working benchmark pipeline');
s.addText('Blind pipeline: labels never form clusters or choose k; they are used only to score.', { x: MX, y: 1.05, w: 10, h: 0.3, fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true });
const boxes = [
  ['1  MIMO images', ['2,000-image pilot', '216 museums, round-robin', 'white-square padding'], 'verified'],
  ['2  Frozen encoders', ['CLIP ViT-B/32 · 512-d', 'SigLIP ViT-B/16 · 768-d', 'CPU fp32, hashed caches'], 'frozen · verified'],
  ['3  Preprocessing', ['raw L2 · centered + L2', 'PCA50 (± L2) · PCA100 + L2', 'global centering only'], 'no labels'],
  ['4  Clustering', ['k-Means · Agglom. · GMM', 'Spectral · HDBSCAN', 'blind k: 4 criteria'], 'no labels'],
  ['5  Evaluation', ['AMI · NMI · ARI', 'vs HS1 / HS2 / HS3', 'silhouette, cluster sizes'], 'labels used only here'],
];
const bw = 1.95, bg = 0.18, by = 1.55, bh = 2.25;
boxes.forEach(([h, lines, tag], i) => {
  const x = MX + i * (bw + bg);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: i === 4 ? 'F4F4F4' : CARD }, line: { color: i === 4 ? '444444' : GREEN, width: 1.5 }, rectRadius: 0.06 });
  s.addText(h, { x: x + 0.12, y: by + 0.12, w: bw - 0.24, h: 0.32, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(lines.map((l, j) => ({ text: l, options: { breakLine: j < lines.length - 1 } })), { x: x + 0.12, y: by + 0.55, w: bw - 0.24, h: 1.05, fontSize: 11.5, color: INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 2 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: by + bh - 0.45, w: bw - 0.4, h: 0.3, fill: { color: i === 4 ? '333333' : GREEN }, line: { color: i === 4 ? '333333' : GREEN }, rectRadius: 0.05 });
  s.addText(tag, { x: x + 0.2, y: by + bh - 0.45, w: bw - 0.4, h: 0.3, fontSize: 10.5, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  if (i < 4) s.addText('→', { x: x + bw - 0.02, y: by + bh / 2 - 0.2, w: bg + 0.04, h: 0.4, fontSize: 16, color: MUTED, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + bw + bg, y: by + bh + 0.15, w: bw, h: 0.62, fill: { color: WHITE }, line: { color: '9A9A9A', width: 1, dashType: 'dash' }, rectRadius: 0.06 });
s.addText([{ text: 'DINOv2 S / B / L', options: { bold: true, breakLine: true } }, { text: 'code ready · weights pending' }], { x: MX + bw + bg + 0.1, y: by + bh + 0.18, w: bw - 0.2, h: 0.56, fontSize: 11, color: MUTED, margin: 0, isTextBox: true, align: 'center', valign: 'middle' });
textCard(s, MX, 4.75, 5.2, 2.0, 'Evaluation protocol (declared before fitting)',
  [{ text: 'HS1 / HS2 / HS3: 1,565 / 1,563 / 1,549 labelled rows; all 2,000 rows are clustered', options: { bullet: true, breakLine: true } },
   { text: 'HS3: 22 classes, largest 18.5%, no frequency filter', options: { bullet: true, breakLine: true } },
   { text: 'Fixed k = 10 (anchor); k grid {2, 3, 5, 8, 10, 15, 20}', options: { bullet: true, breakLine: true } },
   { text: 'Seeds 0–4: mean ± SD, never the best seed', options: { bullet: true } }], CARD, 12);
textCard(s, MX + 5.4, 4.75, 5.19, 2.0, 'What changed vs. the Week 4 plan',
  [{ text: 'Planned: 27,348 scored objects, 13 HS3 classes, held-out split; not available locally', options: { bullet: true, breakLine: true } },
   { text: 'Used now: 2,000-image pilot, 22 HS3 classes, already explored → development results only', options: { bullet: true, breakLine: true } },
   { text: 'The plan’s 0.53 / 0.83 / 0.68 are not comparable with pilot AMIs', options: { bullet: true } }], TINT, 12);
caption(s, 'Protocol: P1_EXPERIMENT_PROTOCOL.json, P3_EXPERIMENT_PROTOCOL.json. Encoders: laion/CLIP-ViT-B-32-laion2B-s34B-b79K; timm/ViT-B-16-SigLIP.', H - 0.95);
s.addNotes(`(~1 min) The pipeline is the one from Week 4, now verified end to end. Images are padded to a white square so long instruments are not cropped. Two frozen encoders, CLIP and SigLIP. Five representations, from raw L2 to centering and PCA. Five clustering families plus four blind k criteria. Evaluation with AMI, NMI and ARI against HS levels 1 to 3: labels enter only here. The protocol was written to a file before any fit. Important change from the plan: the large 27,348-object benchmark with 13 HS3 classes and its held-out split is not available locally, so all results today are on the 2,000-image pilot with 22 HS3 classes. That is why we do NOT compare our numbers with the plan's 0.53 baseline, 0.83 probe or 0.68 target: different data, different classes, different encoder. DINOv2 code is ready, but the weights were not downloaded yet.`);

// =============== 4 RQ1 methods
pres.addSection({ title: 'Results' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '3 · RQ1: Which clustering method works best?');
img(s, 'P2_METHOD_COMPARISON.png', MX, 1.1, 7.3, 3.6);
const rows = [['Method (k = 10)', 'CLIP AMI', 'SigLIP AMI', 'SigLIP ARI']];
[['kmeans', 'k-Means'], ['agglomerative_average_cosine', 'Agglomerative'], ['gmm_diag', 'GMM (diag.)'], ['spectral_knn', 'Spectral (15-NN)']].forEach(([k, n]) => {
  const a = C[`clip_${k}_k10`], b = C[`siglip_${k}_k10`];
  rows.push([n, f3(a.AMI), f3(b.AMI), f3(b.ARI)]);
});
rows.push(['HDBSCAN (mcs 20)*', f3(C.clip_hdbscan20.AMI_noise_as_cluster), f3(C.siglip_hdbscan20.AMI_noise_as_cluster), '–']);
const tb = rows.map((r, i) => r.map((v, j) => ({ text: v, options: { bold: i === 0 || (i === 4 && j === 2) || (i === 2 && j === 3), color: (i === 4 && j === 2) || (i === 2 && j === 3) ? GREEN : INK, fill: { color: i === 0 ? CARD : WHITE }, align: j ? 'right' : 'left' } })));
s.addTable(tb, { x: MX, y: 4.85, w: 7.3, colW: [2.5, 1.6, 1.6, 1.6], rowH: 0.3, fontSize: 12, fontFace: 'Calibri', border: { type: 'solid', pt: 0.5, color: LINE }, margin: [0.02, 0.08, 0.02, 0.08] });
const rx = 8.1, rw = W - MX - rx;
statCard(s, rx, 1.1, rw, 1.3, f3(C.siglip_spectral_knn_k10.AMI), `HS3 AMI, SigLIP + Spectral (5 seeds; subset mean ${f3(C.robust_siglip.spectral_AMI_mean)})`);
statCard(s, rx, 2.55, rw, 1.3, `${f3(C.siglip_agglomerative_average_cosine_k10.ARI)} > ${f3(C.siglip_spectral_knn_k10.ARI)}`, 'HS3 ARI: Agglomerative beats Spectral (10/10 subsets)', 22);
statCard(s, rx, 4.0, rw, 1.3, `${C.siglip_vs_clip.wins} / ${C.siglip_vs_clip.cells}`, 'matched cells where SigLIP > CLIP (all 4 methods × 7 k × 2 repr.)');
card(s, rx, 5.45, rw, 1.45, TINT);
s.addText([{ text: 'Key finding', options: { bold: true, color: GREEN, breakLine: true, fontSize: 13 } }, { text: 'The algorithm ranking depends on the evaluation metric.', options: { bold: true, fontSize: 14, color: INK } }],
  { x: rx + 0.18, y: 5.55, w: rw - 0.36, h: 1.25, margin: 0, isTextBox: true, valign: 'top' });
caption(s, `Centered + L2, HS3 n = 1,549 (22 classes), mean of 5 seeds (agglomerative/HDBSCAN deterministic). *HDBSCAN: noise counted as a cluster; ${C.hdbscan_noise_range[0]}–${C.hdbscan_noise_range[1]}% of images unassigned. Pilot, not held-out; best configuration chosen after seeing results.`, H - 0.95);
s.addNotes(`(~1.5 min) Left: HS3 AMI against the number of clusters, CLIP left panel, SigLIP right. AMI is 0 for random and 1 for perfect agreement with the expert classes. At our predeclared k = 10, SigLIP + Spectral clustering on a 15-nearest-neighbour graph reaches ${f3(C.siglip_spectral_knn_k10.AMI)} (mean of 5 seeds), against ${f3(C.siglip_kmeans_k10.AMI)} for k-Means. But look at ARI in the table: Agglomerative has ${f3(C.siglip_agglomerative_average_cosine_k10.ARI)}, Spectral only ${f3(C.siglip_spectral_knn_k10.ARI)}. Agglomerative wins on ARI in all 10 subset runs. So the ranking depends on the metric; the next slide explains why. Backbone: SigLIP beats CLIP in ${C.siglip_vs_clip.wins} of ${C.siglip_vs_clip.cells} matched comparisons, the clearest result we have. HDBSCAN is not competitive: it leaves 59–70% of images as noise. Caveat: the 0.544 is from the full pilot; on five 80% subsets Spectral averages ${f3(C.robust_siglip.spectral_AMI_mean)}, and we picked Spectral after seeing these results.
Likely question: 'Is Spectral significantly better?' No significance test: the comparisons share images. Spectral beats k-Means in 9/10 subset runs, but beats Agglomerative only for SigLIP (5/5), not for CLIP (2/5).`);

// =============== 5 AMI vs ARI interpretation
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '4 · Why AMI and ARI disagree: form over function');
img(s, 'P3_MIXED_CLUSTER_EXAMPLES.png', MX, 1.05, W - 2 * MX, 4.3, 'center');
const ca = C.amiari_spectral_knn, cg = C.amiari_agglomerative_average_cosine;
const cw = (W - 2 * MX - 0.3) / 3;
textCard(s, MX, 5.5, cw, 1.6, 'Catch-all cluster → low ARI', `${ca.largest_all_rows} images (${ca.largest_all_pct}%), ${ca.largest_n_classes} HS3 classes, purity ${ca.largest_purity}. It holds ${Math.round(ca.share_pairs_largest * 100)}% of all same-cluster pairs, so pair precision is ${ca.pair_precision}.`);
textCard(s, MX + cw + 0.15, 5.5, cw, 1.6, 'Pure clusters → high AMI', 'Brass 97–100%, lutes 91–99%, harps 90% pure: 16 of 22 classes have ≥70% of their images in one cluster.');
textCard(s, MX + 2 * (cw + 0.15), 5.5, cw, 1.6, 'Long tubes merge', 'Reedpipes 52% + flutes 46% in one cluster; Agglomerative merges them too (46% / 45%). Same form, different sound production.');
caption(s, `SigLIP, centered + L2, k = 10, seed 0: Spectral AMI ${f3(ca.AMI)} / ARI ${f3(ca.ARI)}; Agglomerative ${f3(cg.AMI)} / ${f3(cg.ARI)}. Descriptive, one partition; HS3 codes as catalogued. Photos: MIMO partner museums, CC BY-NC-SA 4.0.`, H - 0.95);
s.addNotes(`(~75 s) This is the interpretation slide. Top: images from the two most mixed Spectral clusters, sampled at random within their main classes. Row 1 is a catch-all cluster of small, hand-held objects: vessel flutes, panpipes and whistles (HS 421, edge-blown), castanets and bells (111), rattles (112). That one cluster has ${ca.largest_all_rows} images, ${ca.largest_purity} purity, and contains ${Math.round(ca.share_pairs_largest * 100)}% of all pairs of images that share a cluster. ARI counts pairs, so it grows with the square of cluster size: this one cluster pulls ARI down. AMI is information-based: Spectral's other clusters are very pure (brass, lutes, harps), which gives the most mutual information, so AMI goes up. Row 2 is the most interesting for our project: long tubes. Clarinets, oboes and bagpipes (reedpipes) sit with flutes and fifes, and even a 'Russian bassoon' and a serpent, which are lip-reed (brass) instruments. The images group by visual form; Hornbostel–Sachs groups by how the sound is made. Caveat: one partition, descriptive; a couple of catalogue codes look odd (two 'Tambour' records coded as struck idiophones), so some label noise is likely.`);

// =============== 6 Blind k
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '5 · Can we choose k without labels?');
img(s, 'P3_BLIND_K_ROBUSTNESS.png', MX, 1.1, 7.15, 4.3);
const bkx = 7.95, bkw = W - MX - bkx;
card(s, bkx, 1.1, bkw, 2.35);
s.addText('Full pilot (P2): silhouette picks', { x: bkx + 0.18, y: 1.2, w: bkw - 0.36, h: 0.3, fontSize: 14, bold: true, color: GREEN, margin: 0, isTextBox: true });
const bt = [['', 'CLIP', 'SigLIP'], ['k-Means', `k = ${C.blindk_clip_kmeans.k}`, `k = ${C.blindk_siglip_kmeans.k}`], ['Agglomerative', `k = ${C.blindk_clip_agglomerative_average_cosine.k}`, `k = ${C.blindk_siglip_agglomerative_average_cosine.k}`], ['Spectral', `k = ${C.blindk_clip_spectral_knn.k}`, `k = ${C.blindk_siglip_spectral_knn.k}`]];
s.addTable(bt.map((r, i) => r.map((v, j) => ({ text: v, options: { bold: i === 0 || j === 0, color: INK, fill: { color: CARD }, align: j ? 'center' : 'left' } }))), { x: bkx + 0.18, y: 1.6, w: bkw - 0.36, colW: [1.35, 0.85, 0.83], rowH: 0.27, fontSize: 12, fontFace: 'Calibri', border: { type: 'none' }, margin: [0.01, 0.05, 0.01, 0.05] });
s.addText(`Spectral at k = 5 loses ${f3(C.blindk_clip_spectral_knn.AMI_k10 - C.blindk_clip_spectral_knn.AMI)} (CLIP) and ${f3(C.blindk_siglip_spectral_knn.AMI_k10 - C.blindk_siglip_spectral_knn.AMI)} (SigLIP) HS3 AMI vs k = 10.`, { x: bkx + 0.18, y: 2.75, w: bkw - 0.36, h: 0.6, fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: 'top' });
const rb = C.robust_blindk_spectral;
textCard(s, bkx, 3.6, bkw, 1.8, '5 × 80% subsets (Spectral)', `Eigengap chose k from ${Math.min(...rb.eigengap_ks)} to ${Math.max(...rb.eigengap_ks)}. Better than silhouette in ${rb.eig_better}/10, equal in ${rb.equal}, worse in ${10 - rb.eig_better - rb.equal}. Mean gap to the label-chosen k: ${rb.eig_gap_oracle.toFixed(3)} vs ${rb.sil_gap_oracle.toFixed(3)}.`);
card(s, MX, 5.55, W - 2 * MX, 1.2, TINT);
s.addText([{ text: 'Takeaway  ', options: { bold: true, color: GREEN } }, { text: 'Silhouette favours coarse partitions (k = 5–10) and works best for k-Means. Eigengap is promising for Spectral but unstable across subsets. No reliable blind k yet; HS3 has 22 classes (13 with ≥10 images).', options: { color: INK } }],
  { x: MX + 0.2, y: 5.65, w: W - 2 * MX - 0.4, h: 1.0, fontSize: 14, margin: 0, isTextBox: true, valign: 'middle' });
caption(s, 'k grid {2,3,5,8,10,15,20}; k selected before HS labels were read. Open circle = best k chosen with labels (reference only, not blind). Subsets overlap; not held-out.', H - 0.95);
s.addNotes(`(~75 s) Can we choose k blindly? On the full pilot, maximum silhouette chose k = 8 or 10 for k-Means, which is close to the best possible. For Spectral it chose k = 5 for both backbones and lost 0.07 to 0.09 AMI against k = 10: silhouette favours coarse partitions. In P2 the spectral eigengap looked promising, but on four configurations only. So in P3 we repeated it on five 80% subsets (the plot). Each row is one subset: blue dot = silhouette choice, orange square = eigengap choice, black bar = fixed k = 10, open circle = the best k if we had used labels, shown only as a reference. Eigengap jumps between k = 5 and 20, and beats silhouette in only 5 of 10 cases. Honest conclusion: no blind k rule is reliable yet. Note that HS3 has 22 classes, but only 13 with at least 10 images, so even the 'true' k is not well defined on this pilot.
Likely question: 'Why not just use 22?' That uses label information. Our protocol keeps k fixed at a predeclared 10 or chooses it blindly.`);

// =============== 7 Preprocessing
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '6 · RQ2: Does preprocessing matter?');
img(s, 'P3_PREPROCESSING_ABLATION.png', MX, 1.1, 6.9, 5.45);
const px = 7.65, pw = W - MX - px;
const ag = (e, r) => C[`pre_${e}_${r}_agglomerative_average_cosine`];
statCard(s, px, 1.1, pw, 1.55, `${ag('clip', 'l2').AMI.toFixed(2)} → ${ag('clip', 'centered_l2').AMI.toFixed(2)}`, `Centering rescues Agglomerative (CLIP; SigLIP ${ag('siglip', 'l2').AMI.toFixed(2)} → ${ag('siglip', 'centered_l2').AMI.toFixed(2)}); largest cluster ${Math.round(ag('clip', 'l2').largest_max)}% → ${Math.round(ag('clip', 'centered_l2').largest_max)}%`, 24);
const km = (e, r) => C[`pre_${e}_${r}_kmeans`].AMI_full;
statCard(s, px, 2.8, pw, 1.3, `${(km('clip', 'centered_l2') - km('clip', 'l2')).toFixed(3)} / ${(km('siglip', 'centered_l2') - km('siglip', 'l2')).toFixed(3)}`, 'Centering does not help k-Means (CLIP / SigLIP, k = 10)', 22);
const sp = (e, r) => C[`pre_${e}_${r}_spectral_knn`].AMI_full;
statCard(s, px, 4.25, pw, 1.3, `+${(sp('clip', 'centered_l2') - sp('clip', 'l2')).toFixed(3)} / +${(sp('siglip', 'centered_l2') - sp('siglip', 'l2')).toFixed(3)}`, 'Centering helps Spectral at k = 10 (not at every k)', 22);
textCard(s, px, 5.7, pw, 1.15, 'PCA', 'No gain beyond centering; PCA50 without re-normalisation hurts Spectral.', TINT, 12);
caption(s, 'Same 2,000 rows, masks (HS3 n = 1,549), k = 10 and seeds 0–4; phase-2 representations, refits reproduce earlier assignments 240/240. Global centering only; per-museum centering not run (1–14 images per museum).', H - 0.95);
s.addNotes(`(~75 s) RQ2: same rows, same masks, same k = 10, only the representation changes. Top row AMI, bottom row ARI. Strongest result: on raw L2-normalised embeddings, average-link agglomerative collapses, with about 99% of images in one cluster and AMI ${ag('clip', 'l2').AMI.toFixed(2)}. Simply subtracting the global mean fixes it: ${ag('clip', 'centered_l2').AMI.toFixed(2)} for CLIP and ${ag('siglip', 'centered_l2').AMI.toFixed(2)} for SigLIP. The reason is that raw embeddings all point in a similar direction (mean-vector norm about 0.7), so cosine distances are compressed. But centering is not a universal improvement: k-Means gets slightly worse, and Spectral gains only at k = 10. PCA adds nothing beyond the centering it includes, and PCA50 without re-normalisation hurts Spectral. So preprocessing interacts with the algorithm: it changes which method looks best. Global centering is not the same as per-museum centering; we did not run the latter because museums contribute only 1 to 14 images each in the pilot.`);

// =============== 8 Limitations
pres.addSection({ title: 'Outlook' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Outlook' });
title(s, '7 · Current problems and research limitations');
const lim = [
  ['Pilot is not held out', `All choices were made after seeing pilot scores. The full-pilot ${f3(C.siglip_spectral_knn_k10.AMI)} sits near the top of its subset range (${f3(C.robust_siglip.spectral_AMI_min)}–${f3(C.robust_siglip.spectral_AMI_max)}).`],
  ['Full dataset missing locally', 'The 27,348-object benchmark (13 HS3 classes) and its held-out split are absent, so the plan’s 0.53 / 0.83 / 0.68 cannot be compared.'],
  ['DINOv2 not yet evaluated', 'Weights not downloaded (~1.6 GB for S/B/L). Code path checked; DINOv2-B ≈ 6 min on the pilot (CPU).'],
  ['Imbalance & dominant clusters', 'Aerophones are 52% of HS1 labels. Spectral keeps a 34–39% catch-all cluster, so AMI and ARI disagree.'],
  ['Museums & near-duplicates', '1–14 images per museum; 191 unconfirmed near-duplicate pairs; subsets overlap, so no confidence intervals.'],
  ['Sparse audio pairs (RQ4)', 'Only 868 objects with their own recording (Week 4 count, not recomputed), skewed to a few museums.'],
];
const lw = (W - 2 * MX - 0.3) / 3, lh = 2.25;
lim.forEach(([h, b], i) => { const col = i % 3, row = Math.floor(i / 3); textCard(s, MX + col * (lw + 0.15), 1.2 + row * (lh + 0.25), lw, lh, h, b, i === 0 ? TINT : CARD, 14); });
card(s, MX, 6.05, W - 2 * MX, 0.85, TINT);
s.addText([{ text: 'Why it matters  ', options: { bold: true, color: GREEN } }, { text: 'Every number today is a development estimate that is likely optimistic, on 22 imbalanced classes, without museum-level resampling.', options: { color: INK } }],
  { x: MX + 0.2, y: 6.1, w: W - 2 * MX - 0.4, h: 0.75, fontSize: 14, margin: 0, isTextBox: true, valign: 'middle' });
s.addNotes(`(~1 min) Six open problems. One: the pilot was explored in earlier phases, so every number is a development estimate and probably optimistic; Spectral's 0.544 is near the top of its subset range. Two: the large benchmark and its held-out split are not available locally, which is why we cannot test the 0.68 target today. Three: DINOv2 is not evaluated; we only need authorization to download about 1.6 GB of weights. Four: class imbalance (aerophones are half of HS1) and dominant clusters make AMI and ARI disagree. Five: museums contribute only 1 to 14 images each, and we flagged 191 possible near-duplicate pairs that need human confirmation; overlapping subsets cannot give confidence intervals. Six: for the multimodal RQ4, only 868 objects have their own recording.`);

// =============== 9 Roadmap
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Outlook' });
title(s, '8 · Next steps and research roadmap');
const ms = [
  ['W8 · 30.11.', ['Extend RQ1 comparisons', 'Complete RQ2 ablations', 'Begin RQ3: DINOv2 S/B/L, layers −1/−2/−4', 'Extend stability analyses']],
  ['W10 · 14.12.', ['Complete main dev. experiments', 'Select and freeze the pipeline', 'Set up multimodal experiments']],
  ['W14 · 11.01.', ['Multimodal (VLM text, image + audio) experiments', 'One-shot held-out run']],
  ['W16 · 25.01.', ['Final presentation with demo']],
  ['W17 · 01.02.', ['Report and code submission']],
];
const mw = (W - 2 * MX - 0.6) / 5;
s.addShape(pres.shapes.LINE, { x: MX + 0.2, y: 1.5, w: W - 2 * MX - 0.4, h: 0, line: { color: LINE, width: 2 } });
ms.forEach(([h, b], i) => {
  const x = MX + i * (mw + 0.15);
  s.addShape(pres.shapes.OVAL, { x: x + 0.05, y: 1.36, w: 0.28, h: 0.28, fill: { color: i === 0 ? GREEN : WHITE }, line: { color: GREEN, width: 2 } });
  card(s, x, 1.85, mw, 2.85, i === 0 ? TINT : CARD);
  s.addText(h, { x: x + 0.15, y: 1.95, w: mw - 0.3, h: 0.35, fontSize: 15, bold: true, color: GREEN, margin: 0, isTextBox: true });
  s.addText(b.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < b.length - 1 } })), { x: x + 0.12, y: 2.4, w: mw - 0.24, h: 2.2, fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 4 });
});
textCard(s, MX, 4.95, W - 2 * MX, 1.85, 'Adjusted priorities for Week 8 (from today’s findings)',
  [{ text: 'Authorize DINOv2 weights, then run S/B/L and layers on the same pilot rows; add the linear-probe reference', options: { bullet: true, breakLine: true } },
   { text: 'Restore the 27,348-object benchmark and its held-out split before freezing anything', options: { bullet: true, breakLine: true } },
   { text: 'Always report AMI and ARI together; blind k (silhouette vs eigengap) stays an open question', options: { bullet: true, breakLine: true } },
   { text: 'Human review of the 180-image sample and near-duplicates; museum-level resampling', options: { bullet: true } }], CARD, 13);
caption(s, 'Schedule from the Week 4 project plan. Details: reports/week6/P4_NEXT_STEPS.md, P3_RQ3_READINESS.md.', H - 0.95);
s.addNotes(`(~1 min) The schedule is unchanged from Week 4. For Week 8 we extend RQ1, finish the RQ2 ablations, start RQ3 model scaling and extend the stability analyses. What changes in priority is the order: first, the DINOv2 weights, which is all RQ3 needs, and the linear-probe reference. Second, the large benchmark and its held-out split, so the freeze in Week 10 is not based on the explored pilot. Third, we will always report AMI and ARI together, given today's metric disagreement. Week 10: freeze the pipeline; Week 14: multimodal and the one-shot held-out run; Week 16 the final presentation with demo, Week 17 the report and code.`);

// =============== 10 Thanks
s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Outlook' });
s.addText('Thank you for listening', { x: 0.9, y: 2.4, w: 9.9, h: 0.8, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addText('Questions?', { x: 0.9, y: 3.25, w: 9.9, h: 0.5, fontSize: 20, color: '3DBB5E', margin: 0, isTextBox: true });
s.addText([{ text: 'Tim Henninger · Christian Krengel · Aiman Alkafahje', options: { breakLine: true } }, { text: 'Clustering meets Vision and Language · Dr. Udo Schlegel · LMU Munich' }],
  { x: 0.9, y: 6.4, w: 9.9, h: 0.7, fontSize: 12, color: 'BDBDBD', margin: 0, isTextBox: true, paraSpaceAfter: 4 });
s.addNotes('Backup slides follow: AMI vs ARI heatmaps, subset robustness, cluster-size diagnostics, backbone comparison.');

// =============== Backups
pres.addSection({ title: 'Backup' });
const backups = [
  ['Backup · Why AMI and ARI disagree (cluster × class composition)', 'P3_AMI_VS_ARI.png', 'Each row is a cluster (sorted by labelled size), each column one of the 12 most frequent HS3 classes; the cell is the class share within the cluster. Spectral: one large mixed cluster (C1) plus many pure clusters. Agglomerative: more medium mixed clusters, smaller catch-all.'],
  ['Backup · Spectral robustness on five 80% subsets', 'P3_SPECTRAL_ROBUSTNESS.png', `Five 1,600-image subsets (phase-2 seeds 2026–2030), centering refit inside each subset. Spectral > k-Means AMI in ${C.robust_clip.spec_gt_km_AMI}/5 CLIP and ${C.robust_siglip.spec_gt_km_AMI}/5 SigLIP subsets; Spectral > Agglomerative AMI in ${C.robust_clip.spec_gt_agg_AMI}/5 CLIP and ${C.robust_siglip.spec_gt_agg_AMI}/5 SigLIP; Agglomerative > Spectral ARI in ${C.robust_clip.agg_gt_spec_ARI + C.robust_siglip.agg_gt_spec_ARI}/10.`],
  ['Backup · Cluster-size diagnostics', 'P3_CLUSTER_DIAGNOSTICS.png', 'Largest cluster share at k = 10 per representation (worst seed) and the cluster-size profile for SigLIP centered + L2. Raw-L2 Agglomerative collapses; Spectral keeps one dominant cluster under every representation.'],
  ['Backup · Backbone comparison at k = 10', 'P2_BACKBONE_COMPARISON.png', 'CLIP vs SigLIP on identical rows and masks. This compares complete encoder bundles (architecture, patch size, training data, loss, transform), so the gain cannot be attributed to the pretraining loss alone.'],
];
backups.forEach(([t, f, notes]) => {
  const b = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Backup' });
  b.addText(t, { x: MX, y: 0.35, w: W - 2 * MX, h: 0.55, fontSize: 22, bold: true, color: INK, margin: 0, isTextBox: true, valign: 'middle' });
  img(b, f, MX, 1.05, W - 2 * MX, 6.6, 'center');
  b.addNotes(notes);
});

(async () => {
  const out = path.join(OUTDIR, 'Week6_Clustering_StatusUpdate.pptx');
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log('wrote', out);
})();
