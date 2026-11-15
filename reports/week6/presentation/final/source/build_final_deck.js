// Final Week 6 deck (full MIMO collection + 12,000-object development benchmark). Same theme/helpers as
// reports/week6/source/build_deck.js. Every number is read from ../final_claims.json (recomputed from assignments).
// Usage: NODE_PATH=<node_modules> node build_final_deck.js <repo_root> <apply_theme.js>
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');
const ROOT = process.argv[2];
const { applyTheme } = require(process.argv[3]);
const FIN = path.join(ROOT, 'reports/week6/presentation/final');
const C = JSON.parse(fs.readFileSync(path.join(FIN, 'final_claims.json')));
const D = C.dataset, S = C.scores;
const sc = (b, rep, m, k = 'k10') => S[`${b}|${rep}|${m}|${k}`];
const f3 = (x) => Number(x).toFixed(3);
const n = (x) => Number(x).toLocaleString('en');

const GREEN = '1F8A3B', DARK = '2B2B2B', DARK2 = '3A3A3A', INK = '1A1A1A', MUTED = '5F5F5F', CARD = 'F1F4F1', LINE = 'D9DED9', WHITE = 'FFFFFF', TINT = 'E3F1E6';
const THEME = { name: 'Week6 Green', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: INK, lt1: WHITE, dk2: DARK, lt2: CARD, accent1: GREEN, accent2: '2A78D6', accent3: 'EB6834', accent4: 'EDA100', accent5: '6250D6', accent6: '5F5F5F', hlink: GREEN, folHlink: '5F5F5F' } };
const FOOTER = 'Clustering meets Vision & Language · Topic 1 · Week 6 (16.11.2026)';
const W = 11.69, H = 8.27, MX = 0.55;
const DEVCAP = `Development benchmark: ${n(D.dev)} objects, ${D.hs3_classes} HS3 classes`;

const pres = new pptxgen();
pres.defineLayout({ name: 'A4_LANDSCAPE', width: W, height: H });
pres.layout = 'A4_LANDSCAPE';
pres.theme = { headFontFace: 'Calibri', bodyFontFace: 'Calibri' };
pres.title = 'Week 6 status update (final): Visual Form vs. Acoustic Function in Instrument Clustering';
pres.author = 'Tim Henninger, Christian Krengel, Aiman Alkafahje';
pres.defineSlideMaster({ title: 'CONTENT', background: { color: WHITE },
  objects: [{ text: { text: FOOTER, options: { x: MX, y: H - 0.42, w: 8, h: 0.25, fontSize: 9, color: MUTED, margin: 0 } } }],
  slideNumber: { x: W - MX - 0.5, y: H - 0.42, w: 0.5, h: 0.25, fontSize: 9, color: MUTED, align: 'right' }, placeholders: [] });
pres.defineSlideMaster({ title: 'DARK', background: { color: DARK }, objects: [] });
const NOTES = require(path.join(__dirname, 'notes_en.js'))(C);
const _add = pres.addSlide.bind(pres); let _ni = 0;
pres.addSlide = (o) => { const sl = _add(o); const t = NOTES[_ni++]; if (t) sl.addNotes(t); return sl; };

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function img(slide, file, x, y, maxW, maxH, align = 'left') {
  const p = path.join(ROOT, file); const s = pngSize(p); const r = s.w / s.h;
  let w = maxW, h = w / r; if (h > maxH) { h = maxH; w = h * r; }
  const xx = align === 'center' ? x + (maxW - w) / 2 : x;
  slide.addImage({ path: p, x: xx, y, w, h, altText: path.basename(file, '.png').replace(/_/g, ' ') });
  return { x: xx, y, w, h };
}
function title(s, t) { s.addText(t, { x: MX, y: 0.35, w: W - 2 * MX, h: 0.6, fontSize: 28, bold: true, color: INK, margin: 0, isTextBox: true, valign: 'middle' }); }
function caption(s, t, y = H - 0.95) { s.addText(t, { x: MX, y, w: W - 2 * MX, h: 0.4, fontSize: 10, color: MUTED, margin: 0, isTextBox: true, valign: 'top' }); }
function card(s, x, y, w, h, fill = CARD) { s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.06 }); }
function statCard(s, x, y, w, h, big, text, bigSize = 26) {
  card(s, x, y, w, h);
  s.addText(big, { x: x + 0.18, y: y + 0.1, w: w - 0.36, h: 0.55, fontSize: bigSize, bold: true, color: GREEN, margin: 0, isTextBox: true, valign: 'middle' });
  s.addText(text, { x: x + 0.18, y: y + 0.68, w: w - 0.36, h: h - 0.78, fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: 'top' });
}
function textCard(s, x, y, w, h, head, body, fill = CARD, bodySize = 12) {
  card(s, x, y, w, h, fill);
  s.addText(head, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: 0.32, fontSize: 14, bold: true, color: GREEN, margin: 0, isTextBox: true, valign: 'top' });
  s.addText(body, { x: x + 0.18, y: y + 0.5, w: w - 0.36, h: h - 0.6, fontSize: bodySize, color: INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 3 });
}
function banner(s, y, head, text, h = 0.8) {
  card(s, MX, y, W - 2 * MX, h, TINT);
  s.addText([{ text: head + '  ', options: { bold: true, color: GREEN } }, { text, options: { color: INK } }],
    { x: MX + 0.2, y: y + 0.05, w: W - 2 * MX - 0.4, h: h - 0.1, fontSize: 14, margin: 0, isTextBox: true, valign: 'middle' });
}
const bullets = (arr) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1 } }));

// ---------------- 1 Title
pres.addSection({ title: 'Overview' });
let s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Overview' });
s.addText('Topic 1 · Week 6 Update · 16.11.2026', { x: 0.9, y: 1.15, w: 9, h: 0.35, fontSize: 14, bold: true, color: '3DBB5E', margin: 0, isTextBox: true });
s.addText('Visual Form vs. Acoustic Function in Instrument Clustering', { x: 0.9, y: 1.6, w: 9.9, h: 1.3, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true, valign: 'top' });
s.addText('Week 6 Status Update · Topic 1: Benchmarking Clustering Methods on VLM Embeddings', { x: 0.9, y: 2.95, w: 9.9, h: 0.35, fontSize: 15, color: 'D0D0D0', margin: 0, isTextBox: true });
card(s, 0.9, 3.6, 9.9, 1.75, DARK2);
s.addText([{ text: 'Main research question', options: { fontSize: 12, color: 'BDBDBD', breakLine: true } },
  { text: 'Which unsupervised clustering method, backbone and preprocessing best recover the expert Hornbostel–Sachs classification of museum instrument photographs from frozen Vision/VLM embeddings?', options: { fontSize: 15, color: WHITE } }],
  { x: 1.15, y: 3.75, w: 9.4, h: 1.5, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 4 });
s.addText(`Today: the reconstructed MIMO collection (${n(D.cleaned)} photographs) and the first results on a ${n(D.dev)}-object development benchmark.`, { x: 0.9, y: 5.55, w: 9.9, h: 0.6, fontSize: 13, color: '3DBB5E', margin: 0, isTextBox: true });
s.addText([{ text: 'Tim Henninger · Christian Krengel · Aiman Alkafahje', options: { breakLine: true } }, { text: 'Clustering meets Vision and Language · Dr. Udo Schlegel · LMU Munich' }],
  { x: 0.9, y: 6.4, w: 9.9, h: 0.7, fontSize: 12, color: 'BDBDBD', margin: 0, isTextBox: true, paraSpaceAfter: 4 });

// ---------------- 2 Progress
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
title(s, '1 · What we accomplished since Week 4');
const items = [
  ['Full MIMO collection reconstructed', `${n(D.records)} records → ${n(D.cleaned)} cleaned instrument photographs`],
  ['Selected benchmark rebuilt', `${n(D.benchmark)} eligible objects; ${n(D.dev)}-object development benchmark fixed`],
  ['DINOv2 installed', 'Official S / B / L weights; all three tested on the 2,000-image pilot'],
  ['Three backbones compared', `CLIP, SigLIP and DINOv2-B on the same ${n(D.dev)} development objects`],
  ['Five clustering methods', 'k-Means · Agglomerative · Spectral · GMM · HDBSCAN'],
  ['Results independently validated', 'Every score recomputed from the saved cluster assignments'],
];
items.forEach(([h, b], i) => {
  const y = 1.25 + i * 0.86;
  s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.05, w: 0.36, h: 0.36, fill: { color: GREEN }, line: { color: GREEN } });
  s.addText('✓', { x: MX, y: y + 0.05, w: 0.36, h: 0.36, fontSize: 14, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText([{ text: h, options: { bold: true, fontSize: 15, color: INK, breakLine: true } }, { text: b, options: { fontSize: 13, color: MUTED } }], { x: MX + 0.55, y, w: 5.7, h: 0.75, margin: 0, isTextBox: true, valign: 'top' });
});
const sx = 7.05, sw = 2.0, sh = 1.6;
statCard(s, sx, 1.25, sw, sh, n(D.cleaned), 'cleaned instrument photographs (full MIMO collection)', 24);
statCard(s, sx + sw + 0.15, 1.25, sw, sh, n(D.benchmark), 'eligible objects in the selected benchmark', 24);
statCard(s, sx, 1.25 + sh + 0.15, sw, sh, n(D.dev), 'development objects used for all comparisons', 24);
statCard(s, sx + sw + 0.15, 1.25 + sh + 0.15, sw, sh, n(D.heldout), 'held-out objects, reserved and untouched', 24);
textCard(s, sx, 4.75, 2 * sw + 0.15, 1.45, 'Scope', 'DINOv2-S and -L were tested only on the pilot; DINOv2-B is the DINOv2 model evaluated on the development benchmark. No held-out results yet.', TINT, 12);
caption(s, 'Sources: reports/full_dataset/FULL_DATASET_REPORT.md, reports/dinov2/DINOV2_BENCHMARK.md, reports/week6/presentation/final/Week6_Final_Validation.md.');

// ---------------- 3 Dataset
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
title(s, '2 · Dataset and evaluation protocol');
const fun = [[`${n(D.records)}`, 'original MIMO records (Europeana)', 6.3], [`${n(D.valid_images)}`, 'valid downloaded images', 6.0], [`${n(D.cleaned)}`, 'cleaned photographs: full MIMO collection', 5.5], [`${n(D.benchmark)}`, 'selected benchmark: HS3 + family + name, classes ≥ 30', 4.9]];
fun.forEach(([num, lab, w], i) => {
  const y = 1.2 + i * 0.82, x = MX + (6.3 - w) / 2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.66, fill: { color: i === 3 ? GREEN : CARD }, line: { color: i === 3 ? GREEN : LINE }, rectRadius: 0.05 });
  s.addText([{ text: num + '  ', options: { bold: true, fontSize: 18, color: i === 3 ? WHITE : GREEN } }, { text: lab, options: { fontSize: 12.5, color: i === 3 ? WHITE : INK } }],
    { x: x + 0.15, y, w: w - 0.3, h: 0.66, margin: 0, isTextBox: true, valign: 'middle', align: 'center' });
});
const parts = [[n(D.dev), 'development', 'all experiments today', TINT, GREEN], [n(D.heldout), 'held-out', 'reserved, untouched', CARD, INK], [n(D.pilot_overlap_excluded), 'excluded', 'already seen in pilot', CARD, MUTED]];
const pw = (6.3 - 0.3) / 3;
parts.forEach(([num, lab, sub, fill, col], i) => {
  const x = MX + i * (pw + 0.15), y = 4.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: pw, h: 1.05, fill: { color: fill }, line: { color: i === 0 ? GREEN : LINE, width: i === 0 ? 1.5 : 0.75 }, rectRadius: 0.05 });
  s.addText([{ text: num, options: { bold: true, fontSize: 18, color: col, breakLine: true } }, { text: lab, options: { bold: true, fontSize: 12, color: INK, breakLine: true } }, { text: sub, options: { fontSize: 10.5, color: MUTED } }],
    { x, y: y + 0.05, w: pw, h: 0.95, margin: 0, isTextBox: true, align: 'center', valign: 'middle' });
});
s.addText('↓', { x: MX + 2.9, y: 4.36, w: 0.5, h: 0.35, fontSize: 18, color: MUTED, align: 'center', margin: 0, isTextBox: true });
img(s, 'reports/week6/presentation/final/figures/FINAL_DEV_HS3_CLASSES.png', 7.15, 1.15, 4.0, 3.75);
const chips = [[`${D.hs3_classes}`, 'HS3 classes'], [`${D.families}`, 'instrument families'], [`${D.name_classes}`, 'name classes']];
chips.forEach(([num, lab], i) => statCard(s, 7.15 + i * 1.37, 5.05, 1.27, 1.15, num, lab, 22));
caption(s, `Split: development = the teammate's 12,000-row rule (rng 0); the other benchmark objects are held out, minus ${D.pilot_overlap_excluded} already explored in the pilot. Exact pixel duplicates never cross partitions. Audio-object exclusion from held-out is still pending.`);

// ---------------- 4 Pipeline
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
title(s, '3 · Our working benchmark pipeline');
s.addText('Blind pipeline: HS labels never form clusters or choose k; they are used only to score.', { x: MX, y: 1.05, w: 10, h: 0.3, fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true });
const boxes = [
  ['1  MIMO images', [`${n(D.dev)} development objects`, 'white-square padding', 'hash-verified files'], 'same rows for all'],
  ['2  Frozen encoders', ['CLIP ViT-B/32 · 512-d', 'SigLIP ViT-B/16 · 768-d', 'DINOv2-B/14 · 768-d · 224 px'], 'frozen weights'],
  ['3  Preprocessing', ['raw L2', 'global centering + L2', 'PCA50 / PCA100 + L2'], 'no labels'],
  ['4  Clustering', ['k-Means · Agglomerative', 'Spectral · GMM · HDBSCAN', 'fixed k = 10'], 'no labels'],
  ['5  Evaluation', ['AMI · ARI · NMI', `vs ${D.hs3_classes} HS3 classes`, 'cluster sizes, noise'], 'labels used only here'],
];
const bw = 1.95, bg = 0.18, by = 1.55, bh = 2.25;
boxes.forEach(([h, lines, tag], i) => {
  const x = MX + i * (bw + bg);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, fill: { color: i === 4 ? 'F4F4F4' : CARD }, line: { color: i === 4 ? '444444' : GREEN, width: 1.5 }, rectRadius: 0.06 });
  s.addText(h, { x: x + 0.12, y: by + 0.12, w: bw - 0.24, h: 0.32, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(lines.map((l, j) => ({ text: l, options: { breakLine: j < lines.length - 1 } })), { x: x + 0.12, y: by + 0.55, w: bw - 0.24, h: 1.1, fontSize: 11.5, color: INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 2 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.15, y: by + bh - 0.45, w: bw - 0.3, h: 0.3, fill: { color: i === 4 ? '333333' : GREEN }, line: { color: i === 4 ? '333333' : GREEN }, rectRadius: 0.05 });
  s.addText(tag, { x: x + 0.15, y: by + bh - 0.45, w: bw - 0.3, h: 0.3, fontSize: 10.5, bold: true, color: WHITE, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  if (i < 4) s.addText('→', { x: x + bw - 0.02, y: by + bh / 2 - 0.2, w: bg + 0.04, h: 0.4, fontSize: 16, color: MUTED, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
});
textCard(s, MX, 4.1, 5.2, 1.95, 'Controlled comparison', bullets([`All backbones, methods and preprocessings use the same ${n(D.dev)} objects and labels`, 'Stochastic methods: 5 seeds, mean ± SD (never best seed)', 'Global centering and PCA are fitted on the embeddings only']), CARD, 12.5);
textCard(s, MX + 5.4, 4.1, 5.19, 1.95, 'What these numbers are', bullets(['Development results, chosen and compared on the same objects', `Not measured on all ${n(D.cleaned)} photographs`, `Held-out set (${n(D.heldout)}) reserved for one final run of the frozen pipeline`]), TINT, 12.5);
caption(s, 'Protocol: reports/dinov2/DINOV2_PROTOCOL.json (declared before fitting). DINOv2: timm/vit_base_patch14_dinov2.lvd142m, CLS token after final LayerNorm.');

// ---------------- 5 RQ1 backbones
pres.addSection({ title: 'Results' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '4 · RQ1: Which backbone performs best?');
img(s, 'reports/week6/presentation/final/figures/FINAL_RQ1_BACKBONES.png', MX, 1.1, 7.4, 4.9);
const rx = 8.15, rw = W - MX - rx;
const da = sc('dinov2', 'centered_l2', 'agglo'), dk = sc('dinov2', 'centered_l2', 'kmeans');
statCard(s, rx, 1.1, rw, 1.35, f3(da.AMI), `DINOv2-B + Agglomerative (SigLIP ${f3(sc('siglip', 'centered_l2', 'agglo').AMI)}, CLIP ${f3(sc('clip', 'centered_l2', 'agglo').AMI)})`);
statCard(s, rx, 2.6, rw, 1.35, f3(dk.AMI), `DINOv2-B + k-Means (SigLIP ${f3(sc('siglip', 'centered_l2', 'kmeans').AMI)}, CLIP ${f3(sc('clip', 'centered_l2', 'kmeans').AMI)})`);
textCard(s, rx, 4.1, rw, 1.9, 'Not for every method', `With Spectral, SigLIP leads (${f3(sc('siglip', 'centered_l2', 'spectral').AMI)}) and DINOv2-B drops to ${f3(sc('dinov2', 'centered_l2', 'spectral').AMI)}. k-Means ARI: SigLIP ${f3(sc('siglip', 'centered_l2', 'kmeans').ARI)} > DINOv2-B ${f3(dk.ARI)}.`, TINT, 12);
banner(s, 6.15, 'Finding 1', 'DINOv2-B outperforms CLIP and SigLIP with k-Means and agglomerative clustering (HS3 AMI).', 0.7);
caption(s, `${DEVCAP} · centered + L2 · k = 10 · DINOv2-B at 224 px. Development results, not held-out.`);

// ---------------- 6 RQ1 methods
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '5 · RQ1: Does the clustering method matter?');
img(s, 'reports/week6/presentation/final/figures/FINAL_RQ1_AMI_VS_ARI.png', MX, 1.1, 6.95, 5.0);
const ss = sc('siglip', 'centered_l2', 'spectral'), ds = sc('dinov2', 'centered_l2', 'spectral');
const r6 = 7.7, w6 = W - MX - r6;
textCard(s, r6, 1.1, w6, 1.4, 'Two candidates, same AMI', `DINOv2-B + Agglomerative ${f3(da.AMI)} / ARI ${f3(da.ARI)}\nSigLIP + Spectral ${f3(ss.AMI)} / ARI ${f3(ss.ARI)}`, CARD, 12.5);
textCard(s, r6, 2.65, w6, 1.55, 'Why AMI alone can mislead', `ARI counts pairs of images, so one oversized mixed cluster costs a lot. DINOv2-B + Spectral puts ${ds.largest_pct_max}% of objects in one cluster (ARI ${f3(ds.ARI)}).`, CARD, 12);
textCard(s, r6, 4.35, w6, 1.75, 'Balanced candidate', `DINOv2-B + Agglomerative has the highest ARI and a largest cluster of only ${da.largest_pct_max}%. It is a strong candidate under k = 10, centered + L2, not a universal winner.`, TINT, 12);
banner(s, 6.25, 'Finding 2', 'The best clustering algorithm depends on the backbone and on how clustering quality is measured.', 0.65);
caption(s, `${DEVCAP} · centered + L2 · k = 10 (HDBSCAN chooses its own k). Development results, not held-out.`);

// ---------------- 7 RQ2
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
title(s, '6 · RQ2: Does preprocessing matter?');
img(s, 'reports/week6/presentation/final/figures/FINAL_RQ2_PREPROCESSING.png', MX, 1.05, W - 2 * MX, 3.85, 'center');
const p = (b, r) => sc(b, r, 'agglo');
const w7 = (W - 2 * MX - 0.3) / 3;
statCard(s, MX, 5.0, w7, 1.15, `${f3(p('clip', 'l2').AMI)} → ${f3(p('clip', 'centered_l2').AMI)}`, `CLIP: centering ends the collapse (${p('clip', 'l2').largest_pct_max}% in one cluster)`, 20);
statCard(s, MX + w7 + 0.15, 5.0, w7, 1.15, `${f3(p('siglip', 'l2').AMI)} → ${f3(p('siglip', 'centered_l2').AMI)}`, `SigLIP: same collapse (${p('siglip', 'l2').largest_pct_max}%) and same rescue`, 20);
statCard(s, MX + 2 * (w7 + 0.15), 5.0, w7, 1.15, `${f3(p('dinov2', 'l2').AMI)} → ${f3(p('dinov2', 'centered_l2').AMI)}`, `DINOv2-B: more robust raw (${p('dinov2', 'l2').largest_pct_max}%); PCA100 + L2: ${f3(p('dinov2', 'pca100_l2').AMI)}`, 20);
banner(s, 6.3, 'Finding 3', 'Global centering dramatically improves cosine agglomerative clustering, but the exact impact depends on the embedding model.', 0.75);
caption(s, `${DEVCAP} · agglomerative (average linkage, cosine) · k = 10 · HS3 AMI. Development results, not held-out.`);

// ---------------- 8 Limitations
pres.addSection({ title: 'Outlook' });
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Outlook' });
title(s, '7 · Current problems and research limitations');
const lim = [
  ['Held-out evaluation pending', `The ${n(D.dev)}-object development benchmark is evaluated; the ${n(D.heldout)} held-out objects remain untouched.`],
  ['Not all photos clustered yet', `${n(D.cleaned)} cleaned photographs exist, but clustering so far uses the ${n(D.dev)}-object development subset.`],
  ['Model scaling not yet complete', 'DINOv2-B was evaluated at scale (224 px). S/B/L and layer depth were only explored on the 2,000-image pilot.'],
  ['Museum confounding', `The five largest contributing museums hold ${Math.round(D.top5_museum_share * 100)}% of benchmark objects; museum effects are untested.`],
  ['Potential near-duplicates', 'Exact duplicates are removed, but no near-duplicate screening has been run on the full collection yet.'],
  ['Held-out split preparation', 'Objects with audio recordings still have to be excluded from the held-out set (original protocol). Not done yet.'],
];
const lw = (W - 2 * MX - 0.3) / 3, lh = 2.05;
lim.forEach(([h, b], i) => { const col = i % 3, row = Math.floor(i / 3); textCard(s, MX + col * (lw + 0.15), 1.2 + row * (lh + 0.25), lw, lh, h, b, i === 0 ? TINT : CARD, 13.5); });
banner(s, 6.0, 'Why it matters', 'Today\'s scores are development estimates on one split; the frozen pipeline still has to prove itself on held-out data.', 0.8);

// ---------------- 9 Next steps
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Outlook' });
title(s, '8 · Next steps and research roadmap');
const ms = [
  ['W8 · 30.11.', ['Extend DINOv2 on the development benchmark', 'Model scaling and intermediate layers', 'More preprocessing variants', 'Museum effects, near-duplicates', 'Label-free choice of k']],
  ['W10 · 14.12.', ['Complete main development experiments', 'Select and freeze the pipeline', 'Finish held-out split preparation', 'Prepare multimodal evaluation']],
  ['W14 · 11.01.', ['Remaining multimodal experiments', 'One-shot held-out evaluation with the frozen pipeline']],
  ['W16 · 25.01.', ['Final presentation with demo']],
  ['W17 · 01.02.', ['Final report and code submission']],
];
const mw = (W - 2 * MX - 0.6) / 5;
s.addShape(pres.shapes.LINE, { x: MX + 0.2, y: 1.5, w: W - 2 * MX - 0.4, h: 0, line: { color: LINE, width: 2 } });
ms.forEach(([h, b], i) => {
  const x = MX + i * (mw + 0.15);
  s.addShape(pres.shapes.OVAL, { x: x + 0.05, y: 1.36, w: 0.28, h: 0.28, fill: { color: i === 0 ? GREEN : WHITE }, line: { color: GREEN, width: 2 } });
  card(s, x, 1.85, mw, 4.2, i === 0 ? TINT : CARD);
  s.addText(h, { x: x + 0.15, y: 1.95, w: mw - 0.3, h: 0.35, fontSize: 15, bold: true, color: GREEN, margin: 0, isTextBox: true });
  s.addText(bullets(b), { x: x + 0.12, y: 2.4, w: mw - 0.24, h: 3.5, fontSize: 12.5, color: INK, margin: 0, isTextBox: true, valign: 'top', paraSpaceAfter: 5 });
});
banner(s, 6.25, 'Priority', 'Validate the development findings at scale before freezing: DINOv2 sizes, museum effects, blind k.', 0.65);

// ---------------- 10 Closing
s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Outlook' });
s.addText('Thank you for listening', { x: 0.9, y: 2.4, w: 9.9, h: 0.8, fontSize: 34, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addText('Questions?', { x: 0.9, y: 3.25, w: 9.9, h: 0.5, fontSize: 20, color: '3DBB5E', margin: 0, isTextBox: true });
s.addText([{ text: 'Tim Henninger · Christian Krengel · Aiman Alkafahje', options: { breakLine: true } }, { text: 'Clustering meets Vision and Language · Dr. Udo Schlegel · LMU Munich' }],
  { x: 0.9, y: 6.4, w: 9.9, h: 0.7, fontSize: 12, color: 'BDBDBD', margin: 0, isTextBox: true, paraSpaceAfter: 4 });

// ---------------- Backups
pres.addSection({ title: 'Backup' });
function backup(t, file, cap) {
  const b = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Backup' });
  b.addText(t, { x: MX, y: 0.35, w: W - 2 * MX, h: 0.55, fontSize: 22, bold: true, color: INK, margin: 0, isTextBox: true, valign: 'middle' });
  img(b, file, MX, 1.05, W - 2 * MX, 6.1, 'center');
  caption(b, cap);
  return b;
}
backup('Backup · All five methods, AMI and ARI (development benchmark)', 'reports/dinov2/figures/DINOV2_DEV_PARTITION.png', `${DEVCAP} · centered + L2 · k = 10 · mean of 5 seeds where stochastic.`);
// k=13 label-informed reference table
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Backup' });
s.addText('Backup · Label-informed reference: k = 13 (number of HS3 classes)', { x: MX, y: 0.35, w: W - 2 * MX, h: 0.55, fontSize: 22, bold: true, color: INK, margin: 0, isTextBox: true, valign: 'middle' });
const rows = [['HS3 AMI · centered + L2', 'CLIP k=10 / k=13', 'SigLIP k=10 / k=13', 'DINOv2-B k=10 / k=13']];
[['kmeans', 'k-Means'], ['agglo', 'Agglomerative'], ['spectral', 'Spectral'], ['gmm', 'GMM (diagonal)']].forEach(([m, lab]) => {
  rows.push([lab, ...['clip', 'siglip', 'dinov2'].map((b) => `${f3(sc(b, 'centered_l2', m).AMI)} / ${f3(sc(b, 'centered_l2', m, 'k13').AMI)}`)]);
});
s.addTable(rows.map((r, i) => r.map((v, j) => ({ text: v, options: { bold: i === 0 || j === 0, color: INK, fill: { color: i === 0 ? CARD : WHITE }, align: j ? 'center' : 'left' } }))),
  { x: MX, y: 1.3, w: W - 2 * MX, colW: [3.0, 2.53, 2.53, 2.53], rowH: 0.45, fontSize: 14, fontFace: 'Calibri', border: { type: 'solid', pt: 0.5, color: LINE } });
textCard(s, MX, 4.0, W - 2 * MX, 1.6, 'Not a blind-k result', `k = 13 uses the known number of benchmark HS3 classes, so it is a label-informed reference only. The headline comparison stays at the predeclared k = 10. DINOv2-B + Agglomerative reaches ${f3(sc('dinov2', 'centered_l2', 'agglo', 'k13').AMI)} at k = 13.`, TINT, 13);
caption(s, `${DEVCAP}. Source: reports/dinov2/dev/summary.csv.`);
backup('Backup · Museum distribution of the collection and benchmark', 'reports/full_dataset/figures/FULL_DATASET_MUSEUMS.png', `${D.museums_clean} museums in the cleaned collection, ${D.museums_benchmark} in the benchmark; the five largest hold ${Math.round(D.top5_museum_share * 100)}% of benchmark objects.`);
backup('Backup · Reconstruction vs research-plan counts', 'reports/full_dataset/figures/FULL_DATASET_FUNNEL.png', 'All counts recomputed from the new download; differences of 1–3 objects come from 5 unreachable images and live Wikidata changes.');
backup('Backup · Exploratory pilot only: DINOv2 size and layer depth', 'reports/dinov2/figures/DINOV2_SCALING.png', 'EXPLORATORY: 2,000-image pilot (22 HS3 classes), not the development benchmark; not comparable with the development scores. Full-scale RQ3 validation is future work.');

(async () => {
  const out = path.join(FIN, 'Week6_Final_FullMIMO_StatusUpdate.pptx');
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log('wrote', out);
})();
