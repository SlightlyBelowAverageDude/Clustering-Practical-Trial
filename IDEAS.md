# Idea search, state as of 19.10.2026

Course: "Clustering meets Vision and Language", LMU, WiSe 2026/27. Pitch on 19.10.2026 (3 minutes, 2 slides). Topics 1 to 3 were empty on 12.10.2026, 4 to 6 full.

Decision on 14.10.2026: move forward with the top 3 (instruments, museum objects, Ghost Borders).

Every spike: a sample of real images, frozen encoders (CLIP ViT-B/32, DINOv2-S, sometimes CLIP ViT-L/14), k-means, agglomerative, GMM and HDBSCAN, scored with AMI against the ground-truth label, plus a linear probe as the ceiling for what the embedding carries. 1 seed, no tuning. Numbers rank ideas; they are not results. Scripts in `spikes/`, outputs in `runs/`.

## Ranking

| Rank | Idea | Spike verdict | Literature verdict | Publication read | Topic |
|---|---|---|---|---|---|
| 1 | Musical instruments vs Hornbostel-Sachs | works on the cleaned MIMO set (59,661 photos): name AMI 0.59, H-S level 3 0.55, level 6 0.59, Western family 0.57; photo groups sound alike (AUC 0.91); origin weak | open on every axis | highest: formal hierarchy, 2 taxonomies, 129 names, 868 recordings, machine-readable H-S tree (OMARO); licence CC BY-NC-SA | 1, with 3 |
| 2 | Museum objects, function vs culture | works at scale: function AMI 0.51 on 24,581 Met objects, culture 0.33, culture inside one function 0.16 to 0.55; culture survives background removal for knives, lamps, mirrors and bowls, is photography for beads and combs; residual trick does not help at scale | open on the quantitative question; the methods exist on other data | high; the Met publishes CLIP and DINOv2 embeddings of the whole collection; object-only crops are the basis | 1 and 3 |
| 3 | Ghost Borders | field size separates borders between different agrarian regimes: existing border AUC 0.97, inner German 0.92 (placebos 0.59 and 0.66), Prussian-Russian 0.78 (0.53 and 0.63); no field-size signal for the Austrian-Russian line (0.55) or the 1946 Niedersachsen merger (0.62 vs 0.57 and 0.60) | open: no clustering test of vanished borders, no placebo design, no age curve | high in a remote-sensing or landscape venue; the story is conditional | 1 |
| 4 | Cars over decades | works: decade dominates, national style constant 1970s to 2010s | open; contradicts Costa and Aguzzi 2015 and Bluntzer and Ostrosi 2019; decade part known | medium-high; ad-photo image terms are the weak spot | 1 |
| 5 | Sacred buildings | partial: religion and country confounded; country recoverable inside churches | open; pipeline published once (WikiChurches), question not | medium-high; needs 3 more faiths and climate labels | 1 and 3 |
| 6 | Insects, image vs DNA | works: per-family agreement 0.03 to 0.48 | Heiskanen 2026 (Jyväskylä thesis) and InfoTaxa 2026 are 1 step away | medium-low, scoop risk high | 2 |
| 7 | Dogs, function vs ancestry | works technically; FCI groups partly defined by looks | E and Wang 2026 adjacent; clade table needs journal access | medium | 1 |
| 8 | Art emotion, visual-first | data works; finding reduces to language base rates | ArtECulture 2026 adjacent | weak | 1 |
| 9 | Minerals | no signal (crystal system AMI 0.04) | open, but only a null | drop | |
| 10 | Galaxies | ordinary (AMI 0.25); CLIP text blind to morphology | clustered before | dropped | |
| 11 | Scientific figures | not spiked | taken (Yang et al. 2019; Howe et al. 2017) | drop | |
| 12 | Newspaper layouts | blocked (Library of Congress behind Cloudflare; Europeana serves PDFs) | layout clustering exists; homogenisation open | parked | |
| 13 | Fantasy maps | not spiked | qualitative claims only | parked (copyright, hand collection) | |

## 1. Musical instruments vs Hornbostel-Sachs

Question: when instrument photos are clustered without labels, which taxonomy do the groups follow: the acoustic tree (Hornbostel-Sachs), the player's families (woodwind, brass, string, percussion, keyboard), the instrument type (name), or origin? And where the groups disagree with the experts, is the disagreement nameable?

Data: MIMO (Musical Instrument Museums Online) through the Europeana API, CC BY-NC-SA 4.0 on every record but one, which is marked in copyright (non-commercial, share-alike; the Met's open-access collection is CC0). 63,478 records with a thumbnail from 216 museums; cleaned set 59,661 after removing the Europeana placeholder image (247 records), copies of shared group photos (556) and catalogue records of parts (3,014: bows, mouthpieces, beaters, plectrums, reeds). 40,697 carry an H-S code from MIMO's extended vocabulary, 40,230 an instrument name from the MIMO keyword vocabulary linked to Wikidata (P3763) and English Wikipedia, 34,098 a Western family from the Wikidata tree (harmonica moved to woodwind, Jew's harp left without a family), 35,419 a place of origin, 5,887 a year. Coverage is uneven by museum: Tervuren (8,435 records) is 30 percent coded and 0 percent placed, Soinuenea 5 percent coded; the Western museums are 70 to 86 percent coded; coding depth is 6 to 7 digits in the Western museums and 4 in Berlin. 963 objects have a museum recording. 95 of these share a byte-identical file with another object (Paris attaches one ensemble recording to several instruments, Soinuenea one sample to many accordions) and are excluded, which leaves 868 recordings of the object itself (Stockholm 465, Berlin 158, Paris 107, Edinburgh 87, Soinuenea 19, Florence 18, Leipzig 14). Paris and Soinuenea serve many files as 404, and Edinburgh's sit under a different path than Europeana lists.

Pipeline: DINOv2-B embeddings of thumbnails padded to a white square (the default transform centre-crops the long side; thumbnails have a median aspect ratio of 1.5, and on 4,000 objects padding scores 0.015 AMI higher overall and 0.05 higher on elongated objects), PCA to 50 dimensions, k-means with k = number of classes, AMI against the label, a 5-fold logistic-regression probe as the ceiling, majority baseline. H-S levels are the first d digits of the code with the modifier suffix and compound codes stripped. Scripts: `spikes/mimo_clean.py`, `spikes/clean_analysis.py`, `spikes/figures_clean.py`, `spikes/mimo_audio_run.py`; outputs in `runs/clean` and `runs/figures_clean`. 

Results on the 27,348 rows that carry a code, a family and a name (classes with 30+ objects; levels 4, 5 and 6 are scored on the 26,181, 23,501 and 21,445 of them whose codes are long enough):

| Label | Classes | k-means AMI | Probe accuracy | Probe AMI |
|---|---|---|---|---|
| H-S level 1 | 5 | 0.35 | 0.97 | 0.86 |
| H-S level 2 | 8 | 0.53 | 0.97 | 0.88 |
| H-S level 3 | 13 | 0.55 | 0.92 | 0.83 |
| H-S level 4 | 18 | 0.55 | 0.86 | 0.77 |
| H-S level 5 | 30 | 0.57 | 0.87 | 0.79 |
| H-S level 6 | 30 | 0.59 | 0.88 | 0.81 |
| Western family | 6 | 0.57 | 0.96 | 0.87 |
| instrument name | 129 | 0.59 | 0.67 | 0.71 |

Agreement between the labels themselves (same rows): name vs H-S level 3 0.66, name vs level 4 0.74, family vs level 1 0.74, family vs level 2 0.80, level 6 vs level 3 0.81. A perfect name-level partition scores 0.66 against level 3; the blind groups score 0.55. Depth table on all 40,697 coded objects: k-means 0.33, 0.48, 0.53, 0.54, 0.56, 0.57 at levels 1 to 6 (5, 11, 22, 31, 47, 42 classes), Ward dendrogram cuts 0.32, 0.51, 0.55, 0.54, 0.57, 0.58. The level-1 dendrogram cut changes with the set of rows the tree is built on and is not quoted alone.

Blind groups labelled by their majority name (31,402 objects, 157 names): with 100 groups, 84 percent of objects receive the right level-3 code and 50 percent the right name, and 46 of 100 groups have one name covering at least half of them; with 400 groups, 90 and 62 percent, 250 of 400 groups. Names with 50+ objects whose main group (of 20) belongs to another H-S class: 8 of 112. Xylophones sit with the pan flutes (76 percent of all xylophones are in that group). Tube zithers (82 percent) and ratchets (51 percent) sit in a mixed group of long objects with trumpets, rattles and flutes. Electric guitars sit with the guitars, citterns and banjos (94 percent). Harmonicas (90 percent), synthesizers (93), pipe organs (67) and barrel organs (96) share one mixed group with harps and accordions (42 percent chordophones, 42 percent aerophones). Objects with the same name also split by museum: with 100 groups, a violin group of 148 is 88 percent Brussels and one of 125 is 65 percent Stockholm, and clarinet groups of 355 and 188 are 98 and 80 percent Edinburgh.

Methods on 12,000 of the same rows (AMI for level 3, family, name): k-means 0.55, 0.57, 0.57; Ward 0.56, 0.56, 0.59; average-link with cosine distance 0.58, 0.69, 0.58; UMAP to 10 dimensions then k-means 0.57, 0.59, 0.58; probe ceiling 0.81, 0.85, 0.68. Seeds move k-means by up to 0.016 (level 3 0.54 to 0.55). DINOv2-L gives 0.57, 0.57, 0.57; CLIP L/14 0.54, 0.49, 0.52; SigLIP B/16 0.52, 0.48, 0.53; an ensemble of DINOv2-L and SigLIP 0.58, 0.57, 0.57. Centering each museum's embeddings lowers all 3 (0.52, 0.51, 0.54), because museums differ in what they collect. Zero-shot text prompts for the 4 acoustic classes ("a photo of a wind instrument", "a string instrument", "a drum", "a percussion instrument that is struck or shaken") label 81 percent (CLIP L/14) and 86 percent (SigLIP) of the objects correctly, AMI 0.49 and 0.57, against a majority class of 60 percent.

Origin: region 0.13 at 5 classes (probe accuracy 0.84, majority 0.65), country 0.20 (76 classes), museum 0.18 (84 museums). Inside each museum, with one photographer and one backdrop, the H-S class is recoverable at 0.10 to 0.57 (13 museums, above 0.2 in 12) and region at 0.01 to 0.08 everywhere except Berlin (0.25, the only balanced collection). Wikipedia categories give an origin of the instrument type for 11,054 records; on the 5,626 that also carry a museum place, the 2 agree 78 percent of the time, and the blind groups match both about equally (AMI 0.12 for the type origin, 0.11 for the place), so origin is weak under both labels.

Audio (add-on): CLAP embeddings of the 868 recordings (3 windows of 10 seconds, averaged). CLAP was trained on audio paired with text descriptions, which name instruments, so on this side only the grouping is blind. Sound clusters vs H-S levels 1, 2, 3: 0.64, 0.64, 0.77, vs instrument name (7 names with 20+) 0.94; photo clusters on the same objects 0.38, 0.50, 0.69 and 0.85. Blind photo groups vs blind sound groups of the same objects: AMI 0.53, 0.61, 0.68, 0.64 at 5, 10, 20, 40 groups. Pair test, AUC that a shared group predicts a high audio similarity: photo group (40 groups) 0.91 (0.90 to 0.92 over 5 seeds), H-S level 3 0.91, name 0.97, museum 0.63, permuted codes 0.49. On the coded objects alone the photo groups give 0.92 against 0.91 for the code. Inside Berlin the photo group gives 0.85 and the code 0.80, inside Stockholm 0.80 and 0.97. CLAP's zero-shot names for sound groups are poor. Type-level check on FSD50K (7,012 clips balanced over 5 families): family 0.51 at 5 groups and 0.61 at 10, probe 0.89; 22 instrument leaves 0.77; CLAP was trained on Freesound, so this check is label-exposed, which the museum recordings are not.

Literature (sweep 14.10.2026): no work clusters instrument photographs against H-S; no AMI or tree-distance score between an image cluster tree and H-S, culture or date; nearest are Peron et al. 2018 (726 MIMO violins, hand-measured outlines), Strauch et al. 2025 (318 African harps, hand-coded characters, cladistics), Wang et al. 2026 (Applied Sciences: form classification of 2,755 Chinese ceramics from museum metadata, supervised DINOv2), MINERVA 2021 (instruments in paintings, supervised). H-S hierarchy-aware work exists for audio only. OMARO (Zenodo, CC0) is an organology ontology with a MIMO-derived set of 643 H-S classes and 2,724 instrument concepts. The closest image-clustering work is IC|TC (Kwon et al. 2024), which clusters photos of people playing 7 instruments by a text criterion, without museum objects or H-S.

Risks: Western-heavy (65 percent of placed objects European, 77 percent of family-labelled objects in the 4 Western families); coverage and coding depth differ by museum, so the deeper levels are mostly the Western museums' objects; the family label is Wikidata's and keeps some odd assignments; parts are excluded by vocabulary name and exact title only; 1,030 records carry 2 or more codes (the first is used); thumbnails are 400 px; the top-level dendrogram cut is unstable; licence is non-commercial.

Venues named: CHR, ISMIR, Digital Humanities Quarterly or ACM JOCCH.

## 2. Museum objects, function vs culture

Question: do humans make things alike? Do objects cluster by function (bowl, lamp, comb, knife, mirror) or by culture and period, and where do unrelated cultures converge on the same form?

Data: Met Open Access, CC0. 940 objects fetched across 7 departments and 10 functions (`spikes/museum_fetch.py`). The Met publishes CLIP, DINOv2, SigLIP2 and OpenCLIP embeddings of the collection on Hugging Face (metmuseum/openaccess_embeddings and siblings). ArtiFact (Met, Rijksmuseum, AIC; CC BY-NC-ND) for a multi-museum version, with culture labels mostly from the Met.

Sample of 940 objects (`spikes/museum_run.py`): function AMI 0.56 (DINOv2, k=10), department 0.02 to 0.20. Within one function, department AMI 0.26 to 0.39 (CLIP), probe 0.56 to 0.82 vs majority about 0.15. Residual clustering (subtract function class means) lifts department AMI from 0.09 to 0.21 with CLIP and drops function AMI from 0.48 to 0.02.

Met-wide (`spikes/met_large_run.py`, `runs/met_large`): 24,581 objects with a function and a culture from MetObjects.csv (35 functions, capped at 1,500 objects each; 518 raw cultures merged to 60 classes with 30+ objects), scored on the DINOv2-giant embeddings the Met publishes. Function AMI 0.51 (probe 0.77, majority 0.06), culture 0.33 (probe 0.60), department 0.32 (12 classes, probe 0.80), era 0.19 (5 classes, probe 0.74). Culture inside one function, cultures with 30+ objects: knives 0.55 (4 cultures), seals 0.54 (11), pendants 0.50, mirrors 0.48, bottles 0.47, vases 0.41, rings 0.39, lamps 0.38, bowls 0.37, plates 0.35, cups 0.33, down to 0.16 for the weakest functions. Subtracting the function class means before clustering lowers culture from 0.31 to 0.27 and department from 0.29 to 0.21 at this scale; the gain on the 940-object sample (department 0.09 to 0.21 with CLIP) does not hold at scale.

Confound test (`spikes/confound_run.py`, `runs/confound_museum`, the 940 sample objects in 3 variants: raw thumbnail, object-only crop from rembg, and a background-only frame of 15 percent): function survives the object-only crop (CLIP 0.48 to 0.50, DINOv2-S 0.56 to 0.53) and collapses on background-only (0.18 and 0.23). Department inside one function with CLIP, raw / object-only / background-only: knives 0.62 / 0.60 / 0.41, lamps 0.59 / 0.58 / 0.36, mirrors 0.60 / 0.56 / 0.34, bowls 0.40 / 0.36 / 0.21 (the culture signal is in the object); beads 0.44 / 0.25 / 0.49 and combs 0.28 / 0.27 / 0.37 (the signal is the photography). Any culture claim therefore rests on object-only crops, and the per-function confound check is part of the method.

Literature: MosAIc 2020 (cross-cultural retrieval on Met and Rijksmuseum, no clustering), Altaweel and Squitieri 2020 (sculpture faces, LSH, rising cross-regional similarity after the 4th century BCE), Culture Affordance Atlas 2026 (function labels on Dollar Street), IC|TC 2024 and Multi-MaP 2024 (criterion-conditioned and multiple clustering recipes on generic data), Fan et al. 2025 and StyleDecoupler 2026 (removing a known factor before clustering, on text and on artwork style). Nothing quantifies function vs culture on heritage objects or treats department photography as a confound.

Risks: department photography maps onto culture for some functions (beads, combs), so object-only crops and the background-only control are mandatory; the residual trick does not scale; culture classes need merging (518 raw values); Western-heavy collections; "convergence" needs an operational definition (clusters that contain several unrelated cultures within one function).

Venues named: ACM JOCCH, Journal of Archaeological Science, CHR or VISART.

## 3. Ghost Borders

Question: when a border disappears, how long does it stay visible in the landscape?

Data: Sentinel-2 L2A true colour plus red and NIR bands at 10 m from the Earth Search STAC API (no account), 2 dates (2025-05-10 to 06-25 and 2025-07-15 to 09-10), ESA WorldCover 2021 (cropland mask, class 40), REGIS 1870 to 2020 GeoPackages (CC BY 4.0) for all borders. Tiles 256 px (2.56 km), kept only where cropland is at least 60 percent, sampled 3 to 15 km from the line. Placebos: the same line shifted 35 km east and west.

Without the cropland mask and the second date (1 summer image, 150 tiles per set), the inner German border is not separable from its placebos (AMI 0.13 vs 0.09 to 0.10) and edge density is at chance, so both are part of the method.

Scripts `spikes/ghost_v2_fetch.py` and `spikes/ghost_v2_run.py`; 2,746 tiles over 5 borders and their placebos (240 real, 116 to 160 per placebo). Field size is the mean segment area on the second date; the probe and the clustering use DINOv2-S embeddings of both dates.

| Border | Ended | Mean field segment, side A vs B | AUC of segment area, real (95 percent bootstrap) | Placebos | Probe real vs placebos | Best AMI real vs placebos |
|---|---|---|---|---|---|---|
| Czech-Austrian | still exists | Czechia 12.2 ha vs Austria 4.1 ha | 0.973 [0.955, 0.988] | 0.78, 0.86 | 0.93 vs 0.79, 0.76 | 0.54 vs 0.26, 0.16 |
| Inner German | 1990 | former GDR 14.4 ha vs former FRG 7.9 ha | 0.916 [0.876, 0.946] | 0.59, 0.66 | 0.84 vs 0.74, 0.73 | 0.19 vs 0.04, 0.13 |
| Prussian-Russian partition, Poland | 1918 | former Prussian 6.6 ha vs former Russian 4.3 ha | 0.782 [0.722, 0.832] | 0.53, 0.63 | 0.81 vs 0.61, 0.60 | 0.14 vs 0.01, 0.04 |
| Austrian-Russian partition, Poland | 1918 | former Russian 2.4 ha vs former Austrian 2.3 ha | 0.548 [0.503, 0.621] | 0.66, 0.81 | 0.88 vs 0.52, 0.79 | 0.14 vs 0.26, 0.17 |
| Braunschweig and Oldenburg vs Hanover (Niedersachsen merger) | 1946 | 8.2 ha vs 6.5 ha | 0.620 [0.550, 0.696] | 0.57, 0.60 | 0.75 vs 0.71, 0.79 | 0.08 vs 0.10, 0.22 |

Reading: field size separates the existing border and the 2 vanished borders that divided different agrarian regimes (collectivised vs family farms; Prussian vs Russian land law) from their placebos. It shows nothing for the Austrian-Russian line, where field sizes are equal on both sides, or for the 1946 merger inside Niedersachsen, where the regime was the same. The probe on the embeddings is above both placebos for 4 borders (by 0.09 to 0.20 over the higher placebo), including the Austrian-Russian line, and below the western placebo for the 1946 line. Plain clustering recovers the existing border (0.54) and the vanished ones weakly or not at all (0.08 to 0.19, below a placebo for 2 lines). Visibility therefore depends on what differed across the line and on the time since it vanished, which is the age-curve claim in testable form. The gap between probe and AMI is the Topic 1 question (preprocessing, encoder, method). The explicit field measure is the baseline a reviewer will ask for. The 1952 Baden vs Württemberg line has 12 and 31 cropland tiles per side, too few to score.

More borders with REGIS: Baden vs Württemberg (1952), Hohenzollern (1952), Hesse-Nassau vs Hesse-Darmstadt (1945), Oldenburg, Braunschweig, Schaumburg-Lippe (1946), Lippe (1947), GDR Bezirke (1990), Austrian partition in Poland (1918), Limes as negative control.

Caveats: placebos are high in the probe (0.73 to 0.79 for 4 of the 5 borders), so probe margins are about 0.1; the western placebo of the Austrian-Russian line is higher than the line itself in field size; 1 summer, 1 year, 1 seed; the 1918 result needs a second segment; 5 borders is too few for an age curve. Encoders with coordinate input (Clay, Prithvi "-TL" checkpoints) must stay out.

Literature (handoff plus 13.10.2026 checks): Gatterer et al. 2024, Kuemmerle et al. 2009, Kosczor et al. 2024 and 2026, GeoVille, DLR 2024, Pinkovskiy 2017, Grosfeld and Zhuravskaya, CORINE-based Iron Curtain land-cover studies. No systematic clustering test of vanished borders, no placebo design, no age curve. The Polish partition border is a known positive ("widać zabory"); the quantification is the contribution.

## 4 to 6, kept at lower priority

Cars over decades (`spikes/cars_run.py`, `spikes/cars_era_run.py`): VMMRdb (291,752 images, 9,170 make-model-year classes, 1950 to 2016, MIT licence on the repository, image terms unstated). Decade AMI 0.43 (CLIP); balanced US, Japan, Germany probe 0.72 to 0.82 in every decade from the 1970s; no convergence at 21 to 40 cars per country per decade. Literature: Costa and Aguzzi 2015, Bluntzer and Ostrosi 2019, Jung 2018, Lee et al. 2013 (decade from photos), CaMiT 2025. Venues: Journal of Cultural Analytics, Design Science, CVFAD or VISART.

Sacred buildings (`spikes/sacred_fetch.py`, `spikes/sacred_run.py`): Wikidata plus Commons; 322 images in hand (Buddhist 150, church 120, Hindu 52); religion AMI 0.7 but confounded with country; within churches country AMI 0.4, era 0.18 to 0.24. Commons throttles this machine's IP after bursts; fetch at 1 request per 8 seconds or from another network. Literature: WikiChurches 2021, Kubba et al. 2025, MonuMAI, ArchiLense, Lee et al. 2015, Sun et al. 2022. Venues: NeurIPS Datasets and Benchmarks, SUMAC, Journal of Cultural Heritage.

Insects (`spikes/insects_run.py`): BIOSCAN-30k slice; family AMI 0.43 (DINOv2); per-family image-vs-BIN agreement 0.03 (Psychodidae) to 0.48 (Dolichopodidae). Literature: BIOSCAN-5M 2024 (global AMI), Heiskanen 2026 (per-taxon at order and family, BIOSCAN-1M, BioCLIP), InfoTaxa 2026 (BioCLIP 2, family AMI 0.79), CLIBD 2025. TreeOfLife-200M includes BIOSCAN-5M, so BioCLIP is label-exposed.

## 7 to 13, checked and parked or dropped

Dogs (`spikes/dogs_run.py`): Stanford Dogs, 114 breeds with FCI group from the Wikipedia infobox standard link; breed AMI 0.70 (DINOv2), FCI group probe 0.83, country recoverable after removing group (AMI 0.38). Parker 2017 clade table sits in the Cell Reports supplement (journal access).

Art emotion (`spikes/art_emotion_run.py`): ArtELingo (32,085 joinable paintings via artist and title slug to Artificio/WikiArt); clusters follow genre (0.42) and style (0.29), not emotion (0.05); cross-language divergence is mostly base rates (Chinese annotators 50 percent contentment, English 30).

Minerals (`spikes/minerals_run.py`): MineralImage5k-98; crystal system AMI 0.04, probe 0.40 vs majority 0.21.

Galaxies (`spikes/galaxies_run.py`): Galaxy10 DECaLS; AMI 0.25; CLIP zero-shot with class prompts 0.07.

Scientific figures: Yang et al. 2019 clustered 5M arXiv figures into field signatures. Newspapers: Library of Congress endpoints return Cloudflare challenges to scripts; Europeana returns PDFs. Fantasy maps: copyright on the fantasy side, hand collection.

## Environment

Python 3.12 venv in `.venv` (uv), torch 2.11 cu128 on an RTX 5080, open_clip, timm, scikit-learn, hdbscan, geopandas, rasterio, pystac-client, datasets. Shared code: `spikes/common.py` (encoders with optional pad-to-square, embedding cache keyed on the image list, H-S level helper, clustering, scoring), `spikes/labels.py` (multi-label scoring with probe and majority baseline), `spikes/wiki.py` (polite Wikipedia and Commons client). Data under `data/` and `dl/` (git-ignored). REGIS supplement at `dl/regis_esm.zip` (253 MB).
