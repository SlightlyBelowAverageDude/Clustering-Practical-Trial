# Instruments sorted without labels

What the project does, what came out, and how to read the numbers. Figures are in `runs/figures_clean_v3/`. All numbers come from the cleaned data set (`runs/clean/`, audio from `runs/clean/audio_dedup/`).

## 1. The data

MIMO (Musical Instrument Museums Online) publishes photos of museum instruments through Europeana, each with a catalogue card. The card gives the instrument's name, its origin, the museum, and a Hornbostel-Sachs code. Hornbostel-Sachs is the expert classification organologists have used since 1914. It sorts instruments by how the sound is made: a vibrating string, a vibrating skin, air, or the body of the object itself, and then subdivides each class in a tree of numbered levels.

We harvested 63,478 records from 216 museums. Cleaning removed 247 placeholder images, 556 copies of photos shared by several records, and 3,014 records of instrument parts (bows, mouthpieces, beaters, reeds). That leaves:

| | records |
|---|---|
| photos after cleaning | 59,661 |
| with a Hornbostel-Sachs code | 40,697 |
| with an instrument name linked to Wikipedia | 40,230 |
| with a Western family (woodwind, brass, string, percussion, keyboard) | 34,098 |
| with a place of origin | 35,419 |
| with a usable recording of the same object | 868 |

The images are licensed CC BY-NC-SA 4.0 (all records but one): free for research, no commercial use.

## 2. From photo to point

A vision model called DINOv2 turns each photo into 768 numbers. It was trained on 142 million web images without any labels, so it has never been told what an instrument is. Photos that look alike get similar numbers. Each photo becomes a point in a space where lookalikes sit close together. Before this step each photo is padded to a square, so long objects such as flutes are not cut off.

## 3. Clustering

Clustering means letting the computer find the crowds of points on its own. We give it only the number of groups. The standard method, k-means, places that many centres and assigns every photo to its nearest centre. No labels are used at any point. The question is what the resulting groups correspond to: the expert code, the instrument name, the Western family, the origin, or the sound.

![pipeline](runs/figures_clean_v3/figP_pipeline.png)

## 4. How the groups are scored

AMI (adjusted mutual information) measures how well the blind groups match a label the computer never saw. 1 means the same sorting. 0 means no relation; random sorting scores 0 because the luck of chance matches is subtracted.

As a reference we also train a linear probe: a simple classifier that sees the labels of 80 percent of the photos and guesses the other 20 percent from the same 768 numbers. It shows how much of a label is readable in the numbers at all. It is a practical reference for blind clustering, not a strict upper limit.

## 5. Results

### The groups are instrument types

![three groups](runs/figures_clean_v3/figA_rows.png)

Three of 20 blind groups. The frame colour is the expert's top-level class. Xylophones land with the pan flutes (76 percent of all xylophones are in that group: rows of tubes and bars). Electric guitars land with the guitars, citterns and banjos (94 percent: a neck and a body). Tube zithers land in a group of long objects with flutes and trumpets (82 percent). Each group crosses the expert's class lines in a way you can see.

### Finer labels fit better

Scored on 27,348 objects that carry a code, a family and a name (levels 4 to 6 on 26,181, 23,501 and 21,445 objects, because shorter codes drop out):

| label | classes | blind k-means (AMI) | probe (AMI) |
|---|---|---|---|
| Hornbostel-Sachs level 1 | 5 | 0.35 | 0.86 |
| level 2 | 8 | 0.53 | 0.88 |
| level 3 | 13 | 0.55 | 0.83 |
| level 4 | 18 | 0.55 | 0.77 |
| level 5 | 30 | 0.57 | 0.79 |
| level 6 | 30 | 0.59 | 0.81 |
| Western family | 6 | 0.57 | 0.87 |
| instrument name | 129 | 0.59 | 0.71 |

![depth](runs/figures_clean_v3/figB_depth.png)

The blind groups match the 5 top expert classes worst and the instrument names best. The camera sorts by shape, which the names and the families describe better than the top of the acoustic tree.

### How good is 0.55

![reference](runs/figures_clean_v3/figC_reference.png)

Against the 13 level-3 codes: random sorting scores 0, the blind groups 0.55, a perfect sorting by instrument name 0.66, the probe 0.83. Names and codes disagree with each other too, so 0.66 is what a flawless name-level typology would reach.

A second view of the same question: label each blind group with the name most of its objects carry, then count how many objects got the right label.

| 100 blind groups | k-means | same group sizes, random members |
|---|---|---|
| objects with the right level-3 code | 84 percent | 23 percent |
| objects with the right instrument name | 50 percent | 7 percent |
| groups where one name covers at least half | 46 | 0 |

### Better methods exist, and the gap is the project

On 12,000 of the same objects (AMI against level 3, family, name):

| method | level 3 | family | name |
|---|---|---|---|
| k-means | 0.55 | 0.57 | 0.57 |
| average-link clustering, cosine distance | 0.58 | 0.69 | 0.58 |
| probe | 0.81 | 0.85 | 0.68 |

![cluster map](runs/figures_clean_v3/figJ_cluster_map.png)

### It is the object, not the camera

![museums](runs/figures_clean_v3/figE_museums.png)

Inside one museum the photographer and the backdrop are the same for every object. There the blind groups still match the expert class at 0.10 to 0.57 across 13 museums (above 0.2 in 12). Origin is close to 0 everywhere (0.01 to 0.08) except Berlin (0.25), the only museum with a balanced world collection.

### Add-on: instruments that look alike sound alike

868 objects come with a museum recording of that same instrument. (95 more were excluded because their file is shared with other objects: ensemble recordings in Paris, one sample for many accordions in Soinuenea.) The recordings are turned into numbers by CLAP, an audio model trained on audio paired with text descriptions, which do name instruments, so the sound side is not label-free in training. The grouping itself is blind.

![pair test](runs/figures_clean_v4/figF_pair_auc.png)

Take pairs of objects. AUC says how often a pair from the same group sounds more alike than a pair from different groups; 0.5 is a coin flip. Sharing a blind photo group gives 0.80 with 17 groups and 0.91 with 40 groups (0.90 to 0.92 over 5 seeds). Sharing the expert level-3 code, 17 classes on these objects, gives 0.91, sharing a museum 0.63, permuted codes 0.49. With sound features that never saw a label (timbre statistics from the spectrum, HuBERT) photo groups and code score within 0.05 of each other at equal group counts, between 0.70 and 0.84.

![sound vs photo](runs/figures_clean_v3/figG_audio_vs_image.png)

Sound groups follow the acoustic tree more closely than photo groups (top level 0.64 against 0.38), which is expected for a classification by sound production.

![cross-modal](runs/figures_clean_v3/figH_crossmodal.png)

Photo groups against sound groups of the same objects, 10 each (AMI 0.60; 0.68 with 20 each). Ocarinas, natural trumpets, kohorns, guitars and harpsichords each land almost entirely in one sound group. Violins share their sound group with the guitars. The bell group spreads over many sound groups: small percussion looks alike and sounds different.

## 6. What the project would be

Build the blind typology properly, at several levels of detail, and compare it with the expert tree and the names. The method part is closing the gap between blind clustering and the probe (better clustering than k-means, a hierarchy scored as a tree). The finding part is where the camera and the expert agree, where they disagree, and whether the sound sides with the camera or the expert. Nobody has scored image clusters of museum instruments against the Hornbostel-Sachs tree; the closest work clusters photos of people playing 7 instruments (IC|TC, 2024) or measures violin outlines by hand (Peron et al. 2018).

## 7. Limits

- 65 percent of the objects with a known origin are European, and most named objects belong to the 4 Western families.
- Coverage differs by museum: the largest African collection (Tervuren) is 30 percent coded and has no places.
- The family label comes from Wikidata's tree and keeps some odd assignments.
- Thumbnails are about 400 pixels on the long side.
- The top-level 5-group cut changes with the set of objects clustered and should not be quoted alone.

## 8. Terms

| term | meaning |
|---|---|
| embedding | the 768 numbers DINOv2 writes for a photo; its position on the map |
| clustering, k-means | finding groups without labels; k is the number of groups |
| AMI | 0 to 1 score of how well blind groups match a hidden label, chance removed |
| linear probe | a classifier trained on labels, used as a reference for what the numbers contain |
| AUC | 0.5 to 1 score of how often same-group pairs beat different-group pairs |
| Hornbostel-Sachs | the 1914 classification of instruments by how the sound is made, written as codes like 321.322 |
| MIMO | Musical Instrument Museums Online, the source of photos, catalogue cards and recordings |
