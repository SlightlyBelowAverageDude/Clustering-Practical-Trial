# Woche 6 – Erklärungen für die Präsentation (Deutsch)

Begleitdokument zu `Week6_Final_FullMIMO_StatusUpdate.pptx`. Es erklärt die Fachbegriffe in einfacher Sprache und bereitet auf Fragen von Dr. Schlegel vor. Alle Zahlen stammen aus den geprüften Ergebnisdateien (`final_claims.json`, 224/224 Prüfungen bestanden).

## 1. Worum geht es?

Wir prüfen, ob Computer Musikinstrumente **ohne Etiketten** so gruppieren können, wie Expertinnen und Experten sie klassifizieren. Die Experten-Klassifikation heißt **Hornbostel–Sachs (HS)**. Sie ordnet Instrumente danach, *wie der Ton entsteht*, zum Beispiel durch eine schwingende Saite, eine schwingende Luftsäule oder ein Fell. Ein Foto zeigt dagegen vor allem die *Form*. Daher kommt unser Titel: *Visual Form vs. Acoustic Function*.

**HS3** ist die dritte Ebene dieser Hierarchie. Im Benchmark gibt es **13 HS3-Klassen**, zum Beispiel 421 „Flöten“, 422 „Rohrblattinstrumente“, 321 „Lauten“.

## 2. Die Daten – vier Begriffe sauber trennen

| Begriff | Anzahl | Bedeutung |
|---|---|---|
| MIMO-Datensätze | 63.478 | Alle Instrumenten-Datensätze mit Bild aus Europeana |
| **Volle MIMO-Sammlung** | **59.658** | Bereinigte Fotos (ohne Platzhalterbilder, kopierte Gruppenfotos, Zubehör wie Bögen oder Mundstücke) |
| **Ausgewählter Benchmark** | **27.347** | Objekte mit HS3-Code, Instrumentenfamilie und Instrumentenname; jede Klasse hat mindestens 30 Objekte |
| **Entwicklungs-Benchmark** | **12.000** | Teil des Benchmarks, auf dem **alle heutigen Experimente** laufen |
| **Held-out-Set** | **14.696** | Zurückgehaltene Objekte. Sie wurden **nicht** benutzt und dienen erst ganz am Ende zur fairen Prüfung |
| Ausgeschlossen | 651 | Schon im alten Pilot (2.000 Bilder) angeschaut, deshalb nicht mehr „unbekannt“ |

**Warum bedeuten 59.658 Fotos nicht 59.658 bewertete Objekte?**
- Zum Bewerten braucht jedes Objekt ein Experten-Etikett. Etwa ein Drittel der Fotos hat keinen HS-Code.
- Sehr seltene Klassen werden entfernt.
- Vom Rest wird ein Teil als Held-out-Set zurückgehalten.
- Gerechnet haben wir deshalb bisher auf **12.000** Objekten. Man darf nie sagen, wir hätten alle 60.000 Fotos geclustert.

## 3. Fachbegriffe einfach erklärt

**Frozen Embeddings (eingefrorene Einbettungen):** Ein vortrainiertes Bildmodell wandelt jedes Foto in eine Liste von Zahlen um (z. B. 768 Zahlen). Ähnliche Bilder bekommen ähnliche Zahlen. „Eingefroren“ heißt: Wir trainieren das Modell **nicht** weiter, wir benutzen es nur.

**CLIP, SigLIP, DINOv2:** drei solche Bildmodelle.
- **CLIP** und **SigLIP** wurden mit Bild-Text-Paaren aus dem Internet trainiert. Sie haben also gelernt, Bilder zu Beschreibungen zu passen. SigLIP ist eine neuere Variante mit anderem Trainingsverfahren.
- **DINOv2** wurde **nur mit Bildern**, ohne Text, trainiert. Es lernt visuelle Ähnlichkeit selbst.
- Wir nutzen CLIP ViT-B/32, SigLIP ViT-B/16 und DINOv2-B/14 (Bildgröße 224 Pixel).

**Clustering:** Der Computer bildet Gruppen ähnlicher Bilder, **ohne** die Experten-Etiketten zu sehen. Erst danach vergleichen wir die Gruppen mit den Etiketten.

**k-Means:** wählt k Gruppenzentren und ordnet jedes Bild dem nächsten Zentrum zu. Das Ergebnis hängt etwas vom Startpunkt ab, deshalb rechnen wir mit 5 Zufalls-Seeds und geben Mittelwert ± Streuung an.

**Agglomeratives Clustering (average linkage, Kosinus):** Am Anfang ist jedes Bild eine eigene Gruppe. Dann werden immer die zwei ähnlichsten Gruppen verschmolzen, bis k Gruppen übrig sind. Das Verfahren ist deterministisch (kein Zufall).

**Spectral Clustering:** baut zuerst ein Nachbarschafts-Netz (jedes Bild mit seinen 15 nächsten Nachbarn) und schneidet dieses Netz dann in k Teile. Es ist gut für unregelmäßig geformte Gruppen, kann aber eine riesige „Sammelgruppe“ bilden.

**GMM und HDBSCAN:** GMM beschreibt die Daten als Mischung von Glockenkurven. HDBSCAN sucht dichte Bereiche und kann Bilder als „Rauschen“ übrig lassen; Rauschen zählen wir als eigene Gruppe.

**AMI vs. ARI (die zwei Bewertungsmaße):**
- **AMI** (Adjusted Mutual Information) misst, wie viel *Information* die Gruppen über die Experten-Klassen enthalten. 0 = Zufall, 1 = perfekt.
- **ARI** (Adjusted Rand Index) zählt *Bildpaare*: Liegen zwei Bilder derselben Klasse auch in derselben Gruppe, und umgekehrt? 0 = Zufall, 1 = perfekt.
- Unterschied: Eine einzige riesige, gemischte Gruppe erzeugt sehr viele falsche Paare. Das bestraft ARI stark, AMI weniger. Deshalb zeigen wir beide.
- Beispiel: DINOv2-B + Spectral packt **50,1 %** aller Objekte in eine Gruppe und hat ARI 0,165.

**NMI:** ähnlich wie AMI, aber ohne Zufallskorrektur. Wir speichern es, zeigen es aber nicht prominent.

**Zentrieren und L2-Normalisierung:** Beim Zentrieren ziehen wir von allen Embeddings den Durchschnittsvektor ab. Bei der L2-Normalisierung bringen wir jeden Vektor auf Länge 1. Rohe CLIP- und SigLIP-Embeddings zeigen alle fast in dieselbe Richtung. Ohne Zentrieren sehen deshalb alle Bilder „ähnlich“ aus, und agglomeratives Clustering wirft **99,8 %** in eine Gruppe (AMI ≈ 0,00). Mit Zentrieren steigt der AMI auf 0,512 (CLIP) bzw. 0,542 (SigLIP).
- **Globales** Zentrieren (über alle Bilder) ist etwas anderes als Zentrieren pro Museum. Das zweite haben wir noch nicht getestet.

**PCA:** reduziert die vielen Zahlen auf die 50 bzw. 100 wichtigsten Richtungen. Bei DINOv2-B mit agglomerativem Clustering bringt PCA100 einen kleinen Gewinn (0,568 statt 0,561), bei anderen Kombinationen nicht.

**Entwicklungs- vs. Held-out-Daten:** Auf den Entwicklungsdaten probieren wir aus und vergleichen. Weil wir dort Entscheidungen treffen, sind diese Zahlen eher etwas zu optimistisch. Das Held-out-Set wird erst **einmal am Ende** mit der eingefrorenen Pipeline geprüft. Erst das ist eine faire Schätzung.

**Festes k vs. blindes k:** k ist die Anzahl der Gruppen.
- Wir haben **k = 10 vorher festgelegt**, ohne auf die Etiketten zu schauen.
- k = 13 entspricht genau der Zahl der HS3-Klassen, nutzt also Etiketten-Wissen. Das ist nur eine Referenz im Backup (DINOv2-B + agglomerativ: 0,584).
- **Blindes k** heißt: k ohne Etiketten automatisch wählen (z. B. per Silhouette oder Eigengap). Das ist noch offen.

## 4. Die drei Kernaussagen (Entwicklungs-Benchmark, 12.000 Objekte, HS3, k = 10)

1. **DINOv2-B ist mit k-Means und agglomerativem Clustering am besten**: AMI 0,538 bzw. 0,561 (SigLIP 0,523 / 0,542; CLIP 0,451 / 0,512). Aber **nicht** mit Spectral: Dort ist SigLIP besser (0,559 vs. 0,476).
2. **Welche Methode „die beste“ ist, hängt vom Modell und vom Maß ab.** DINOv2-B + agglomerativ (AMI 0,561, ARI 0,417) und SigLIP + Spectral (AMI 0,559, ARI 0,311) sind beim AMI gleichauf. Beim ARI und bei der Gruppengröße (21,6 % vs. 34,7 % in der größten Gruppe) ist DINOv2-B + agglomerativ ausgewogener. Wir sagen trotzdem nicht, dass es universell am besten ist.
3. **Zentrieren ist entscheidend** für agglomeratives Clustering mit CLIP und SigLIP (0,002 → 0,512; 0,001 → 0,542). DINOv2-B ist roh robuster (0,361), profitiert aber ebenfalls.

## 5. Was wir NICHT behaupten dürfen

- keine Ergebnisse auf allen 59.658 Fotos;
- keine Held-out-Ergebnisse;
- DINOv2-S und -L wurden **nur auf dem Pilot** (2.000 Bilder) getestet, nicht auf 12.000 Objekten; die Pilotwerte sind nicht mit den Entwicklungswerten vergleichbar;
- keine statistische Signifikanz (nur ein Split, keine Konfidenzintervalle);
- die Werte 0,53 / 0,83 / 0,68 aus dem Projektplan werden heute nicht direkt verglichen.

## 6. Wahrscheinliche Fragen von Dr. Schlegel – kurze Antworten

**„Sind das Ergebnisse auf allen 60.000 Fotos?“**
Nein. Es sind 12.000 Entwicklungsobjekte aus dem 27.347-Objekte-Benchmark. Die volle Sammlung ist rekonstruiert, aber noch nicht vollständig geclustert.

**„Ist DINOv2 also das beste Modell?“**
Für k-Means und agglomeratives Clustering auf unseren Entwicklungsdaten ja, für Spectral nicht. Das ist ein einzelner Split ohne Signifikanztest.

**„Warum k = 10 und nicht 13?“**
k = 10 war vorab festgelegt und nutzt kein Etiketten-Wissen. 13 wäre die bekannte Klassenzahl, also nur eine Referenz. Das blinde Wählen von k ist ein offenes Thema für Woche 8.

**„Warum ist AMI so ähnlich, ARI aber so verschieden?“**
ARI bestraft große gemischte Gruppen stark, weil es Bildpaare zählt. SigLIP + Spectral hat eine größere Sammelgruppe (34,7 %) als DINOv2-B + agglomerativ (21,6 %).

**„Warum bricht agglomeratives Clustering ohne Zentrieren zusammen?“**
Rohe CLIP- und SigLIP-Vektoren teilen eine starke gemeinsame Richtung. Die Kosinus-Abstände werden dadurch winzig, und alles verschmilzt zu einer Gruppe. Das Zentrieren entfernt diese gemeinsame Richtung.

**„Habt ihr das Held-out-Set angeschaut?“**
Nein. Es ist reserviert; das wurde technisch geprüft (keine Held-out-Embeddings). Vorher müssen noch die Objekte mit Audioaufnahmen ausgeschlossen werden.

**„Könnten die Gruppen nur Museen widerspiegeln?“**
Das ist möglich: Die fünf größten Museen stellen 53 % des Benchmarks. Wir haben das noch nicht untersucht; es ist für Woche 8 geplant.

**„Welche Bildauflösung bei DINOv2?“**
224 Pixel auf den Entwicklungsdaten. 518 Pixel (der Standard des Modells) lief nur auf dem Pilot, aus Rechenzeitgründen (nur CPU).

**„Wie sicher sind die Zahlen?“**
Ein separates Prüfskript hat jede Zahl aus den gespeicherten Gruppenzuordnungen neu berechnet (224/224 korrekt). k-Means und Spectral wurden mit 5 Seeds gerechnet; die Streuung liegt bei höchstens ±0,011.

## 7. Zeitplan (10 Minuten)

Folie 1: 0:40 · Folie 2: 1:00 · Folie 3: 1:10 · Folie 4: 1:00 · Folie 5: 1:20 · Folie 6: 1:20 · Folie 7: 1:10 · Folie 8: 1:00 · Folie 9: 0:50 → **9:30**, danach Fragen.
