# Einschätzung PedRecNet v2 gegenüber v1, Stand 31.10.2026

Die v2-Richtung ist für ein effizientes Mehrzwecknetz sinnvoll. Ein allgemeiner
Genauigkeitsgewinn gegenüber v1 ist noch nicht belegt. Den Backbone würde ich
derzeit beibehalten und zuerst die kanonische Geometrie und die Orientierungs-
Fehlerverteilung kontrolliert prüfen. Die vorhandenen Daten rechtfertigen keinen
pauschalen Rückbau auf v1 oder den Austausch der gesamten Architektur.

## Was die Ergebnisse tatsächlich zeigen

Quellen: [Referenztabelle](README.md),
[neue Best-Validierung](../../data/models/pedrec/single_results/experiment_pedrec_v2_p2d3d_c_o_0_best_validation.log),
[Becken-Notebook](../../notebooks/pelvis_consistency.ipynb).

| Größe | v1-Referenz mit MEBOW | v2, neuer Best-Checkpoint | Einordnung |
| --- | --- | --- | --- |
| COCO Body-Phi, Trainingsvalidierung | ca. 10,3° | ca. 9,2° | v2 besser; Orientierung wird nicht adaptiert |
| SIM Body / Head Phi | ca. 2,3° / 3,4° | ca. 4,5° / 4,5° | anderer Split; v1 validierte auf mittrainierten Daten |
| H36M MPJPE, Trainingsvalidierung | 64,37 mm | ca. 70,5 mm adaptiert | v2 weiter schlechter; unterschiedliche Ausgabekonventionen |
| H36M MPJPE ohne Adapter, neue Validierung | hier kein entsprechend neu ausgewerteter v1-Wert | 150,6 mm | enthält echte Konventionsdifferenzen und möglicherweise Inkonsistenz |
| COCO PCK@0.05 | 53,72 % | 54,61 % kanonisch / 55,42 % adaptiert | selbst kanonisch etwas besser; noch kein COCO-AP-Vergleich |
| TUD Body-Phi, separates Eval | 14,37° | noch nicht ausgewertet | Generalisierung offen |
| Reale Head-Orientierung | keine passende aktuelle Vergleichsevaluation | keine passende aktuelle Vergleichsevaluation | aus COCO-Body nicht ableitbar |

9,2° aus der Trainingsvalidierung nicht direkt gegen 10,05° aus dem separaten
v1-MEBOW-Eval stellen; zunächst gleiches Skript und gleiche Filter verwenden.
Auch ein erneutes Auswerten alter v1-Gewichte auf dem neuen SIM-Holdout macht
diesen nicht zu unbekannten Daten: v1 hat diese Avatare bereits mittrainiert.
Für eine faire Generalisierungsaussage müssten beide Modelle denselben sauberen
Trainings-/Holdout-Split verwenden.

Dass ein Modell bei ähnlichem oder besserem Mittelwert schlechter wirkt, bleibt
möglich: wenige 90°-/180°-Fehler oder eine schwache Winkelregion können entscheidend
sein. Deshalb erfasst das Notebook auch Median, P95, Acc22.5, Fehler >90° und
Fehler nach GT-Azimut, getrennt nach Körper/Kopf und Dataset.

## Welche Architekturteile sich wirklich geändert haben

Im Git-Branch `main` geprüft: `pedrec_net.py`, `pedrec_pose_head_3d.py` und
`pedrec_orientation_head_shared.py`. Die folgenden Aussagen trennen Codebefund
von möglichen Ursachen.

### Backbone und Pose

v1 verwendet ResNet-50 und Deconvolutions, v2 HGNetV2-B3 und eine FPN-artige Fusion
von Stride 4/8/16/32. Beide liefern Pose-Heatmaps mit 64×48 bei 256×192 Eingabe.
Laut Repository-Benchmark sinkt der Rechenaufwand pro Crop von 6,28 auf 2,45 GMACs
(ca. 61 % weniger). Die vorhandenen CPU-Pipeline-Zeiten verwenden Zufallsgewichte;
sie belegen weder GPU-Latenz noch Genauigkeit eines trainierten Modells.

Die kleineren Rechenkosten und die bisher erhaltene 2D-Qualität sprechen für
diesen Backbone als Arbeitsbasis. Sie beweisen keine gleichwertigen Merkmale für
3D oder kleine Köpfe. Ein kausaler Backbone-Vergleich braucht identische Heads,
Trainingsdaten und Trainingsbudget; aktuell ändern sich zu viele Faktoren.

Der 3D-Grundansatz ist **in beiden Versionen** eine XY-Heatmap mit Soft-Argmax plus
heatmapgewichteter Tiefenkarte. v2 hat hier keinen grundsätzlich neuen
Regressionsansatz eingeführt. Differenzierbare Integral-Regression ist grundsätzlich
ein etablierter Weg, siehe [Sun et al., Integral Human Pose Regression](https://arxiv.org/abs/1711.08229).
Die v1/v2-Tiefenkarte ist eine kompakte Ausprägung und kein vollständiges
volumetrisches Modell aus dieser Arbeit.

### Kanonische Geometrie und Adapter

Die Wurzel und beide Hüften werden bislang unabhängig vorhergesagt. Der Decoder
erzwingt nicht `hip_center = (left_hip + right_hip) / 2`, obwohl die kanonische
Konvention genau das verlangt. Das ist ein konkreter struktureller Schwachpunkt.
Mit dem neuen Notebook lässt er sich ohne GT und ohne Adapter auf realen Bildern
messen. Ein kleiner Konsistenzfehler ist eine notwendige Kontrolle, aber kein
hinreichendes Kriterium für gute Pose.

Die Idee, Label-Konventionen im Multi-Dataset-Training anzupassen, ist sachlich
begründet. Die derzeit frei gelernten affinen Gewichte können jedoch geometrisch
unerwünschte Freiheitsgrade absorbieren. Der Root-Fix beseitigt den Hüft-Pfad,
begrenzt aber nicht jede Abschwächung einer Verschiebung gegen die Wurzel:

* Gespeicherte neue H36M-Root-Zeile: ca. `0,453 root + 0,731 spine_center − 0,184 neck`.
* Eine künstliche gemeinsame Verschiebung aller anderen Gelenke um 10 mm kommt
  am adaptierten Knie noch mit 4,53 mm an; vorher waren es 1,38 mm.
* Das ist ein analytischer Test am Adapter, keine Messung der tatsächlichen Drift.
* Die gelernte Root-Konventionsverschiebung auf der Referenzpose beträgt etwa
  10,3 cm. Deshalb ist eine große kanonisch/adaptiert-Lücke nicht automatisch
  ein Fehler. Die frühere Erwartung einer Lücke von nur 1–3 cm war zu optimistisch.

Bei einem deutlich erhöhten kanonischen Beckenfehler ist mein erster
Trainingsversuch eine explizite geometrische Bindung der kanonischen Wurzel
oder ein gezielter Konsistenzterm. Eine harte Bindung setzt die Notebook-Metrik
automatisch auf null: Erfolg muss zusätzlich an Posefehlern, Proportionen,
Bildbeispielen und Generalisierung gemessen werden. Nicht nachträglich nur für
die Evaluation eine inkonsistente Pose reparieren.

### Orientierung: Biternion ist kein grundsätzlich schlechterer Ersatz

v1 erzeugt 360 Phi- und 180 Theta-Logits und liest daraus einen **linearen**
Soft-Argmax aus. Die tatsächliche Supervision erfolgt über den Winkelfehler des
Punktwertes (`AngularErrorLoss`), nicht über eine explizit vorgegebene
Winkel-Heatmap. Eine v1-Verteilung mit gleichen Peaks bei 1° und 359° würde
mit diesem Decoder ungefähr 180° ergeben. Die Kreisgeometrie wird im Decoder
also nicht korrekt abgebildet.

v2 schätzt Sinus/Cosinus mit Cosine-Loss und `atan2`. Das behandelt den Umlauf
korrekt und ist ein etablierter Ansatz, siehe
[Beyer et al., Biternion Nets](https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf).
Die bessere COCO-Body-Validierung liefert aktuell keinen Grund, diese Darstellung
pauschal zu verwerfen.

Es gibt dennoch sinnvolle Gegenargumente und offene Punkte:

* Ein einzelnes Biternion liefert eine Richtung, keine frei geformte Verteilung
  über mehrere plausible Richtungen. Bei Front/Back-Ambiguität kann der Kreis-
  Mittelwert instabil oder wenig aussagekräftig sein. Die Norm wird im aktuellen
  Loss gegen eins gezogen; sie ist daher nicht einfach eine kalibrierte Unsicherheit.
* v2 verkleinert gleichzeitig die Pose-Einbettung von 768 auf 256 Merkmale und
  ersetzt die 1D-Deconvolutions durch ein MLP. Ein Unterschied wäre deshalb
  nicht allein der Winkelrepräsentation zuzuschreiben.
* Beide Versionen poolen globale Stride-32-Bildmerkmale (bei dieser Eingabe 8×6).
  Ein eigener Kopfbereich wird nicht gezielt hochauflösend ausgelesen. Ein
  Kopf-RoI aus Stride-4/8-Features ist deshalb ein plausibler Versuch für
  Head-Orientierung; ein Vorteil ist noch nicht gemessen.
* Beide Versionen verwenden `pose.detach()` für die Orientierungs-Eingabe.
  Orientierung trainiert den gemeinsamen Bild-Backbone, gibt aber über diesen
  Pose-Zweig keinen direkten Gradienten an die Pose zurück. Das ist kein
  neu eingeführter v2-Nachteil.
* Cosine-Loss und der v1-Winkelfehler gewichten kleine und große Fehler anders.
  Ein besserer Mittelwert beweist deshalb keine besseren Verteilungsschwänze.

Wenn eine Verteilung gewünscht ist, wäre eine **zirkuläre Bin-Verteilung mit
geglätteten Labels** eine saubere Alternative. MEBOW verwendet 72 Bins und eine
zirkuläre Gauß-Zielverteilung, siehe
[Wu et al., MEBOW](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_MEBOW_Monocular_Estimation_of_Body_Orientation_in_the_Wild_CVPR_2020_paper.pdf).
Das ist nicht identisch mit v1s nur indirekt trainierter Logit-Verteilung.
Auch bei Bins muss die Punkt-Auslese Kreisgeometrie berücksichtigen: zirkulärer
Mittelwert oder Modus, mit separater Behandlung mehrdeutiger Verteilungen.
Ein linearer Erwartungswert über den Winkelbereich wäre wieder derselbe Fehler.

## Reihenfolge der nächsten Entscheidungen

| Priorität | Versuch | Festhalten | Aussage / Entscheidung |
| --- | --- | --- | --- |
| 1 | Becken-Notebook alt/neu ausführen | identische Bilder, nacktes Inferenznetz | Ist die interne Inkonsistenz auf realen Bildern tatsächlich kleiner? |
| 2 | Falls auffällig: kanonische Root-Konsistenz im Training absichern | Backbone, Datenmix, übrige Heads und Schedule | Behebt Geometrie, ohne Posequalität zu verlieren? |
| 3 | Biternion gegen zirkulären Bin-Head | gleicher v2-Backbone, gleiche Pose-Einbettung, Daten, MTL-Einstellungen und Seeds | Mittelwert **und** P95/Front-Back-Fehler prüfen, Körper/Kopf separat |
| 4 | Für Head-Orientierung lokales Kopf-Feature ergänzen | zunächst gleicher Winkel-Head | Fehlen visuelle Details statt Winkelkapazität? |
| 5 | Erst danach Backbone-/Neck-Kapazität variieren | Daten, Geometrie, Heads | Ist die verbleibende Lücke ein Kapazitäts-/Merkmalproblem? |

Für die Head-Ablation ist ein eingefrorener identischer Feature-Extractor ein
günstiger erster kontrollierter Vergleich; anschließend die aussichtsreichen
Varianten mit gleichem Budget gemeinsam feintrainieren. Ein einzelner Seed ist
ein Screening, keine belastbare Überlegenheitsaussage. Neue Ablationen nicht
ständig am endgültigen Holdout auswählen.

Mein aktueller Arbeitsentscheid: **v2-Backbone beibehalten, Geometrie zuerst
diagnostizieren, Orientierungs-Head gezielt vergleichen.** Für Körperorientierung
gibt es bereits ein positives Signal, für reale Kopf-Generalisation noch keinen
passenden Beleg. Der frühere v1-SIM-Wert ist dafür keine faire Zielmarke.
