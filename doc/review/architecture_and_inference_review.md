# Review: PedRecNet, EHPI3D und Inferenz-Pipeline

Stand: Oktober 2026. Ursprünglich reines Review; den Umsetzungsstand zeigt der folgende Abschnitt. Bewertet wird der Code in
`pedrec/networks`, `pedrec/utils/torch_utils`, `pedrec/tracking`, `pedrec/utils/pose_deconv_helper.py`,
`pedrec/utils/ehpi_helper.py` und die Trainingsprozedur in `pedrec/training`. Referenzpunkt sind die in 2024 bis 2026
etablierten Standards für 2D/3D-Pose, Orientierung, skelettbasierte Aktionserkennung, Detektion und Tracking.

## Umsetzungsstand (Branch `v2`)

Dieser Branch enthält genau eine Konfiguration (keine Varianten oder Schalter für alte Komponenten). Die Punkte 1, 3
und 5 stammen aus dem Hauptzweig, die übrigen brauchen ein neues Training.

| Prio | Maßnahme | Status |
| --- | --- | --- |
| 1 | `inference_mode`, ein PedRecNet-Batch pro Frame, gebatchte Crops und Rücktransformation | umgesetzt (`pedrec/inference`) |
| 2 | Orientierung als Biternion (cos, sin) mit Kosinus-Loss, Konfidenz mit `BCEWithLogitsLoss` | umgesetzt (`pedrec_orientation_head_shared.py`, `loss_functions.py`) |
| 3 | Best-Checkpoint, EMA, AMP, Gradient-Clipping, Resume, NaN-Schutz | umgesetzt; dazu Kendall-Gewichtung mit begrenzten log-Varianzen, Half-Body-, Farb- und Erasing-Augmentation, Datensatz-Balancing |
| 4 | `torch.compile`, `channels_last`, fp16, GPU-Preprocessing | umgesetzt; ONNX entfernt, weil der RT-DETR-Export mit aktuellem torch / transformers nicht verlustfrei funktioniert |
| 5 | ByteTrack, One-Euro-Filter | umgesetzt, einziger Tracker |
| 6 | Detektor RT-DETRv2 (`PekingU/rtdetr_v2_r18vd`, NMS-frei) | umgesetzt, YoloV4 entfernt; seitenverhältnistreue Eingabe (640 x 352 bei 16:9, ~28 statt 33.8 GMACs von YoloV4) |
| 7 | Konfidenz aus Heatmap-Statistik | umgesetzt (`pedrec_pose_conf_head_heatmap.py`) |
| 8 | Backbone und UDP | HGNetV2-B3 (ImageNet SSLD) + FPN-Neck statt ResNet-50 + Deconvs: 2.45 statt 6.28 GMACs pro Crop, gleiche 64 x 48 Heatmaps; UDP als einzige Konvention; Trainingskette auf zwei Stages ab ImageNet verkürzt |
| 9 | ST-GCN statt ResNet-50 auf dem EHPI-Bild | umgesetzt (`ehpi_stgcn.py`, 0.2 Mio. Parameter, 116 statt 167 MMACs pro Sequenz, gleiche EHPI-Daten) |
| 10 | Lifting über die Zeit | umgesetzt: kausaler zeitlicher Lifter pro Track (`pose_lifter.py`, 27 Frames, 0.7 Mio. Parameter, ~1 ms für alle Tracks); SMPL bewusst nicht (eigenes Modell, kein Mehrwert für die Pipeline) |

Die Trainingsdaten (Dataframes, Bilder, Annotationen) werden unverändert verwendet. Die v2-Gewichte müssen trainiert
werden (`mise run download:models`, dann `mise run train:all`); die veröffentlichten v1-Gewichte passen nicht zu v2.

Laufzeit (CPU, 4 Threads, Zufallsgewichte, 1920 x 1080, 6 Personen, Median pro Frame): v1 1129 ms, v2 648 ms
(Detektion 509 -> 362, PedRecNet 544 -> 249, Aktionen 51 -> 30, Lifting 5 ms). GPU-Zahlen fehlen noch
(`mise run bench --fast` auf beiden Branches). Die Genauigkeit von v2 ist erst nach dem Training messbar; die
Lifter-Validierung meldet direkt, ob die gelifteten 3D-Posen besser sind als die Einzelbild-Posen.

## 1. Zusammenfassung

Die Architektur ist ein solides Multi-Task-Design auf Basis von "Simple Baselines" (ResNet-50 + Deconv-Heads, 2018),
erweitert um Integral-Regression (Soft-Argmax), einen Tiefen-Head, einen Orientierungs-Head und einen
Konfidenz-Head. Das Design ist in sich konsistent und reproduzierbar, aber in mehreren Punkten vom heutigen
Stand abgehängt:

| Bereich | Status | Größtes Verbesserungspotential |
| --- | --- | --- |
| Backbone (ResNet-50) | 2018-Standard | ViTPose / RTMPose (CSPNeXt) / HRNet: +5 bis +10 AP auf COCO bei gleicher Eingabegröße |
| 2D-Head (Heatmap + Soft-Argmax, L1) | ok, bekannte Bias-Probleme | SimCC- oder RLE-Head, UDP-Datenverarbeitung |
| 3D-Head (2D-Heatmap x Tiefenkarte) | ok, ohne Kamerawissen | Kamera-/Skalen-Normierung, Lifting- oder Body-Model-Ansätze |
| Orientierungs-Head (Soft-Argmax über 360 Bins) | konzeptionell problematisch am 0°/360°-Übergang | (cos, sin)-Regression oder zirkuläre Klassifikation (MEBOW-Stil) |
| Konfidenz-Head (FC über 42240 Werte) | funktional, teuer und starr | Konfidenz aus Heatmap-Statistik oder RLE-Sigma |
| MTL-Loss (Unsicherheitsgewichtung) | abweichende Formel | Originalformulierung oder einfache feste Gewichte |
| Training | kein AMP/EMA/DDP, schwache Augmentierung | AMP, EMA, Cutout/Halbkörper-Aug, Checkpoint-Auswahl per Validierung |
| EHPI3D (ResNet-50 auf 32x32-Pseudobild) | überdimensioniert, nicht graphbasiert | ST-GCN/CTR-GCN/PoseC3D oder kleiner TCN |
| Detektor (YOLOv4, CPU-NMS) | veraltet | YOLOv8/11 oder RT-DETR, GPU-NMS |
| Tracking (LK-Optical-Flow + Heuristiken) | fragil, Debug-Ausgaben im Hot-Path | ByteTrack/OC-SORT mit Kalman-Filter |
| Inferenz | kein `inference_mode`, CPU-Preprocessing, doppelter PedRecNet-Lauf | siehe Abschnitt 9, realistisch 2x bis 3x Durchsatz |

## 2. Architekturüberblick (Ist-Zustand)

```
Bild (256x192, ImageNet-Norm)
  └─ ResNetHeadless (ResNet-50, 2048x8x6)
       ├─ PedRecConvTransposeBase: 2 gemeinsame Deconv-Stufen (256 ch, 32x24) + je eine eigene Deconv-Stufe pro Kopf (64x48)
       │    ├─ PedRecPose2DHead:  1x1 Conv -> 26 Heatmaps -> SoftArgmax2d -> (x, y) in [0, 1]
       │    └─ PedRecPose3DHead:  1x1 Conv -> 26 Heatmaps + 26 Tiefenkarten -> SoftArgmax2d + DepthRegression -> (x, 1-y, z)
       ├─ PedRecPoseConfHead:     concat(softmax2d, softmax3d) [52x64x48] -> 2 Conv -> FC(42240, 128) -> FC(128, 26) -> Sigmoid
       └─ PedRecOrientationsHead: avgpool(2048) ++ ConvTranspose1d-Features des (detached) 3D-Skeletts (768)
            ├─ Body: Linear -> 180 Theta-Bins / 360 Phi-Bins -> SoftArgmax1d
            └─ Head: Linear -> 180 Theta-Bins / 360 Phi-Bins -> SoftArgmax1d
Loss: PedRecNetLossHead = L1(2D) + L1(3D) + Winkelabstand(Orientierung) + BCE(Konfidenz), gewichtet über lernbare Sigmas
```

Inferenz: YOLOv4 (608x320) -> Person-Crops per `cv2.warpAffine` -> PedRecNet (gebatcht über Personen) -> Tracking
(Lucas-Kanade auf Gelenkpunkten + Merger) -> EHPI-Pseudobild aus 32 bzw. 64 Frames -> EHPI3D (ResNet-50) -> Sigmoid
Multi-Label.

## 3. Backbone und Feature-Extraktor

- `ResNetHeadless` ist eine Kopie der torchvision-ResNet-Implementierung ohne Klassifikationskopf. Das ist in Ordnung,
  könnte aber durch `torchvision.models.resnet50(weights=...)` plus `nn.Sequential(*list(children())[:-2])` oder
  `torchvision.models.feature_extraction.create_feature_extractor` ersetzt werden. Damit entfällt der Umweg über die
  pose-resnet-Gewichte (`initialize_pose_resnet`) für die ImageNet-Initialisierung.
- Output-Stride 32 (8x6 Feature-Map bei 256x192) ist für Pose eher grob; die drei Deconv-Stufen gleichen das nur
  teilweise aus. HRNet (Multi-Resolution), ViTPose (ViT-B mit MAE-Vortraining, Patch 16, einfacher Deconv-Decoder)
  und RTMPose (CSPNeXt + SimCC) liefern auf COCO bei gleichem Input mehrere AP-Punkte mehr; ViTPose-B liegt bei
  ca. 75.8 AP gegenüber ca. 70.4 AP für SimpleBaseline-R50 bei 256x192.
- Für die Zielanwendung (Fußgänger aus Fahrzeugperspektive, Echtzeit) ist RTMPose-m/l der pragmatische Kandidat: ähnlicher
  Rechenaufwand wie R50, deutlich bessere Genauigkeit, auf Mobil-/Edge-Hardware optimiert. ViTPose ist genauer, aber
  teurer.
- Empfehlung (groß): Backbone austauschbar machen (Config-Feld statt `PedRecNet50Config`-Vererbung) und RTMPose/ViTPose
  als Alternative anbieten. Die Heads bleiben bei entsprechendem Feature-Kanal unverändert.

## 4. 2D-Pose-Head

- Integral-Regression (Soft-Argmax über die Softmax der Heatmap) ist korrekt umgesetzt (`SoftArgmax2d`). Bekannte
  Schwächen: (a) bei diffusen Heatmaps zieht der Erwartungswert zur Bildmitte, (b) die Temperatur (`norm_val = 1.0`
  auf rohen Logits) wird nicht gelernt, (c) ohne Heatmap-Supervision kann die Heatmap "entarten" (mehrere Moden).
  Gängige Abhilfen: zusätzlicher Heatmap-Loss (Integral + Heatmap, wie in "Integral Human Pose Regression"), oder
  gleich SimCC (getrennte 1D-Klassifikation für x und y, sub-pixel Auflösung, kein Deconv nötig), oder RLE
  (Residual Log-Likelihood) für reine Koordinatenregression.
- Der Loss ist ein maskierter L1 auf normalisierten Koordinaten. Das ist für Integral-Regression üblich; RLE würde die
  Unsicherheit pro Gelenk gleich mitliefern und damit den separaten Konfidenz-Head weitgehend überflüssig machen.
- Datenverarbeitung: Die Affin-Transformationen (`get_affine_transform`, Flip-Test mit hartkodierter Bildbreite 1000
  bzw. 1980 in den Export-Skripten) entsprechen dem SimpleBaseline-Original und tragen den dort bekannten
  Quantisierungs-/Flip-Bias (UDP, Huang et al. 2020: ca. +1 bis +2 AP durch "unbiased data processing").
- 1x1-`final_conv_kernel` und `padding=1 if kernel == 3 else 0` sind korrekt, Kernel 3 würde minimal glätten.

## 5. 3D-Pose-Head

- Konzept: eigene Heatmap pro Gelenk (x, y) plus Tiefenkarte; Tiefe = Erwartungswert von `sigmoid(depth_map)` unter der
  Softmax der 3D-Heatmap (`DepthRegression`). Das ist eine kompakte Variante der volumetrischen Integral-Regression
  (Sun et al. 2018) ohne das volle 3D-Volumen und spart Speicher. Die Koordinaten sind hüftnormiert im Würfel
  `skeleton_3d_range = 3000 mm`, der Wert ist in `pose_deconv_helper.py`, `ehpi_helper.py` und den Datasets jeweils
  hartkodiert dupliziert (3000 / 1500).
- `pose_3d_coords[:, :, 1] = 1 - pose_3d_coords[:, :, 1]` dreht die y-Achse in-place nach dem Soft-Argmax. Das
  funktioniert, ist aber eine stille Konvention, die sich nur aus dem Dataset-Code ergibt; ein Kommentar oder eine
  explizite Achsen-Definition in `skeleton_pedrec.py` wäre angebracht.
- Es fließt kein Kamerawissen ein (keine Intrinsics, keine Bounding-Box-Skala als Zusatzeingabe). Damit ist die
  absolute Skala der 3D-Pose nur aus dem Crop ableitbar. Aktuelle Ansätze geben die Crop-Skala/Fokallänge mit (z. B.
  CLIFF, "camera-aware" Regression) oder arbeiten mit einem Körpermodell (SMPL: HMR2.0 / 4D-Humans, SMPLer-X), das
  Längen-Priors mitbringt. Für die EHPI3D-Weiterverarbeitung wird ohnehin ein Einheitsskelett gebaut
  (`get_unit_skeleton_sequence`), der Skalenfehler wird dort also neutralisiert, nicht aber für die Positionsschätzung.
- Alternative mit geringem Aufwand: 2D-zu-3D-Lifting über die Zeit (VideoPose3D, MotionBERT, PoseFormerV2) auf den
  getrackten 2D-Skeletten. Das würde die zeitliche Konsistenz erhöhen, die im Moment nur ein 2-Frame-Mittel liefert.
- Der 3D-Head hat eine eigene Deconv-Stufe (zweiter `deconv_head`), also laufen für 2D und 3D zwei 64x48-Branches mit
  je 256 Kanälen. Ein gemeinsamer Branch mit 2x26 Ausgabekanälen würde Rechenzeit sparen, ohne dass in der Literatur ein
  Genauigkeitsverlust dokumentiert ist.

## 6. Orientierungs-Head

- Phi (Azimut, 0 bis 2π) wird über 360 Bins und Soft-Argmax als Erwartungswert bestimmt. Der Erwartungswert über eine
  lineare Achse ist für eine zirkuläre Größe nicht definiert: eine bimodale Verteilung bei 5° und 355° liefert 180°.
  Der Loss (`AngularErrorLoss`) ist zwar zirkulär (`min(2π - d, d)`), das Netz kann aber nahe der Sprungstelle nur mit
  einer sehr scharfen Verteilung korrekt antworten. Das dürfte ein Teil der in `doc/diss_eval/orientation_*` sichtbaren
  Ausreißer erklären.
  Standardlösungen: (a) (cos φ, sin φ)-Regression mit L2/Cosine-Loss und `atan2` (Biternion-Netze; der Code enthält
  bereits einen ungenutzten `AngularErrorCartesianCoordinatesLoss`), (b) zirkuläre Klassifikation mit Gauß-geglätteten
  Labels und zirkulärem Erwartungswert über den Einheitskreis (MEBOW: 72 Bins), (c) von-Mises-Likelihood.
- Theta (Elevation, 0 bis π) ist linear, dort ist Soft-Argmax unproblematisch.
- Die Pose-Features entstehen über `ConvTranspose1d` entlang der Gelenkachse (3 -> 6 -> 12 -> 24 Kanäle, Länge 26 -> 32,
  flach 768). Die Gelenkreihenfolge im Vektor ist eine willkürliche Enumeration ohne Nachbarschaftssemantik, eine
  1D-Faltung darüber hat keinen strukturellen Vorteil gegenüber einem MLP. Ein kleiner GCN über die Skelett-Kanten oder
  ein MLP wäre einfacher und erklärbarer. Die 768 sind hartkodiert (`pose_size`).
- Die 3D-Pose wird mit `detach()` eingespeist, der Orientierungs-Loss beeinflusst also die Pose nicht. Das ist
  vertretbar; eine Variante mit durchgeleitetem Gradienten wäre ein lohnendes Ablations-Experiment.
- Kopf-Orientierung teilt denselben Eingabevektor; sinnvoll wäre eine Kopf-spezifische Feature-Auswahl (z. B.
  RoI-Pooling um die Kopfgelenke), da der globale Avg-Pool die Kopfregion bei 8x6 Features kaum auflöst.

## 7. Konfidenz-Head

- Eingabe sind die Softmax-Karten beider Pose-Köpfe (52x64x48), gefolgt von zwei 3x3-Convs ohne Padding, Max-Pool und
  `Linear(42240, 128)`. Die 42240 koppeln den Head fest an 256x192 Eingaben (5.4 M Parameter nur in dieser Schicht).
  Ein `AdaptiveAvgPool2d` oder ein Global-Pool pro Gelenkkanal würde die Kopplung lösen und Parameter sparen.
- `self.pose_conf = nn.Linear(2048, num_joints)` wird nie verwendet (toter Parameter, taucht aber im State-Dict und im
  Optimizer auf).
- `JointConfLoss` nutzt `BCELoss` auf Sigmoid-Ausgaben; `BCEWithLogitsLoss` ist numerisch stabiler und AMP-tauglich.
- Fachlich: Die Konfidenz lernt "ist das Gelenk sichtbar/korrekt", also eine Funktion der Heatmap-Form. Dieselbe
  Information liefern (a) der Maximalwert/die Entropie der Softmax-Karte ohne Zusatzparameter oder (b) die RLE-Sigma.
  Das offene TODO in `doc/todos.md` ("Korrelation gelernte Konfidenz vs. Softmax-Maximum") sollte vor einer
  Weiterentwicklung beantwortet werden; wenn die Korrelation hoch ist, kann der Head entfallen.

## 8. Multi-Task-Loss und Training

- `PedRecNetLossHead` implementiert eine Variante der Unsicherheitsgewichtung (Kendall, Gal, Cipolla 2018):
  `Σ 1/(2σ_i²)·L_i + log(1 + Π σ_i²)`. Das Original verwendet `Σ log σ_i` (bzw. `log σ_i²`) als Regularisierer; das
  Produkt plus 1 ändert die Gradienten der Sigmas qualitativ (gekoppelte statt unabhängige Regularisierung, und für
  kleine Sigmas verschwindet der Regularisierer statt negativ zu werden). Für die Konfidenz wird zudem `1/σ²` statt
  `1/(2σ²)` genutzt (für Klassifikation korrekt nach Kendall, im Code aber nicht kommentiert). Empfehlung: entweder die
  Originalform (`log σ_i²` pro Task, numerisch als `log_var`-Parameter) oder feste, per Validierung gewählte Gewichte;
  Alternativen wie GradNorm/PCGrad bringen bei vier eng verwandten Tasks erfahrungsgemäß wenig.
- `torch.isnan(loss)`-Prüfungen überspringen Tasks ohne Labels im Batch. Funktional, aber die Loss-Module geben dafür
  einen NaN-Tensor zurück, der jedes Mal ein `.item()`-artiges Sync auslöst. Ein Maskierungs-Ansatz mit
  `torch.where`/Zählern wäre sauberer und AMP-kompatibel.
- Trainingsprozedur (in `train_pedrec.py` übernommen): AdamW + OneCycle, Runde 1 mit eingefrorenem Backbone, Runde 2
  komplett, Batch 48, 10 + 5 Epochen. Was fehlt im Vergleich zu aktuellen Setups: Mixed Precision (bf16/fp16 mit
  `GradScaler`), EMA der Gewichte, Gradient-Clipping, Auswahl des besten Checkpoints anhand der Validierung (aktuell
  wird der letzte gespeichert), Speichern von Optimizer-/Scheduler-Zustand zum Fortsetzen, DistributedDataParallel,
  `channels_last`, `torch.compile`. `set_fixed_seeds` setzt den NumPy-Seed nicht (auskommentiert), Augmentierungen in
  den Datasets sind damit nicht reproduzierbar.
- Augmentierung: Flip, Skalierung 0.25, Rotation 30° (0 mit MEBOW-Labels). Stand der Technik für Top-down-Pose:
  zusätzlich Halbkörper-Crops, Random-Erasing/Cutout, photometrische Augmentierung, stärkere Skalierung, und
  rotations-invariante Orientierungslabels (Rotation um die optische Achse ändert Phi nicht, sondern nur die
  Bildlage; die Rotation könnte mit Label-Korrektur statt mit 0 betrieben werden).
- Datenmischung: COCO, Human3.6m (systematisch 1/10), SIM; es gibt kein Sampling-Balancing zwischen den Quellen, SIM
  dominiert je nach Subsampling. Ein gewichteter Sampler pro Datensatz ist eine kleine, wirksame Änderung.
- Validierung läuft nach jeder Epoche über alle Sets mit Batch 48 und 12 Workern; bei den 15 Epochen pro Stage ist
  das vertretbar.

## 9. Inferenz-Pipeline

Beobachtungen in `pose_deconv_helper.py`, `yolo_v4_helper.py`, `pipeline.py` (früher `demo_actionrec_dev.py`):

1. Kein `torch.no_grad()`/`torch.inference_mode()` um die Detektor- und PedRecNet-Forwards. Es wird nur `.detach()`
   auf den Ausgaben genutzt, der Autograd-Graph wird trotzdem aufgebaut (Speicher und ca. 10 bis 20 % Zeit).
2. Preprocessing auf der CPU: pro Person `cv2.warpAffine`, dann `ToTensor` + `Normalize` je Crop, anschließend
   `torch.stack`; YOLO-Eingabe wird als float32 auf der CPU normiert und dann hochgeladen. Besser: uint8 hochladen,
   Normierung auf der GPU, Crops über `torchvision.ops.roi_align` oder `kornia.geometry.warp_affine` gebatcht.
3. Die Rücktransformation der 2D-Koordinaten läuft in einer Python-Schleife pro Person
   (`affine_transform_coords_2d`); ein `torch.bmm` über den Batch ersetzt die Schleife.
4. Zwei PedRecNet-Durchläufe pro Frame: erst auf den YOLO-Boxen, dann `do_redetect_pose_recognition` für getrackte,
   aber nicht detektierte Personen. Beide Mengen lassen sich in einem Batch verarbeiten.
5. Detektor: YOLOv4 mit CPU-NMS in NumPy (`bb_nms_cpu`) und `torch.autograd.Variable` (seit PyTorch 0.4 obsolet).
   `torchvision.ops.batched_nms` auf der GPU ist ein Einzeiler. Ein Wechsel auf YOLOv8/11-n/s oder RT-DETR halbiert
   die Detektorzeit und verbessert kleine Fußgänger deutlich; da nur die Klasse "person" gebraucht wird, reicht ein
   person-only Modell.
6. Tracking: Lucas-Kanade-Optical-Flow auf den 26 Gelenkpunkten plus heuristischer Merger (Farbmittelwert der Box,
   Pose-Ähnlichkeit) ohne Bewegungsmodell. `human_merger.py` enthält `print`-Ausgaben im Hot-Path. ByteTrack/OC-SORT
   (Kalman-Filter + IoU-Zuordnung in zwei Konfidenzstufen) sind robuster, schneller und Standard; mit OKS-basierter
   Zuordnung der Skelette lässt sich das Pose-Tracking direkt integrieren.
7. Glättung: 2-Frame-Mittel von 3D-Pose und Orientierung, das die Historie in-place verändert. Ein One-Euro-Filter pro
   Gelenk oder ein Kalman-Filter liefert bei gleicher Latenz deutlich ruhigere Ausgaben.
8. EHPI-Aufbau: Pro Frame und Person wird die komplette 32/64-Frame-Historie neu normiert (Einheitsskelett, Umordnung,
   uint8-Quantisierung). Ein Ringpuffer mit inkrementeller Aktualisierung reduziert das auf O(1) pro Frame. Die
   uint8-Quantisierung (Bereich ±20 Einheiten auf 255 Stufen) verwirft Präzision ohne Not; float16/float32-Eingaben
   wären für das Netz gleich teuer.
9. Kein Export-Pfad (ONNX, TensorRT, TorchScript), obwohl `.onnx`/`.engine` bereits in `.gitignore` stehen. Mit
   `torch.export`/ONNX plus TensorRT fp16 sind auf einer Desktop-GPU typischerweise 2x bis 4x gegenüber eager
   PyTorch fp32 erreichbar; `torch.compile` und `channels_last` bringen ohne Export bereits 1.3x bis 1.8x.
10. Die Pipeline ist seriell (Capture -> Detektor -> Pose -> Aktion) in einem Thread. Capture/Decode kann in einen
    eigenen Thread (es existiert ein ungenutzter `WebcamProviderAsync`), Detektor und Pose könnten auf getrennten
    CUDA-Streams überlappen (Detektor für Frame t+1 während Pose für Frame t).
11. Numerik: Phi kann exakt 2π werden (Soft-Argmax über 360 Bins inkl. letztem Bin), `eval_tud_orientation.py`
    behandelt das mit `min(..., 360)`; ein `fmod` an der Quelle wäre sauberer.

## 10. EHPI3D (Aktionserkennung)

- Eingabe: Skelettsequenz als Pseudobild (Gelenke x Frames x xyz als RGB, uint8), Netz: komplettes ResNet-50
  (23 M Parameter) auf 32x32 bzw. 64x32 Pixeln, Ausgabe: Multi-Label-Sigmoid über 20 Klassen mit BCE.
- Ein ImageNet-ResNet-50 auf einem 32x32-Bild ist stark überdimensioniert; die ersten Stufen (7x7-Conv Stride 2 +
  MaxPool) reduzieren das Pseudobild sofort auf 8x8. Bei EHPI-artigen Encodings sind kleine Netze (ResNet-18-Varianten
  ohne Stem-Downsampling oder 1D-TCNs) in der Regel gleich gut. Stand der Technik für skelettbasierte Aktionserkennung
  sind Graph-Netze (ST-GCN, 2s-AGCN, CTR-GCN, InfoGCN), Heatmap-Volumen (PoseC3D) und Skelett-Transformer
  (Hyperformer, SkateFormer); sie nutzen die Gelenktopologie explizit, die im EHPI-Bild nur über die Zeilenreihenfolge
  (`SKELETON_PEDREC_TO_PEDRECEHPI3D`) implizit vorliegt.
- Zeitliche Auflösung ist fest an 30 fps gekoppelt (`frame_sampling = 2` für 15 fps als eigene Variante). Eine
  Resampling-Stufe auf eine feste Ziel-Framerate würde Varianten überflüssig machen und die Demo robust gegen andere
  Videoquellen machen.
- Mischung von GT- und vorhergesagten Skeletten im Training (`gt_result_ratio`) ist sinnvoll (Domain-Gap); zusätzlich
  üblich: Gelenk-Dropout, zeitliche Verzerrung, Rotation um die Vertikale, Skalenrauschen.
- Multi-Label-Schwellen (0.7 in der Demo, 0.8 in der Evaluation) sind nicht pro Klasse kalibriert; eine
  Schwellenkalibrierung auf dem Validierungsset (oder Temperatur-Skalierung) ist ein kleiner Schritt mit sichtbarem
  Effekt auf OF1/CF1.
- `ehpi_transform` normiert mit datensatzspezifischem Mittel/Std (0.400/0.443/0.401), die im Code hartkodiert sind und
  sich bei anderem Skelett-Encoding ändern müssten.

## 11. Codequalität rund um die Netze (nicht umgesetzt)

- Ungenutzter oder doppelter Code: `SpatialSoftArgmax2d`, `AngularErrorCartesianCoordinatesLoss`, `Pose2DL2Loss`,
  `Pose3DL2Loss`, `EnvPositionL2Loss`, `pedrec_net_2d_only.py`, `pedrec_pose_conf_head_simple.py`,
  `PedRecPoseConfHead.pose_conf`, `WebcamProviderAsync`, `get_ehpi2d_from_human_history`.
- Hartkodierte Konstanten: 42240 und 768 (Head-Dimensionen), 3000/1500 (3D-Bereich) in drei Modulen, 1000/1980
  (Flip-Test-Bildbreiten in den Export-Skripten), Mittel/Std des EHPI-Transforms.
- Debug-Reste: `print` in `human_merger.py`, `raise("WTF")` in `eval_tud_orientation.py`, `a = 1`-Breakpoint-Zeilen in
  mehreren Evaluationsskripten, `exit(-1)` in `do_detect`.
- Keine Unit-Tests für die Transformationen (Affin-Transform, Flip der 3D-Gelenke, Orientierungs-Flip, Soft-Argmax).
  Genau dort entstehen erfahrungsgemäß die stillen Fehler; eine Handvoll Tests mit synthetischen Heatmaps und bekannten
  Winkeln wäre günstig.
- Die Loss-Masken verlassen sich auf feste Spaltenindizes (`target[:, :, 3]`, `[:, :, 4]`); benannte Konstanten oder
  ein kleines Label-Dataclass würden die Konventionen aus dem README im Code sichtbar machen.

## 12. Priorisierte Empfehlungen

| Prio | Aufwand | Maßnahme | Erwarteter Effekt |
| --- | --- | --- | --- |
| 1 | klein | `torch.inference_mode()` um alle Forwards, Redetektion in denselben Batch, GPU-NMS, Koordinaten-Rücktransformation gebatcht | 1.3x bis 1.8x Inferenzdurchsatz ohne Genauigkeitsänderung |
| 2 | klein | Orientierungsausgabe als (cos, sin) bzw. zirkuläre Klassifikation; `BCEWithLogitsLoss` | weniger Ausreißer an 0°/360°, AMP-tauglich |
| 3 | klein | Bestes-Checkpoint-Auswahl, EMA, AMP, NumPy-Seed, Gradient-Clipping im Training | stabileres Training, meist +0.5 bis +1 AP |
| 4 | mittel | ONNX/TensorRT-Export oder `torch.compile` + `channels_last`; GPU-Preprocessing der Crops | 2x bis 4x Durchsatz |
| 5 | mittel | ByteTrack/OC-SORT statt LK-Optical-Flow + Merger; One-Euro-Filter statt 2-Frame-Mittel | stabilere IDs, ruhigere 3D-Pose, weniger Doppel-Personen |
| 6 | mittel | Detektor auf YOLOv8/11 oder RT-DETR (person-only) | bessere Erkennung kleiner/verdeckter Fußgänger, kürzere Detektorzeit |
| 7 | mittel | Konfidenz aus Heatmap-Statistik oder RLE statt FC-Head; gemeinsamer Deconv-Branch für 2D/3D | weniger Parameter, schnellere Heads |
| 8 | groß | Backbone/Head auf RTMPose oder ViTPose; UDP-Datenverarbeitung | +5 bis +10 AP 2D, entsprechend bessere 3D-Basis |
| 9 | groß | EHPI3D durch GCN/Transformer über die Skelettsequenz ersetzen, Resampling auf feste Framerate | genauer und ca. 10x kleiner als ResNet-50 |
| 10 | groß | Kamera-/Skalenbewusste 3D-Schätzung oder Lifting über die Zeit, optional SMPL-Körpermodell | konsistentere absolute 3D-Pose, bessere Positionsschätzung |

Die Punkte 1 bis 3 sind ohne Änderung der Gewichte bzw. mit einem Fine-Tuning der betroffenen Heads umsetzbar; ab
Punkt 6 sind neue Trainings der Stage-Kette nötig (siehe `pedrec/training/experiments/pedrec_stages.py`).
