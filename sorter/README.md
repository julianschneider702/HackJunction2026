# Stein-Sortierer

Findet markierte (beschädigte) Steine und sortiert sie: **gelb = gut, blau = defekt**.
Studio muss laufen. Alle Befehle mit Studios Python:

```
P=~/hackjunction_2026/physical-ai-studio/application/backend/.venv/bin/python
```

## Neu: Policy greift und legt ab, das Skript prüft (`policy-sort`)
**bad = gelb, gut = blau.** Sitzung in der Studio-Web-UI schließen, dann:
```
policy-sort            # echter Lauf
policy-sort --manual   # nur Prüfablauf testen: Stein von Hand in den Greifer, ENTER, nichts wird abgelegt
policy-sort --window   # zusätzlich das Debug-Fenster mit den drei Kameras (dort LEERTASTE / q / ESC)
```
Ohne `--window` gibt es kein Fenster: das Log läuft im Terminal, der Rest steht im Dashboard.
`ENTER` Pause/weiter · `q` + `ENTER` Stop · `Ctrl+C` Stop (zweimal = Not-Halt).
1. Policy greift mit `put the brick in the yellow bowl`. Sobald der Stein angehoben ist, hält das Skript die Policy an.
2. Wrist-Kamera und Anomalie-Modell (`~/bricks-data/model_anomalydino.ckpt`): bad → gelb.
3. Sonst: Stein vor die Webcam, Handgelenk dreht durch alle Winkel. Das zweite Anomalie-Modell
   (`config.json` → `anomaly.webcam_model`, bis dahin der Strich-Detektor) prüft jeden Winkel. Danach fährt der Arm
   zurück an die Stelle, an der die Policy angehalten wurde.
4. Policy übernimmt wieder mit `put the brick in the yellow/blue bowl` und legt ab.

Jedes geprüfte Bild landet mit Score im Namen in `sorter/log/policy_<zeit>/` (gut als Trainingsbilder),
die Ergebnisse in `sorter/log/policy_<zeit>.csv`. Grenzen und Zeiten: `config.json` → `policy_sort`, `anomaly`.
Die Schalen müssen dort stehen, wo sie in den Trainings-Episoden standen: die Policy findet sie nicht woanders.

**Dashboard:** Läuft das Team-Dashboard (`frontend/hub.py`, http://127.0.0.1:8090), meldet `policy-sort` jeden Stein
dorthin: Entscheidung, Schale, Zykluszeit, Score der entscheidenden Kamera, Prüfbild und Heatmap, dazu die Phase
und die Dauer jeder Anomalie-Prüfung. Das Dashboard ist eine reine Anzeige ohne Knöpfe; Pause und Stop gehen über das
Terminal (oder das `--window`-Fenster). Ohne Dashboard läuft alles wie vorher.
Einstellungen: `config.json` → `dashboard`.

## Ablauf pro Stein (älteres `sort_bricks.py`: Skript legt selbst ab)
1. **Webcam schaut in die Box** nach einem Stein mit Strich oder Punkt.
   Gezielt greifen geht erst, wenn das Modell mit Texten wie `pick up the red brick` trainiert ist
   (dann `direct_pick_task` in `config.json` setzen). Bis dahin wird der Fund nur angezeigt.
2. **Modell greift** einen Stein. Sobald der Greifer zu ist, übernimmt das Skript:
   anheben, Transit-Pose, Wrist-Kamera prüft *ist ein Stein drin?* und *ist ein Strich zu sehen?*
3. **Kein Strich:** Stein vor die Webcam (Prüf-Pose), das Handgelenk dreht einmal durch alle Winkel,
   die Webcam sucht den Strich.
4. **Ablegen:** Die RealSense verfolgt beide Schalen live. Die Ablage-Pose wird aus der aktuellen
   Schalenposition berechnet, die Schalen dürfen also auch während des Laufs verschoben werden.

## 1. Erkennung testen (bewegt nichts)
In der Studio-Web-UI die Umgebung offen lassen, dann:
```
$P ~/hackjunction_2026/sorter/sort_bricks.py --test
```
Zeigt live die Schalen, die Box und jeden erkannten Strich (rot umrandet).

## 2. Kalibrieren (einmal, ca. 10 min)
Sitzung in der Web-UI **schließen**, dann:
```
$P ~/hackjunction_2026/sorter/calibrate.py
```
Den Follower bewegt ihr mit dem Leader-Arm. Tasten:

| Taste | was |
|---|---|
| `t` | **Transit-Pose**: hoch über allem, Greifer **zu und leer**. Speichert auch das Leer-Referenzbild |
| `k` | in der Transit-Pose **mit Stein** im Greifer: Halte-Probe |
| `i` | **Prüf-Pose**: Stein direkt vor der Webcam (ist schon mit der gemerkten Pose vorbelegt) |
| `h` | Home (optional, Standard = Startpose der Trainings-Episoden) |
| `y` / `b` | Mitte der gelben / blauen Schale merken (Arm aus dem Bild!) |
| `Leertaste` | danach: Greifer über die Schalenmitte fahren und speichern |
| `u` | letzten Ablagepunkt löschen |
| `r` | Webcam-Bereich, in dem der gehaltene Stein in der Prüf-Pose erscheint |
| `m` | Webcam-Bereich der Box (für Schritt 1) |
| `q` | Ende (alles wird sofort gespeichert) |

**Ablagepunkte:** 6–9 Stück. Die Schale jedes Mal woanders hinstellen und alle Stellen abdecken,
an denen die Schalen stehen können. Außerhalb dieses Bereichs legt das Skript **nicht** ab, es wartet.

## 3. Sortieren
Sitzung in der Web-UI geschlossen:
```
$P ~/hackjunction_2026/sorter/sort_bricks.py
```
`Leertaste` Pause/weiter · `q` Stop (ein gehaltener Stein kommt zurück in die Box) · `ESC` Not-Halt.
Ergebnisse: `sorter/log/run_*.csv`, Bilder jedes gefundenen Strichs in `sorter/log/`.

## Sicherheit
- Alle Bewegungen bleiben im Gelenkbereich der 400 Trainings-Episoden und der kalibrierten Posen.
- Erreicht der Arm ein Ziel nicht (blockiert), hält er an und das Skript pausiert.
- "Greifer leer" führt immer zum Öffnen **über der Box**. Liegt das Skript falsch, fällt der Stein zurück.
- Bricht die Verbindung ab, hält der Follower seine Stellung (Studio schaltet sein Drehmoment nicht ab).

## Einstellen
`config.json` (Geschwindigkeiten, Drehwinkel der Prüfung, Greiferwerte) und unter `"vision"`
die Grenzen der Strich-Erkennung (siehe `vision.py`, `DEFAULTS`).
Die Strich-Erkennung ist an 9 echten Markierungen und 129 Bildern ohne Markierung eingestellt.
`test_vision.py` prüft sie erneut, wenn neue Beispielbilder in `~/bricks-data` liegen.
