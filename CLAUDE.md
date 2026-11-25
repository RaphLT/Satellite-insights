# Satellite Insight: Group 8 project, SKEMA Lille

## What this project is
A student project for the course *AI for Business & Society* (SKEMA Business School, Lille).
**Research question:** Was selling AI-processed satellite data to investors a viable business for a startup like
**Orbital Insight**, and what would a new entrant need to do differently?
- Case: Orbital Insight (Palo Alto, 2013; sold to Privateer in May 2024). It counted cars in retailers' parking lots from satellite images and sold the counts to hedge funds.
- Main article: Katona, Painter, Patatoukas & Zeng (2025), *On the capital market consequences of big data: Evidence from outer space*, JFQA 60(2), 551–579.
- Our note so far: `docs/GroupN_S7_note.pdf`. The professor's brief: `docs/professor_vibecoding_ideas.txt`.

Our thesis: the signal was worth most while few funds had it. It lost value as it spread, so the exclusive-data business could not carry ~$130M of venture capital (2023 revenue was ~$9.5M). A new entrant needs customers who do not need exclusivity.

## The official prototype: "Car Counter from Above" (question A)
`site/` (moved from `prototypes/0_car_counter_tool/`) is a static web app (open `index.html` through any web server, e.g. `python -m http.server`; opening it as a plain file will not load the models).
- Runs the AI **in the browser** with onnxruntime-web (wasm, single thread). Engine files are hosted next to the page.
- Two models, stored as `.wasm` files only so the hosting would accept them (they are ONNX): `own-model.wasm` (YOLO11n trained by us on COWC; input 640, works at 0.10 m/px, conf 0.25) and `dota-n-obb-model.wasm` (YOLO11n-OBB from Ultralytics/DOTA; input 1024, 0.07 m/px, conf 0.25, sharpening recommended).
- Features: upload or sample photo (IGN Englos 2018/2021/2025), metres-per-pixel input, draw a zone or use the "Auchan car parks" preset, count, two-photo % change, scale self-check using median detection size (expected ~5.2 m for ours, ~4.5 m for ready-made).
- `ref_pipeline.py` is the Python copy of the exact browser method (it expects the ONNX files under `w/`). Browser counts were within ~5% of it: 2021 Auchan zone 166 vs 168; 2025 zone 875 vs 896.
- Live at https://raphlt.github.io/Satellite-insights/ (the old /tool/ address redirects there).

The professor's rules (all must hold): 1) public link, no login; 2) uses our case data; 3) prompt log appendix with every prompt, tool and date (`docs/APPENDIX_prompt_log_and_limits.md`; **keep adding prompts**); 4) one honest limits paragraph including who did what (team: Raphael Le Theo, Dinis Ramalhao, Dorian Barrier, Gabin Routier; who did what still to fill in); 5) the prototype appears in the three-page note and on slide 9. The user handles rule 5 themselves.
Only ONE prototype counts. Question A is on the professor's list. Prototypes 1–3 are supporting work; the time series (4) is NOT on the list and needs his written yes.

## Prototype v2: "Wazemmes Parking from Above" (Session 9)
**Question it answers:** can free aerial photos plus an open AI model measure how full a city district's public parking is, and could paid weekly satellite photos turn that into a service for many towns?
Also published as a claude.ai artifact: https://claude.ai/artifact/9CRQXJrTDybZfxNWyUNBRP (private until the user shares it).

What the page does (all in the browser, nothing uploaded). The retraining, three-tests, business-case and sources sections were removed from the page at the user's request (their numbers stay in this file and in the note):
- Map of Wazemmes (Lille, about 1 km²) on 3 IGN photo dates (latest, 2021 market day, 2018), every street coloured by estimated occupancy.
- **Live map (GitHub site only):** draw a square up to 500 m anywhere in France. The page downloads the IGN photo for that square (latest, 2021 or 2018) at 0.20 m/px and the OpenStreetMap parking (Overpass), clips both to the square, and counts the free spaces. Claude.ai artifacts block downloads from other sites, so there it shows a note instead.
- **Upload tool:** load any aerial photo, or one of 12 hand-checked Wazemmes squares with OpenStreetMap parking pre-drawn, and set metres per pixel. Then press "Count free spaces":
  - With no marks, it finds rows of parked cars automatically (cars linked within 14 m along their row's direction, rows extended up to 25 m and joined across gaps up to 30 m). Car-sized gaps count as free spaces, minus the cars the model misses.
  - Users can also draw rows or car-park outlines by hand.
  - Output: free spaces with a range, and a table per street or row.
- Remaining sections: how it works step by step; why city streets are harder than retail car parks.
- Removed from the live page on 25 Nov at the user's request: the retraining results (confusion matrices), the three tests (other dates, satellite sharpness, real change), the weekly-service business case, and the "Method, limits and sources" footer. The numbers stay below and in `prototypes/5_parking_v2/results/`; the claude.ai artifact and `satellite-insight 3/prototypes/5_parking_v2/web/` still have the full page.
- **Latest version at `site/v3/`** (live at https://raphlt.github.io/Satellite-insights/v3/, added 25 Nov): adds the live map and automatic row finding. `site/v2/` is kept unchanged at its old address; the old car counter page now links to v3.
  - Live map test: a 273 m square of Wazemmes gave ≈426 free of 647 spaces (34% full) in a stand-in test. If the IGN photo is blocked (CORS error), try the WMTS tiles or tell the user; do not add a proxy without asking. If Overpass is busy (HTTP 429/504), the page falls back to finding parking automatically.

Method (same model as v1, `own-model.wasm` = YOLO11n trained on COWC; the "new" part is the method):
1. Resize the photo to 0.10 m/px and cut it into 640 px tiles with a 96 px overlap.
2. Keep detections with confidence ≥ **0.10** (chosen by 3-fold cross-validation, seed 0; v1 used 0.25). Merge marks within 1.5 m.
3. Keep only cars inside parking: within 7 m of an OpenStreetMap street centre line tagged `parking:*`, inside a public `amenity=parking` polygon, or within 3 m of a row the user draws.
4. Spaces: kerb length ÷ 5.5 m (parallel), 3.0 m (angled), 2.5 m (perpendicular), minus 10 m per street; car parks use `capacity` or area ÷ 25 m².
5. Corrected cars = seen × precision ÷ recall (0.895 ÷ 0.819) ÷ (1 − tree share; tree share from the IGN infrared photo, NDVI > 0.25).
6. Free = spaces − corrected cars. Range from the 3 photo dates: correction factor 1.06 (2021) to 1.22 (2018).

Key results (do not change without re-running; full tables in `prototypes/5_parking_v2/results/results_retrain_dates.md`):
- **Cross-validation:** street and car-park cars on the latest photo, 6 hand-labelled squares, 432 cars:

  | Run | Found | False | Missed | Precision | Recall |
  |---|---|---|---|---|---|
  | Original model, cut-off 0.10 (**winner**) | 331 | 39 | 73 | 89.5% | 81.9% |
  | Retrained: Englos + Wazemmes ×3 | 289 | 40 | 115 | 87.8% | 71.5% |
  | Gentle retraining: frozen, lr 0.002 | 313 | 85 | 91 | 78.6% | 77.5% |
  | Retrained: Englos only | 292 | 121 | 112 | 70.7% | 72.3% |
  | Retrained: Wazemmes only | 279 | 63 | 125 | 81.6% | 69.1% |
  | Old cut-off 0.25 | 258 | 16 | 146 | 94.2% | 63.9% |

  Lesson: about 300 local cars are not enough to retrain on; a properly chosen cut-off helped more.
- **Other dates**, street and car-park cars, labels corrected from model pre-marks, so recall may be flattering:
  - 2018: precision 96.6%, recall 79.2%.
  - 2021 market day: precision 76.8%, recall 72.5%; 30 false marks were market stalls.
- **Satellite sharpness test** (photos blurred to that resolution): 30 cm gives precision 89.6%, recall 76.7%; 50 cm gives 79.8% and 62.6%. So 30 cm satellites are viable and 50 cm ones are not.
- **Retail car parks vs city streets**, same model, cut-off 0.25, whole picture:
  - Englos: precision 80.6%, recall 90.9% (1,181 cars).
  - Wazemmes: precision 77.1%, recall 63.9% (432 cars).
- **Street occupancy** (4,154 street spaces from OpenStreetMap):
  - latest photo: 33% (≈1,381 cars);
  - 2021: 36%;
  - 2018: 16%. The hand labels confirm about half as many cars in 2018.
- **Business inputs** (Land Info price list, 2026; UP42):
  - New 30 cm Pléiades Neo shots cost $42.50/km², with a 100 km² minimum, so ≈ $4,250 per weekly town (our arithmetic). Archive photos cost $22.50/km².
  - The model counts 1 km² in 56 s on 2 CPU cores.

Where it lives in this repository:
- `site/v2/`: the site (copied from `prototypes/5_parking_v2/web/`), live at https://raphlt.github.io/Satellite-insights/v2/. The old tool's page links to it ("New: count free parking spaces (v2)").

Folder `prototypes/5_parking_v2/`:
- `code/`: Python, in run order:
  - `prep_ign.py` and `train_ign.py`: Englos fine-tune;
  - `eval_waz.py`: test on Wazemmes;
  - `cv.py` and `cv_score.py`: cross-validation recipes;
  - `occupancy.py`: spaces and occupancy;
  - `make_pack.py`: labelling packs for other dates;
  - `eval_dates.py` and `occ_dates.py`: date tests and occupancy by date;
  - `wazemmes_osm.geojson`: Overpass export.
- `labels/`: YOLO-format hand labels; class 0 = car, class 1 = "maybe" that the user confirmed as a car.
- `results/`: JSON outputs and the results table.

The scripts still use workspace paths (`/home/claude/...`); adjust them before re-running. The IGN mosaics (5000×5000 px at 0.20 m/px, bbox 50.6220–50.6310 N, 3.0424–3.0566 E) are not included; links are in the results file.

Limits to keep stating (also in section B of `docs/APPENDIX_prompt_log_and_limits.md`):
- IGN flight days and hours are unknown.
- Capacity ignores driveways and loading bays, so occupancy is a lower bound.
- OpenStreetMap misses some kerbs.
- Test labels were checked by one person.
- The 30 cm test blurs aerial photos, which only approximates a real satellite image.
- A live "free space now" app needs sensors or cameras, not satellites.

## Results (corrected 18 Nov after a colour-order bug; do not change numbers without re-running)
**Bug note:** Ultralytics `predict()` treats numpy images as BGR. Our early code passed RGB, which degraded every Prototype 2/3 test. Fixed in `carcount.py` (`crop[..., ::-1]`). The ONNX/browser path feeds RGB directly and was never affected.

**P1 Signal decay** (toy model; inputs tagged REAL or ASSUMPTION). REAL inputs: 4.76% three-day return (Table 3); price of "hundreds of thousands" (p. 556); $130M raised and $9.5M expected 2023 revenue (Poulson 2023). At the reported price, the fast-decay curves imply 7–13 clients and $2–6M of revenue.

**P2 Ready-made medium model (YOLO11m-OBB, DOTA)**: COWC 2_12 (113 hand-marked cars), no zoom: 117 counted, 106 right, 11 false alarms, 7 missed. IGN Englos 2025 photo: 2,335 cars (×5 zoom + sharpening). Hand-check area (x 1300–1700, y 700–1000; ~103 cars): ~93 found, ~2 false alarms, ~10 missed.

**P3 Our own model (YOLO11n, 515 COWC cars, 30 epochs, ~1 h CPU)**: validation 6_8 picks conf 0.55. Test 2_12: 106 counted, 102 right, 4 false alarms, 11 missed (90% found, 96% right), matching the ready-made model on Potsdam. Englos box: 69 at 0.55 (too strict for IGN) and about 95 at 0.25.

**Browser tool (ONNX pipeline, default settings)**: ours on COWC 2_12: 124 counted, 108 right, 16 false alarms, 5 missed; Englos box 95. Ready-made nano: 114/104/10/9 on COWC; Englos box 89 with sharpening. It undercounts badly on full IGN zones (304 vs ours 875 in the 2025 Auchan zone).

**P4 Englos over time (unfinished, not on the professor's list)**: IGN years for this area: 2004, 2009, 2012, 2015, 2018, 2021, 2025 (2012 still to download; link pattern in `docs/` notes and below). The partial counts were made before the bug fix and must be re-run. Candidate flight dates: 2015: 11 Jun Thu / 30 Jun Tue / 1 Jul Wed / 10 Jul Fri. 2018: 8 May Tue (holiday) / 29 Jun Fri / 26 Jul Thu. 2021: 1 Jun Tue / 7 Jun Mon / 9 Jun Wed / 13 Jun Sun. 2025: 20 May Tue. The browser tool already shows the key point: 2021 vs 2025 in the Auchan zone gives +427%, most likely because of the day each photo was flown.
2012 image: `https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:4326&BBOX=50.62586,2.96105,50.63214,2.97095&WIDTH=2576&HEIGHT=2576&FORMAT=image/jpeg&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS2012`

## Website (live)
- https://raphlt.github.io/Satellite-insights/ is the car counter only (user's choice, 18 Nov).
- https://raphlt.github.io/Satellite-insights/v2/ is prototype v2, Wazemmes Parking from Above (added 25 Nov), served from `site/v2/`.
- https://raphlt.github.io/Satellite-insights/v3/ is the latest v2 page (live map of France and automatic rows), served from `site/v3/`; the old page links to it. Repo: https://github.com/RaphLT/Satellite-insights
- An earlier multi-page site (home, supporting pages, conclusion, appendix page) was removed from `site/`; it is in git history (commit "Merge updated project...") if ever needed.
- The tool links to the prompt log on GitHub (`docs/APPENDIX_prompt_log_and_limits.md`).
- Publish: commit, `git push`, then `git subtree push --prefix site origin gh-pages`. No GitHub Actions: the user's token cannot push workflow files.

## Folder map
- `prototypes/1_signal_decay/`: `index.html` (interactive, Chart.js), `signal_decay_simulation.ipynb`, `build.py` (generates the notebook), figures.
- `site/`: the official prototype, Car Counter from Above (see above).
- `site/v2/`: prototype v2 as first published (25 Nov).
- `site/v3/`: latest prototype v2 page, with the live map (see "Prototype v2").
- `prototypes/5_parking_v2/`: v2 code, hand labels and results (`code/`, `labels/`, `results/`).
- `prototypes/2_car_counter/`: `carcount.py` (**shared library, colour bug fixed**: tiling, detection, de-duplication, COWC label reading, matching), `run_eval.py`, Colab notebook, results page and images.
- `prototypes/3_own_model/`: `train_prep.py` (COWC dots → YOLO dataset), `train.py`, `eval_own.py`, `train_own_model.ipynb`, trained weights `own_cowc_yolo11n.pt`, `results.csv`, results page.
- `prototypes/4_englos_timeseries/`: `count_years.py`, partial results.
- `data/cowc_labels/`: COWC `_Annotated_Cars.png` label files (one red pixel per car, transparent background).
- `data/ign_englos/`: IGN photos by year, 2576 px for 700 m (0.2715 m/px), same box every year.
- COWC photos are not included (~10 MB each). The user has them, or download from
  `https://gdo152.llnl.gov/cowc/download/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/top_potsdam_<tile>_RGB.png`.
- Scripts in P2/P3 still use paths from the original workspace (`data/`, `cowc/`, `w/`); adjust before re-running.

## Rules for working on this project
- Every number on the site must come from a run or a source in this folder; tag inputs REAL or ASSUMPTION as P1 does.
- Never present the toy model's outputs as forecasts. Never call 2004/2009 counts real counts.
- Short quotes from the article only (under 15 words); paraphrase the rest.
- Write in plain English for business students; avoid jargon or explain it.
- Log every user prompt in `docs/APPENDIX_prompt_log_and_limits.md`.

Dependencies: `pip install ultralytics numpy pillow scipy matplotlib nbformat` (Python 3.10+).
