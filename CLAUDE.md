# Satellite Insight: Group 7 project, SKEMA Lille

## What this project is
A student project for the course *AI for Business & Society* (SKEMA Business School, Lille).
**Research question:** Was selling AI-processed satellite data to investors a viable business for a startup like
**Orbital Insight**, and what would a new entrant need to do differently?
- Case: Orbital Insight (Palo Alto, 2013; sold to Privateer in May 2024). It counted cars in retailers' parking lots from satellite images and sold the counts to hedge funds.
- Main article: Katona, Painter, Patatoukas & Zeng (2025), *On the capital market consequences of big data: Evidence from outer space*, JFQA 60(2), 551–579.
- Our note so far: `docs/GroupN_S7_note.pdf`. The professor's brief: `docs/professor_vibecoding_ideas.txt`.

Our thesis: the signal was worth most while few funds had it. It lost value as it spread, so the exclusive-data business could not carry ~$130M of venture capital (2023 revenue was ~$9.5M). A new entrant needs customers who do not need exclusivity.

## The official prototype: "Car Counter from Above" (question A)
`site/tool/` (moved from `prototypes/0_car_counter_tool/` so it is published with the site) is a static web app (open `index.html` through any web server, e.g. `python -m http.server`; opening it as a plain file will not load the models).
- Runs the AI **in the browser** with onnxruntime-web (wasm, single thread). Engine files are hosted next to the page.
- Two models, stored as `.wasm` files only so the hosting would accept them (they are ONNX): `own-model.wasm` (YOLO11n trained by us on COWC; input 640, works at 0.10 m/px, conf 0.25) and `dota-n-obb-model.wasm` (YOLO11n-OBB from Ultralytics/DOTA; input 1024, 0.07 m/px, conf 0.25, sharpening recommended).
- Features: upload or sample photo (IGN Englos 2018/2021/2025), metres-per-pixel input, draw a zone or use the "Auchan car parks" preset, count, two-photo % change, scale self-check using median detection size (expected ~5.2 m for ours, ~4.5 m for ready-made).
- `ref_pipeline.py` is the Python copy of the exact browser method (it expects the ONNX files under `w/`). Browser counts were within ~5% of it: 2021 Auchan zone 166 vs 168; 2025 zone 875 vs 896.
- Live at https://raphlt.github.io/Satellite-insights/tool/ (also published earlier as a claude.ai artifact).

The professor's rules (all must hold): 1) public link, no login; 2) uses our case data; 3) prompt log appendix with every prompt, tool and date (`docs/APPENDIX_prompt_log_and_limits.md`; **keep adding prompts**); 4) one honest limits paragraph including who did what (team: Raphael Le Theo, Dinis Ramalhao, Dorian Barrier, Gabin Routier; who did what still to fill in); 5) the prototype appears in the three-page note and on slide 9. The user handles rule 5 themselves.
Only ONE prototype counts. Question A is on the professor's list. Prototypes 1–3 are supporting work; the time series (4) is NOT on the list and needs his written yes.

## Results (corrected 18 Nov after a colour-order bug; do not change numbers without re-running)
**Bug note:** Ultralytics `predict()` treats numpy images as BGR. Our early code passed RGB, which degraded every Prototype 2/3 test. Fixed in `carcount.py` (`crop[..., ::-1]`). The ONNX/browser path feeds RGB directly and was never affected.

**P1 Signal decay** (toy model; inputs tagged REAL or ASSUMPTION). REAL inputs: 4.76% three-day return (Table 3); price of "hundreds of thousands" (p. 556); $130M raised and $9.5M expected 2023 revenue (Poulson 2023). At the reported price, the fast-decay curves imply 7–13 clients and $2–6M of revenue.

**P2 Ready-made medium model (YOLO11m-OBB, DOTA)**: COWC 2_12 (113 hand-marked cars), no zoom: 117 counted, 106 right, 11 false alarms, 7 missed. IGN Englos 2025 photo: 2,335 cars (×5 zoom + sharpening). Hand-check area (x 1300–1700, y 700–1000; ~103 cars): ~93 found, ~2 false alarms, ~10 missed.

**P3 Our own model (YOLO11n, 515 COWC cars, 30 epochs, ~1 h CPU)**: validation 6_8 picks conf 0.55. Test 2_12: 106 counted, 102 right, 4 false alarms, 11 missed (90% found, 96% right), matching the ready-made model on Potsdam. Englos box: 69 at 0.55 (too strict for IGN) and about 95 at 0.25.

**Browser tool (ONNX pipeline, default settings)**: ours on COWC 2_12: 124 counted, 108 right, 16 false alarms, 5 missed; Englos box 95. Ready-made nano: 114/104/10/9 on COWC; Englos box 89 with sharpening. It undercounts badly on full IGN zones (304 vs ours 875 in the 2025 Auchan zone).

**P4 Englos over time (unfinished, not on the professor's list)**: IGN years for this area: 2004, 2009, 2012, 2015, 2018, 2021, 2025 (2012 still to download; link pattern in `docs/` notes and below). The partial counts were made before the bug fix and must be re-run. Candidate flight dates: 2015: 11 Jun Thu / 30 Jun Tue / 1 Jul Wed / 10 Jul Fri. 2018: 8 May Tue (holiday) / 29 Jun Fri / 26 Jul Thu. 2021: 1 Jun Tue / 7 Jun Mon / 9 Jun Wed / 13 Jun Sun. 2025: 20 May Tue. The browser tool already shows the key point: 2021 vs 2025 in the Auchan zone gives +427%, most likely because of the day each photo was flown.
2012 image: `https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:4326&BBOX=50.62586,2.96105,50.63214,2.97095&WIDTH=2576&HEIGHT=2576&FORMAT=image/jpeg&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS2012`

## Website (live)
- https://raphlt.github.io/Satellite-insights/ , repo https://github.com/RaphLT/Satellite-insights
- `site/`: static HTML. Home puts the tool first, then supporting pages 1–4, conclusion, appendix. Images in `site/img/`; simulator JS `site/assets/signal-decay.js` (copied unchanged from P1).
- `site/car-counter.html` and `site/own-model.html` are converted from the P2/P3 results pages (base64 images swapped for files in `site/img/`); if those pages change, update the site copies.
- `site/appendix.html` is generated: `python3 site/build_appendix.py` (from the prompt log + `site/appendix_template.html`). Do not edit it by hand.
- Publish: commit, `git push`, then `git subtree push --prefix site origin gh-pages`. No GitHub Actions: the user's token cannot push workflow files.
- P4 on the site shows only the post-fix tool counts (2021: 166, 2025: 875 in the Auchan zone). Re-run P4 only if the professor approves it.

## Folder map
- `prototypes/1_signal_decay/`: `index.html` (interactive, Chart.js), `signal_decay_simulation.ipynb`, `build.py` (generates the notebook), figures.
- `site/tool/`: the official prototype (see above).
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
