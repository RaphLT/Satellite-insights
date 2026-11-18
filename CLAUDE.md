# Satellite Insight: Group [N] project, SKEMA Lille

## What this project is
A student project for the course *AI for Business & Society* (SKEMA Business School, Lille).
**Research question:** Was selling AI-processed satellite data to investors a viable business for a startup like
**Orbital Insight**, and what would a new entrant need to do differently?
- Case: Orbital Insight (Palo Alto, 2013; sold to Privateer in May 2024). It counted cars in retailers' parking lots from satellite images and sold the counts to hedge funds.
- Main article: Katona, Painter, Patatoukas & Zeng (2025), *On the capital market consequences of big data: Evidence from outer space*, JFQA 60(2), 551–579.
- Our note so far: `docs/GroupN_S7_note.pdf`. The professor's brief: `docs/professor_vibecoding_ideas.txt`.

Our thesis: the signal was worth most while few funds had it. It lost value as it spread, so the exclusive-data business could not carry ~$130M of venture capital (2023 revenue was ~$9.5M). A new entrant needs customers who do not need exclusivity.

## Goal now: build a website
One public website, opening without a login, that presents the four prototypes as a single story for the professor and classmates.
It must meet the professor's vibe-coding rules:
1. **It runs**: a public link that opens in a browser without a login. GitHub Pages, Netlify or Vercel are fine; a static site is enough.
2. **It uses our case**: Orbital Insight, the article, Auchan Englos. No demo data.
3. **Prompt log in an appendix**: every prompt, in order, with tool and date. See `docs/APPENDIX_prompt_log_and_limits.md`. **Keep adding prompts from Claude Code to that log.**
4. **Honest about limits**: say what works, what doesn't, what data each part was tested on, and who in the team did what (team: Raphael Le Theo, Dinis Ramalhao, Dorian Barrier and Gabin Routier; who did what still to fill in).
5. It appears in the three-page note and on slide 9. The user handles this rule themselves.

Suggested site structure (confirm with the user):
- Home: the question, the thesis in two sentences, and a four-card path through the prototypes.
- 1. **How a data signal loses value**: the interactive simulator (`prototypes/1_signal_decay/index.html` is self-contained; reuse its JS).
- 2. **Can a student rebuild the car counter?** (question A, ready-made model): `prototypes/2_car_counter/results_page.html`.
- 3. **Training our own model**: `prototypes/3_own_model/results_page.html`.
- 4. **Englos over time** (unfinished, see below).
- Conclusion: what a new entrant should do.
- Appendix: prompt log + limits + sources.
Pages 2 and 3 embed their images as base64; for the site, prefer real image files (they are in the same folders).

## Results so far (all checked; do not change numbers without re-running)
**P1 Signal decay** (toy model; inputs tagged REAL or ASSUMPTION)
- REAL: 4.76% three-day return when exclusive (article, Table 3); price of "hundreds of thousands" $/yr (article, p. 556); $130M raised and $9.5M expected 2023 revenue (Poulson 2023).
- At the reported price, the three fast-decay curves imply 7–13 clients and $2–6M of vendor revenue; the slow curve never reaches that price. Best case under any curve is far below what $130M of funding needs.

**P2 Ready-made detector** (YOLO11m-OBB trained on DOTA, no training by us)
- COWC Potsdam tile 2_12 (113 cars marked by hand): 98 counted, 92 right, 6 false alarms, 21 missed (81% found, 94% right). Zoom ×1.5.
- IGN Auchan Englos 2025 photo (~0.27 m/px): 2,217 cars counted. Needs zoom ×5 plus sharpening; sharpening *hurts* on COWC.
- Hand check of one area (pixels x 1300–1700, y 700–1000): ~103 cars, 89 found, 1 false alarm, 14 missed.

**P3 Our own model** (YOLO11n from COCO, trained on 515 COWC cars, 30 epochs, ~1 h CPU)
- Train tiles 2_10, 2_11, 2_13, 3_10; validation 6_8 (188 cars, used to pick conf = 0.20); test 2_12.
- Test 2_12: 110 counted, 73 right, 37 false alarms, 40 missed (65% found, 66% right), against 81% / 94% for the ready-made model (DOTA: 2,806 images, 188,282 labelled objects).
- Englos hand-check area: about 87 of ~100 found, 2 false alarms, roughly equal to the ready-made model.
- Lesson: the architecture is free and labels make the difference, but public labelled datasets have eroded that moat too.

**P4 Englos over time** (UNFINISHED)
- IGN yearly photos for this area: 2004, 2009, 2012, 2015, 2018, 2021, 2025. We have every year except 2012 (the 2012 download was a duplicate of 2025). Get it from
  `https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:4326&BBOX=50.62586,2.96105,50.63214,2.97095&WIDTH=2576&HEIGHT=2576&FORMAT=image/jpeg&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS2012`
- Counts done with the ready-made model: 2004 = 0 cars, 2009 = 24 (both 50 cm/px, too coarse, while many cars are visible by eye). 2015, 2018, 2021 and 2025 still to run:
  `cd prototypes/4_englos_timeseries && python count_years.py dota` (about 7 min per image on CPU; GPU much faster), then optionally `python count_years.py own ../3_own_model/own_cowc_yolo11n.pt 0.2` as a cross-check.
- Candidate flight dates (from IGN's mosaic layer; IGN does not say which day applies where): 2015: 11 Jun Thu, 30 Jun Tue, 1 Jul Wed, 10 Jul Fri. 2018: 8 May Tue (**public holiday**), 29 Jun Fri, 26 Jul Thu. 2021: 1 Jun Tue, 7 Jun Mon, 9 Jun Wed, 13 Jun **Sun** (store closed). 2025: 20 May Tue.
- "Auchan zone" = the two rectangles in `ZONES` in count_years.py (Auchan's own car parks). The full image is the whole retail park, roads included.
- The point to make: one snapshot every ~3 years, taken on unknown days, cannot tell an investor how Auchan's quarter went. Orbital Insight bought near-daily paid imagery for that.

## Folder map
- `site/`: the website (static HTML; images in `site/img/`, simulator JS in `site/assets/signal-decay.js`, copied unchanged from P1). `site/appendix.html` is generated from the prompt log by `site/build_appendix.py` + `appendix_template.html`; do not edit it by hand. Deployed by `.github/workflows/pages.yml`. Once P4 counts are run, update the table in `site/englos.html`.
- `prototypes/1_signal_decay/`: `index.html` (interactive, Chart.js), `signal_decay_simulation.ipynb`, `build.py` (generates the notebook), figures.
- `prototypes/2_car_counter/`: `carcount.py` (**shared library**: tiling, detection, de-duplication, COWC label reading, matching), `run_eval.py`, Colab notebook, results page and images.
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
