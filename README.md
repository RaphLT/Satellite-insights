# Satellite Insight: Group 8

Was selling AI-processed satellite data to investors a viable business for a startup like Orbital Insight?
Group 8 (AI for Business & Society, SKEMA Lille): Raphael Le Theo, Dinis Ramalhao, Dorian Barrier, Gabin Routier.

The official prototype is **Car Counter from Above** (`site/`, live at https://raphlt.github.io/Satellite-insights/), an in-browser car counter. Prototype v2, **Wazemmes Parking from Above** (free parking spaces from aerial photos), is at https://raphlt.github.io/Satellite-insights/v2/ (`site/v2/`; code and results in `prototypes/5_parking_v2/`).

Supporting work:

1. How a data signal loses value as more funds buy it: `prototypes/1_signal_decay/`
2. Can a student rebuild Orbital Insight's car counter? `prototypes/2_car_counter/`
3. Training our own car counter: `prototypes/3_own_model/`
4. Auchan Englos over time (unfinished): `prototypes/4_englos_timeseries/`

Prompt log and limits: `docs/APPENDIX_prompt_log_and_limits.md`. Start with `CLAUDE.md`.

To update the live site: commit, `git push`, then `git subtree push --prefix site origin gh-pages`.
