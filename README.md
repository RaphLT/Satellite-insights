# Satellite Insight: Group [N]

Was selling AI-processed satellite data to investors a viable business for a startup like Orbital Insight?
Four vibe-coded prototypes test the question with real data and free tools. Start with `CLAUDE.md`.

**Website:** `site/` (open `site/index.html` locally). It is published on GitHub Pages from the `gh-pages` branch, which holds only the `site/` folder. To update the live site: `python3 site/build_appendix.py`, commit, `git push`, then `git subtree push --prefix site origin gh-pages`.

1. How a data signal loses value as more funds buy it: `prototypes/1_signal_decay/index.html`
2. Can a student rebuild Orbital Insight's car counter? `prototypes/2_car_counter/results_page.html`
3. Training our own car counter from scratch: `prototypes/3_own_model/results_page.html`
4. Auchan Englos car counts, 2004–2025 (in progress): `prototypes/4_englos_timeseries/`

Prompt log and limits: `docs/APPENDIX_prompt_log_and_limits.md`
