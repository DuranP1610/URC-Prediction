# URC Rugby Predictor

Predicting United Rugby Championship matches with a tuned Elo model and an ML model that learns what Elo misses, then simulating the season 10,000 times.

📓 **[Read the full analysis → findings.ipynb](findings.ipynb)**

**Status:** models built, evaluated and significance-tested. Predicting the 2026/27 season live, with every prediction committed to git before kickoff.

## Key finding

**Home advantage in the URC isn't a constant. Most of it comes from travel and altitude.**

| Match type | Home advantage |
|---|---|
| European v European | ~6 points |
| SA team hosting a European team at the coast | ~9 points |
| European team at Loftus / Ellis Park | ~12 points |

Elo assumes one flat home advantage for every match. A 3-feature linear model (Elo + cross-continent travel + altitude) fixes that.

## Results

Held-out 2025/26 season (151 matches, not used for training):

| Model | Accuracy | Brier ↓ | Log loss ↓ |
|---|---|---|---|
| Home team always wins | 0.701 | 0.202 | 0.618 |
| Elo (Model A) | 0.715 | 0.172 | 0.545 |
| **Elo + travel + altitude (Model B)** | 0.708 | **0.167** | **0.529** |

- Home teams win 70% of URC matches, so accuracy barely separates the models. Brier score and log loss (the quality of the probabilities) are what matter.
- **Model B beats Elo in 98–99% of 10,000 bootstrap resamples.** The improvement is modest, but it's unlikely to be luck.
- Linear tied with a 12-feature Ridge model and beat gradient boosting (which overfit). The simplest model was chosen.

## Season simulation

Model B plays out the 2026/27 season 10,000 times (Monte Carlo): random margins around each prediction, URC league points, then the playoffs.

**Pre-season title odds:** Leinster 43%, Bulls 25%, Glasgow 9%, Stormers 7%, Connacht 6%. The odds are updated after every round, and their history is saved in `data/odds_history.csv`.

## How it works

1. **Data:** 753 URC matches (2021–2026) scraped from Wikipedia, with dates validated and team names normalised
2. **Model A (Elo):** one rating per team, in points. Updates after each match are online gradient descent on squared margin error. Settings are tuned by grid search (K = 0.10, home advantage = 6, off-season carry = 0.7)
3. **What Elo misses:** Elo's errors correlate with travel and altitude, not with form
4. **Model B:** linear regression on Elo difference + travel + altitude, trained on 2022–25 and tested on 2025/26
5. **Significance:** paired bootstrap over the test season
6. **Simulation:** vectorised NumPy Monte Carlo of the remaining fixtures and playoffs
7. **Live:** predictions are logged before every round, and the git timestamps prove when they were made

**Caveat:** feature selection looked at data that included the test season, so the live 2026/27 season is the true out-of-sample test.

## Project structure

| File | Purpose |
|---|---|
| `elo.py` | Rating maths: predict, win probability, update, replay |
| `urc.py` | Live CLI: Elo + Model B predictions, results entry, ratings |
| `scrape_history.py` | Scrapes 5 seasons of results → `data/history.csv` |
| `fetch_fixtures.py` | Full 2026/27 fixture list → `data/season_fixtures.csv` |
| `backtest.py` | Replays history, tunes Elo, scores the test season |
| `features.py` | Builds the leakage-free feature table |
| `model_b.py` | Trains and compares ML models against Elo |
| `bootstrap.py` | Paired bootstrap significance test |
| `simulate.py` | Monte Carlo season simulation → title / top-8 odds |
| `findings.ipynb` | The full write-up, with charts |

## Usage

```bash
pip install -r requirements.txt

# rebuild the analysis
python scrape_history.py && python backtest.py && python features.py
python model_b.py && python bootstrap.py

# weekly live loop
python urc.py result <round> <home> <home_score> <away> <away_score>   # after the games
python simulate.py                                                     # updated season odds
python urc.py predict <round>                                          # before kickoff
```

## Roadmap
- [x] Elo model, tuned and backtested
- [x] ML model + bootstrap significance test
- [x] Monte Carlo season simulation
- [x] Findings notebook
- [ ] Automated weekly pipeline (GitHub Actions)
- [ ] Live website ([urc-predictor-web](#), in progress)
- [ ] End-of-season review: Elo vs Model B vs bookmakers
