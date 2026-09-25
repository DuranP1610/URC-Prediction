# URC Rugby Predictor

Predicting United Rugby Championship match outcomes with a points-based Elo model and an ML model that learns what Elo misses.

**Status:** Elo (Model A) and ML comparison (Model B) done. Live 2026/27 predictions in progress.

## Key finding

**Home advantage in the URC isn't a constant. Most of it comes from travel and altitude.**

Model B's fitted equation:

```
margin = 5.7 + 1.08*elo_diff + 3.09*cross_continent + 3.37*altitude
```

| Match type | Home advantage |
|---|---|
| European v European | ~6 points |
| SA team hosting a European team at the coast | ~9 points |
| European team at Loftus / Ellis Park | ~12 points |

## Results (test season 2025/26, 151 matches, never seen during training)

| Model | Accuracy | Brier | Log loss | Margin MAE |
|---|---|---|---|---|
| Home team always wins (baseline) | 0.701 | 0.202 | 0.618 | 13.57 |
| Elo, default settings | 0.701 | 0.182 | 0.568 | 12.76 |
| Elo, tuned (Model A) | 0.715 | 0.172 | 0.545 | 12.39 |
| **Linear: Elo + travel + altitude (Model B)** | 0.708 | **0.167** | **0.529** | **12.05** |
| Ridge, all 12 features | 0.750 | 0.167 | 0.530 | 12.05 |
| Gradient boosting, all 12 features | 0.743 | 0.171 | 0.539 | 12.65 |

- Home teams win 70% of URC matches, so accuracy barely separates the models. Brier score and log loss (the quality of the probabilities) are the metrics that matter.
- Travel and altitude improve on Elo across every probability metric.
- The 3-feature linear model matches Ridge with 12 features. Form, attack/defence and rest days add nothing beyond Elo.
- Gradient boosting overfits with about 450 training matches.

## Method

### Model A: points-based Elo
- Each team has one rating in points (+5 = 5 points better than an average URC team)
- Predicted margin = home rating + home advantage − away rating
- Win probability = normal CDF of the predicted margin (σ = 14.1)
- Update after each match: rating += K × (actual margin − predicted margin), capped at ±30
- This is online stochastic gradient descent on squared margin error, with K as the learning rate
- Ratings shrink toward average between seasons (carry = 0.7)
- Tuned by grid search: K = 0.10, home advantage = 6, carry = 0.7

### Model B: learning what Elo misses
- Features built from information available before kickoff only (rolling windows use `shift(1)` to prevent leakage)
- Correlating features with Elo's *error* showed that travel and altitude explain what Elo misses, while form does not
- Models compared on the same test season with the same metrics; the simplest model tied for best

### Data split
| Season | Role |
|---|---|
| 2021/22 | Warm-up (Elo ratings settle) |
| 2022/23 – 2024/25 | Train / tune |
| 2025/26 | Test (held out) |
| 2026/27 | Live predictions, logged before kickoff |

### Data
753 URC matches (2021–2026), scraped from Wikipedia season pages. Dates are validated against each season's window, and team names are normalised.

## Project structure

| File | Purpose |
|---|---|
| `elo.py` | Rating maths: predict, win probability, update,

## Usage

```bash
pip install -r requirements.txt

python scrape_history.py      # fetch historical results
python backtest.py            # tune + evaluate Elo
python features.py            # build features
python model_b.py             # compare models

python urc.py predict <round>
python urc.py result <round> <home> <home_score> <away> <away_score>
python urc.py ratings
python urc.py history
```

## Roadmap
- [x] Elo model, tuned and backtested
- [x] Historical data (753 matches)
- [x] Feature engineering + ML comparison
- [ ] Model B live predictions alongside Elo
- [ ] Bootstrap significance test (Model B vs Elo)
- [ ] Monte Carlo season simulation (playoff probabilities)
- [ ] Rating-over-time charts
- [ ] Live 2026/27 results vs bookmakers