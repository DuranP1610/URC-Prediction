# URC Rugby Predictor

Predicting United Rugby Championship match outcomes with a points-based Elo model and an ML model that learns what Elo misses.

**Status:** Elo (Model A) and ML model (Model B) built, evaluated and significance-tested. Both are predicting the 2026/27 season live, with every prediction logged before kickoff.

## Key finding

**Home advantage in the URC isn't a constant. Most of it comes from travel and altitude.**

Model B's fitted equation (trained 2022–25):

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
- Gradient boosting overfits with about 450 training matches.

### Is the improvement real?

Paired bootstrap (10,000 resamples of the 151 test matches):

| Metric | Improvement (Elo − Model B) | 95% CI | P(Model B better) |
|---|---|---|---|
| Log loss | 0.0165 | [0.0028, 0.0299] | 99% |
| Brier | 0.0057 | [0.0001, 0.0113] | 98% |
| Margin error | 0.34 pts | [−0.01, 0.69] | 97% |

Model B's improvement is modest but unlikely to be luck. The clearest gain is in log loss, which suggests B mainly avoids being confidently wrong in travel and altitude games.

**Caveat:** the feature selection looked at data that included the test season, and the final model was picked after seeing the test scores. The live 2026/27 season is therefore the true out-of-sample confirmation.

### Choosing Model B

Linear and Ridge tied on the test season. **Linear was chosen for simplicity and interpretability:**
- 3 features instead of 12, so less room to overfit noise
- Every coefficient reads directly in points ("altitude is worth +3.4")
- Ridge's extra 9 features (form, attack/defence, rest days) added nothing, which is evidence they carry no signal beyond Elo
- Ridge's higher accuracy was about 5 extra correct picks out of 151, within noise

For live use, the chosen model was retrained on 2022–2026 (including the test season), and its specification is frozen for the whole 2026/27 season.

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
- Correlating features with Elo's *error* showed that travel and altitude explain what Elo misses (Elo underrated home teams by about 6.5 points at altitude and 4.3 points against travelling teams), while form does not
- All models were compared on the same test season with the same metrics
- Each model's σ comes from out-of-fold errors, so no model looks more certain than it is

### Data split
| Season | Role |
|---|---|
| 2021/22 | Warm-up (Elo ratings settle) |
| 2022/23 – 2024/25 | Train / tune |
| 2025/26 | Test (held out), then added to training for live use |
| 2026/27 | Live predictions, logged before kickoff |

### Data
753 URC matches (2021–2026), scraped from Wikipedia season pages. Dates are validated against each season's window, and team names are normalised.

## Live 2026/27 season

Every round, both models' predictions are written to `data/predictions_log.csv` and committed **before kickoff**. The git timestamps prove the predictions weren't made after the results were known. Results are entered after each round, updating the Elo ratings and Model B's inputs.

## Project structure

| File | Purpose |
|---|---|
| `elo.py` | Rating maths: predict, win probability, update, replay |
| `urc.py` | Live CLI: Elo + Model B predictions, results entry, ratings |
| `scrape_history.py` | Scrapes 5 seasons of results → `data/history.csv` |
| `backtest.py` | Replays history, grid-searches Elo settings, scores on the test season |
| `features.py` | Builds the leakage-free feature table → `data/features.csv` |
| `model_b.py` | Trains and compares ML models against Elo |
| `bootstrap.py` | Paired bootstrap significance test: Model B vs Elo |

## Usage

```bash
pip install -r requirements.txt

# historical pipeline
python scrape_history.py      # fetch historical results
python backtest.py            # tune + evaluate Elo
python features.py            # build features
python model_b.py             # compare models
python bootstrap.py           # significance test

# weekly live loop
python urc.py predict <round>                                          # before kickoff
python urc.py result <round> <home> <home_score> <away> <away_score>   # after the games
python urc.py ratings
python urc.py history
```

## Roadmap
- [x] Elo model, tuned and backtested
- [x] Historical data (753 matches)
- [x] Feature engineering + ML comparison
- [x] Live predictions: Elo + Model B side by side
- [x] Bootstrap significance test (Model B vs Elo)
- [ ] Monte Carlo season simulation (playoff probabilities)
- [ ] Findings notebook with charts
- [ ] Live 2026/27 results vs bookmakers