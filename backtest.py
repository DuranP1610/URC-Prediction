import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import elo

HISTORY = Path("data/history.csv")
BURN_IN = "2021-22"                                 # ratings start at 0, not scored
TUNE_SEASONS = ["2022-23", "2023-24", "2024-25"]    
TEST_SEASON = "2025-26"                             

GRID = {
    "k": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "hfa": [0, 2, 3, 4, 5, 6, 8],
    "carry": [0.3, 0.5, 0.7, 0.85, 1.0],           
}


def run(matches, params, carry):
    """Replay every season in order. Between seasons ratings shrink toward 0 by `carry`."""
    ratings, rows = {}, []
    for season, games in matches.groupby("season", sort=True):
        ratings = {t: carry * v for t, v in ratings.items()}
        ratings, hist = elo.replay(games.itertuples(), ratings, params)
        for h in hist:
            h["season"] = season
        rows += hist
    return pd.DataFrame(rows)


def score(h, sigma):
    """How good were the pre-match predictions?"""
    outcome = np.sign(h.actual_margin)                      # 1 home win, 0 draw, -1 away win
    y = (outcome + 1) / 2                                   # 1 / 0.5 / 0
    p = h.pred_margin.apply(lambda m: elo.win_prob(m, sigma)).clip(1e-6, 1 - 1e-6)
    decided = outcome != 0
    return {
        "matches": len(h),
        "accuracy": (np.sign(h.pred_margin[decided]) == outcome[decided]).mean(),
        "brier": ((p - y) ** 2).mean(),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
        "margin_mae": (h.actual_margin - h.pred_margin).abs().mean(),
    }


def baseline(h):
    """Knows nothing about teams: home side always wins by the average home margin."""
    b = h.copy()
    b["pred_margin"] = h.actual_margin.mean()
    return b


def main():
    matches = pd.read_csv(HISTORY).sort_values("date")
    print(f"{len(matches)} matches, seasons: {sorted(matches.season.unique())}\n")

    # 1. grid search on the tuning seasons, judged on margin error
    results = []
    for k, hfa, carry in itertools.product(*GRID.values()):
        h = run(matches, {"k": k, "hfa": hfa}, carry)
        tune = h[h.season.isin(TUNE_SEASONS)]
        results.append({"k": k, "hfa": hfa, "carry": carry,
                        "margin_mae": (tune.actual_margin - tune.pred_margin).abs().mean()})
    grid = pd.DataFrame(results).sort_values("margin_mae")
    print("Top 5 settings (tuning seasons):")
    print(grid.head().round(3).to_string(index=False))

    best = grid.iloc[0]
    params, carry = {"k": best.k, "hfa": best.hfa}, best.carry

    # 2. sigma = how spread out real margins are around our predictions
    h = run(matches, params, carry)
    tune = h[h.season.isin(TUNE_SEASONS)]
    sigma = (tune.actual_margin - tune.pred_margin).std()
    print(f"\nBest: k={best.k}, hfa={best.hfa}, carry={best.carry}, fitted sigma={sigma:.1f}")

    # 3. final exam on the untouched test season
    test = h[h.season == TEST_SEASON]
    default = run(matches, {}, 0.6)
    report = pd.DataFrame({
        "tuned Elo": score(test, sigma),
        "default Elo": score(default[default.season == TEST_SEASON], elo.SIGMA),
        "home-always baseline": score(baseline(test), sigma),
    }).T
    print(f"\nTest season {TEST_SEASON}:")
    print(report.round(3).to_string())

    h.to_csv("data/elo_backtest.csv", index=False)
    print("\nSaved pre-match predictions to data/elo_backtest.csv")


if __name__ == "__main__":
    main()