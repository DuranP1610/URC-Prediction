"""Model B: learn what Elo misses. Compare against Elo on the untouched 2025-26 season.

python model_b.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import TEST_SEASON, TUNE_SEASONS, score

DATA = Path("data")
SMALL = ["elo_diff", "cross_continent", "altitude"]
ALL = SMALL + ["form_diff", "home_attack", "home_defence", "away_attack", "away_defence",
               "home_changed_continent", "away_changed_continent", "playoff", "rest_diff"]

MODELS = {
    "Linear (Elo + travel)": (SMALL, LinearRegression()),
    "Ridge (all features)": (ALL, make_pipeline(StandardScaler(), Ridge(alpha=10))),
    "Gradient boosting (all)": (ALL, GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
}


def load():
    f = pd.read_csv(DATA / "features.csv", parse_dates=["date"])
    b = pd.read_csv(DATA / "elo_backtest.csv", parse_dates=["date"])[["date", "home", "away", "pred_margin"]]
    d = f.merge(b.rename(columns={"pred_margin": "elo_pred"}), on=["date", "home", "away"])
    d = d.dropna(subset=ALL)
    return d[d.season.isin(TUNE_SEASONS)], d[d.season == TEST_SEASON]


def evaluate(pred, actual, sigma):
    h = pd.DataFrame({"pred_margin": pred, "actual_margin": actual.values})
    return score(h, sigma)


def main():
    train, test = load()
    print(f"train: {len(train)} matches ({', '.join(TUNE_SEASONS)})   test: {len(test)} matches ({TEST_SEASON})\n")

    # Model A for reference: same test games, same metrics
    elo_sigma = (train.margin - train.elo_pred).std()
    report = {"Elo (Model A)": evaluate(test.elo_pred.values, test.margin, elo_sigma)}

    for name, (cols, model) in MODELS.items():
        # sigma from out-of-fold errors, so a model can't look more certain than it is
        oof = cross_val_predict(model, train[cols], train.margin, cv=KFold(5))
        sigma = (train.margin - oof).std()
        model.fit(train[cols], train.margin)
        report[name] = evaluate(model.predict(test[cols]), test.margin, sigma)

        if name.startswith("Linear"):
            coefs = dict(zip(cols, model.coef_.round(2)))
            print(f"{name}: margin = {model.intercept_:.1f} + " +
                  " + ".join(f"{c}*{n}" for n, c in coefs.items()) + "\n")

    print(f"Test season {TEST_SEASON}:")
    print(pd.DataFrame(report).T.round(3).to_string())


if __name__ == "__main__":
    main()