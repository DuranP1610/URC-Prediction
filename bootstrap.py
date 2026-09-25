"""
Paired bootstrap: resample the test season 10,000 times and see how often B still wins.
python bootstrap.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

import elo
from model_b import SMALL, load

N_BOOT = 10_000
rng = np.random.default_rng(42)          # fixed seed -> same answer every run


def per_match_losses(pred_margin, actual, sigma):
    """Brier, log loss and margin error for EACH match (not averaged)."""
    y = (np.sign(actual) + 1) / 2                                   # 1 / 0.5 / 0
    p = np.clip([elo.win_prob(m, sigma) for m in pred_margin], 1e-6, 1 - 1e-6)
    return {
        "brier": (p - y) ** 2,
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "margin_error": np.abs(actual - pred_margin),
    }


def main():
    train, test = load()
    actual = test.margin.values

    # Model A: Elo (same sigma rule as model_b.py)
    elo_sigma = (train.margin - train.elo_pred).std()
    elo_loss = per_match_losses(test.elo_pred.values, actual, elo_sigma)

    # Model B: linear on Elo diff + travel + altitude, trained on 2022-25 only
    model = LinearRegression()
    oof = cross_val_predict(model, train[SMALL], train.margin, cv=KFold(5))
    b_sigma = (train.margin - oof).std()
    model.fit(train[SMALL], train.margin)
    b_loss = per_match_losses(model.predict(test[SMALL]), actual, b_sigma)

    n = len(test)
    idx = rng.integers(0, n, size=(N_BOOT, n))     # each row = one resampled "season"

    rows = {}
    for metric in elo_loss:
        diff = elo_loss[metric] - b_loss[metric]   # positive = Model B better on that match
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows[metric] = {
            "elo": elo_loss[metric].mean(),
            "model_b": b_loss[metric].mean(),
            "improvement": diff.mean(),
            "95% CI low": lo,
            "95% CI high": hi,
            "P(B better)": (boot > 0).mean(),
        }

    print(f"Paired bootstrap, {N_BOOT:,} resamples of {n} test matches\n")
    print(pd.DataFrame(rows).T.round(4).to_string())
    print("\nCI entirely above 0 -> the improvement is unlikely to be luck.")


if __name__ == "__main__":
    main()