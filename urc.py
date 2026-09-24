"""CLI:
python urc.py ratings
python urc.py predict <round>
python urc.py result <round> <home> <home_score> <away> <away_score>
python urc.py history
"""
import sys
from pathlib import Path

import pandas as pd

import elo

DATA = Path("data")


def load_seeds():
    seed = pd.read_csv(DATA / "standings_2025_26.csv")
    return {t.team: elo.seed_rating(t.pf, t.pa, t.played) for t in seed.itertuples()}


def current_ratings(params=None):
    results = pd.read_csv(DATA / "results.csv").sort_values(["date", "round"])
    ratings, _ = elo.replay(results.itertuples(), load_seeds(), params)
    return ratings


def save_history(params=None):
    """Pre-match ratings + prediction for every played match."""
    results = pd.read_csv(DATA / "results.csv").sort_values(["date", "round"])
    _, history = elo.replay(results.itertuples(), load_seeds(), params)
    df = pd.DataFrame(history)
    df.to_csv(DATA / "elo_history.csv", index=False)
    print(df.round(2).to_string(index=False))

def show_ratings():
    ratings = pd.Series(current_ratings()).sort_values(ascending=False).round(1)
    print(ratings.to_string())


def predict(rnd):
    r = current_ratings()
    fx = pd.read_csv(DATA / "fixtures.csv")
    fx = fx[fx["round"] == rnd].copy()
    if fx.empty:
        sys.exit(f"No fixtures for round {rnd} - add them to data/fixtures.csv first")

    fx["model_margin"] = [elo.predict_margin(r.get(h,0.0), r.get(a,0.0)) for h, a in zip(fx.home, fx.away)]
    fx["p_home_win"] = fx["model_margin"].apply(elo.win_prob).round(2)
    fx["model_margin"] = fx["model_margin"].round(1)

    print(fx[["home", "away", "model_margin", "p_home_win", "bookie_home_margin"]].to_string(index=False))

    # log predictions BEFORE kickoff (replace this round's rows if re-run)
    log_path = DATA / "predictions_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        fx = pd.concat([log[log["round"] != rnd], fx])
    fx.to_csv(log_path, index=False)


def add_result(rnd, home, home_score, away, away_score):
    """Save one result, but only if it matches a real fixture."""
    fx = pd.read_csv(DATA / "fixtures.csv")
    match = fx[(fx["round"] == rnd) & (fx.home == home) & (fx.away == away)]
    if match.empty:
        sys.exit(f"No fixture: round {rnd} {home} v {away} (check spelling and home/away order)")

    res_path = DATA / "results.csv"
    res = pd.read_csv(res_path)

    # entering the same match twice replaces it (handy for fixing typos)
    dup = (res["round"] == rnd) & (res.home == home) & (res.away == away)
    if dup.any():
        print("Result already existed - replacing it")
        res = res[~dup]

    row = pd.DataFrame([{
        "round": rnd, "date": match.iloc[0]["date"], "home": home, "away": away,
        "home_score": home_score, "away_score": away_score,
    }])
    res = row if res.empty else pd.concat([res, row])
    res.sort_values(["date", "round"]).to_csv(res_path, index=False)
    print(f"Saved: {home} {home_score}-{away_score} {away}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ratings"
    if cmd == "ratings":
        show_ratings()
    elif cmd == "predict":
        predict(int(sys.argv[2]))
    elif cmd == "result":
        _, _, rnd, home, hs, away, as_ = sys.argv
        add_result(int(rnd), home, int(hs), away, int(as_))
    elif cmd == "history":
        save_history()
    else:
        sys.exit(__doc__)