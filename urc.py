"""CLI:
python urc.py ratings
python urc.py predict <round>
python urc.py result <round> <home> <home_score> <away> <away_score>
python urc.py history
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

import elo
from features import HIGHVELD, SA_TEAMS

DATA = Path("data")
MODEL_B_FEATURES = ["elo_diff", "cross_continent", "altitude"]


def load_seeds():
    """Replay every past season, then carry ratings into 2026-27."""
    history = pd.read_csv(DATA / "history.csv").sort_values("date")
    ratings = {}
    for _, games in history.groupby("season", sort=True):
        ratings = {t: elo.CARRY * v for t, v in ratings.items()}
        ratings, _ = elo.replay(games.itertuples(), ratings)
    return {t: elo.CARRY * v for t, v in ratings.items()}


def load_results():
    return pd.read_csv(DATA / "results.csv").sort_values(["date", "round"])


def current_ratings(params=None):
    ratings, _ = elo.replay(load_results().itertuples(), load_seeds(), params)
    return ratings


def add_context(df):
    """Travel + altitude: known from the team names alone, before kickoff."""
    df = df.copy()
    df["cross_continent"] = (df.home.isin(SA_TEAMS) != df.away.isin(SA_TEAMS)).astype(int)
    df["altitude"] = (df.home.isin(HIGHVELD) & ~df.away.isin(HIGHVELD)).astype(int)
    return df


def train_model_b():
    """Frozen spec: linear regression on Elo diff + travel + altitude, trained on 2022-26."""
    f = pd.read_csv(DATA / "features.csv")
    f = f[f.season != "2021-22"].dropna(subset=MODEL_B_FEATURES)
    model = LinearRegression()
    oof = cross_val_predict(model, f[MODEL_B_FEATURES], f.margin, cv=KFold(5))
    sigma = (f.margin - oof).std()
    model.fit(f[MODEL_B_FEATURES], f.margin)
    return model, sigma


def save_history(params=None):
    """Pre-match ratings + prediction for every played match."""
    _, history = elo.replay(load_results().itertuples(), load_seeds(), params)
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

    # Model A: Elo
    fx["elo_margin"] = [elo.predict_margin(r.get(h, 0.0), r.get(a, 0.0)) for h, a in zip(fx.home, fx.away)]
    fx["elo_p_home"] = fx["elo_margin"].apply(elo.win_prob)

    # Model B: Elo diff + travel + altitude
    fx["elo_diff"] = [r.get(h, 0.0) - r.get(a, 0.0) for h, a in zip(fx.home, fx.away)]
    fx = add_context(fx)
    model, sigma = train_model_b()
    fx["b_margin"] = model.predict(fx[MODEL_B_FEATURES])
    fx["b_p_home"] = fx["b_margin"].apply(lambda m: elo.win_prob(m, sigma))

    show = ["home", "away", "elo_margin", "elo_p_home", "b_margin", "b_p_home", "bookie_home_margin"]
    print(fx[show].round(2).to_string(index=False))

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