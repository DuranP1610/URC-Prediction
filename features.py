"""""
python features.py
"""
from pathlib import Path

import pandas as pd

DATA = Path("data")
SA_TEAMS = {"Bulls", "Lions", "Sharks", "Stormers"}
HIGHVELD = {"Bulls", "Lions"}          # Loftus + Ellis Park, ~1,500 m altitude
WINDOW = 5                             # "recent form" = last 5 games


def load():
    m = pd.read_csv(DATA / "history.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    m["match_id"] = m.index
    return m


def team_view(m):
    """Long format: every match becomes 2 rows, one per team, so we can roll per team."""
    common = {"match_id": m.match_id, "date": m.date, "season": m.season,
              "match_in_sa": m.home.isin(SA_TEAMS).astype(int)}
    home = pd.DataFrame({**common, "team": m.home, "is_home": 1, "pf": m.home_score, "pa": m.away_score})
    away = pd.DataFrame({**common, "team": m.away, "is_home": 0, "pf": m.away_score, "pa": m.home_score})
    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
    long["margin"] = long.pf - long.pa
    return long


def add_team_features(long):
    g = long.groupby("team")

    # rolling averages of the PREVIOUS games (shift(1) = exclude this match -> no leakage)
    for col, name in [("margin", "form"), ("pf", "attack"), ("pa", "defence")]:
        long[name] = g[col].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())

    long["rest_days"] = g["date"].diff().dt.days.clip(upper=30)       # off-season -> capped at 30
    prev_in_sa = g["match_in_sa"].shift(1)
    long["changed_continent"] = (prev_in_sa.notna() & (prev_in_sa != long.match_in_sa)).astype(int)
    return long


def build(m, long):
    cols = ["form", "attack", "defence", "rest_days", "changed_continent"]
    home = long[long.is_home == 1].set_index("match_id")[cols].add_prefix("home_")
    away = long[long.is_home == 0].set_index("match_id")[cols].add_prefix("away_")
    df = m.set_index("match_id").join(home).join(away).reset_index()

    # pre-match Elo from the backtest
    elo = pd.read_csv(DATA / "elo_backtest.csv", parse_dates=["date"])[["date", "home", "away", "elo_home", "elo_away"]]
    df = df.merge(elo, on=["date", "home", "away"], how="left")
    missing = df.elo_home.isna().sum()
    if missing:
        print(f"WARNING: {missing} matches missing Elo - rerun backtest.py")

    # match-level features
    df["elo_diff"] = df.elo_home - df.elo_away
    df["form_diff"] = df.home_form - df.away_form
    df["rest_diff"] = df.home_rest_days - df.away_rest_days
    df["cross_continent"] = (df.home.isin(SA_TEAMS) != df.away.isin(SA_TEAMS)).astype(int)
    df["altitude"] = (df.home.isin(HIGHVELD) & ~df.away.isin(HIGHVELD)).astype(int)
    df["playoff"] = (~df.section.str.startswith("Round")).astype(int)

    # targets
    df["margin"] = df.home_score - df.away_score
    df["home_win"] = (df.margin > 0).astype(int)
    return df


def main():
    m = load()
    long = add_team_features(team_view(m))
    df = build(m, long)
    df.to_csv(DATA / "features.csv", index=False)

    feats = ["elo_diff", "form_diff", "home_attack", "home_defence", "away_attack", "away_defence",
             "home_rest_days", "away_rest_days", "rest_diff", "home_changed_continent",
             "away_changed_continent", "cross_continent", "altitude", "playoff"]
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to data/features.csv\n")
    print("Missing values (first games in the data have no history yet):")
    print(df[feats].isna().sum()[lambda s: s > 0].to_string(), "\n")
    print("Correlation with home margin (first look at what matters):")
    print(df[feats + ["margin"]].corr()["margin"].drop("margin").sort_values(key=abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()