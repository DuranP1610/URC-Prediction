"""Monte Carlo the rest of the 2026-27 season with Model B.

python simulate.py            (10,000 seasons)
"""
import numpy as np
import pandas as pd

from urc import DATA, MODEL_B_FEATURES, add_context, current_ratings, load_results, train_model_b

N_SIMS = 10_000
rng = np.random.default_rng(42)


def predicted_margins(pairs, ratings, model):
    """Model B margin for any list of (home, away) pairs."""
    df = pd.DataFrame(pairs, columns=["home", "away"])
    df["elo_diff"] = [ratings.get(h, 0.0) - ratings.get(a, 0.0) for h, a in pairs]
    return model.predict(add_context(df)[MODEL_B_FEATURES])


def league_points(margin):
    """URC points from the home side's view: win 4, draw 2, losing bonus 1 (lose by <= 7).
    Try bonus (4+ tries) not modelled - we only predict margins, not tries."""
    home = 4 * (margin > 0) + 2 * (margin == 0) + 1 * ((margin < 0) & (margin >= -7))
    away = 4 * (margin < 0) + 2 * (margin == 0) + 1 * ((margin > 0) & (margin <= 7))
    return home, away


def main():
    fixtures = pd.read_csv(DATA / "season_fixtures.csv")
    played = load_results()
    teams = sorted(set(fixtures.home) | set(fixtures.away))
    T = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    ratings = current_ratings()
    model, sigma = train_model_b()

    # ---- table so far (real results) ----
    base = np.zeros((3, n_teams))                      # rows: points, wins, points difference
    for m in played.itertuples():
        margin = m.home_score - m.away_score
        hp, ap = league_points(np.array(margin))
        h, a = T[m.home], T[m.away]
        base[0, h] += hp
        base[0, a] += ap
        base[1, h] += margin > 0
        base[1, a] += margin < 0
        base[2, h] += margin
        base[2, a] -= margin

    # ---- simulate remaining regular-season games ----
    done = set(zip(played.home, played.away))
    remaining = fixtures[[(h, a) not in done for h, a in zip(fixtures.home, fixtures.away)]]
    pred = predicted_margins(list(zip(remaining.home, remaining.away)), ratings, model)
    print(f"{len(played)} played, {len(remaining)} to simulate, sigma={sigma:.1f}, {N_SIMS:,} seasons\n")

    margins = np.rint(pred + sigma * rng.standard_normal((N_SIMS, len(remaining))))
    hp, ap = league_points(margins)
    H = np.eye(n_teams)[[T[t] for t in remaining.home]]     # one-hot: game -> home team
    A = np.eye(n_teams)[[T[t] for t in remaining.away]]
    points = base[0] + hp @ H + ap @ A
    wins = base[1] + (margins > 0) @ H + (margins < 0) @ A
    pdiff = base[2] + margins @ H - margins @ A

    # ---- final standings: points, then wins, then points difference ----
    key = points * 1e6 + wins * 1e3 + pdiff + rng.random(points.shape) * 1e-3   # tiny noise breaks exact ties
    order = np.argsort(-key, axis=1)                   # order[s, 0] = team finishing 1st in season s
    pos = np.argsort(order, axis=1)                    # pos[s, team] = finishing position (0 = 1st)

    # ---- play-offs: 1v8, 2v7, 3v6, 4v5, higher seed at home every round ----
    pair_margin = predicted_margins([(h, a) for h in teams for a in teams], ratings, model).reshape(n_teams, n_teams)
    rows = np.arange(N_SIMS)

    def play(x, y):
        home = np.where(pos[rows, x] < pos[rows, y], x, y)
        away = np.where(home == x, y, x)
        m = pair_margin[home, away] + sigma * rng.standard_normal(N_SIMS)
        return np.where(m > 0, home, away)

    qf = [play(order[:, i], order[:, 7 - i]) for i in range(4)]    # 1v8, 2v7, 3v6, 4v5
    sf1, sf2 = play(qf[0], qf[3]), play(qf[1], qf[2])
    champion = play(sf1, sf2)

    # ---- summary ----
    out = pd.DataFrame({
        "exp_points": points.mean(axis=0),
        "top_8": (pos < 8).mean(axis=0),
        "top_4": (pos < 4).mean(axis=0),
        "1st": (pos == 0).mean(axis=0),
        "champion": np.bincount(champion, minlength=n_teams) / N_SIMS,
    }, index=teams).sort_values("champion", ascending=False)
    print(out.round(3).to_string())
    out.to_csv(DATA / "season_odds.csv")
    # keep a snapshot per round, so odds can be charted over the season
    after_round = int(played["round"].max()) if len(played) else 0
    snap = out.reset_index(names="team").assign(after_round=after_round)
    hist_path = DATA / "odds_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        snap = pd.concat([hist[hist.after_round != after_round], snap])   # re-running a round replaces it
    snap.to_csv(hist_path, index=False)
    print(f"\nSaved odds after round {after_round} to {hist_path}")


if __name__ == "__main__":
    main()