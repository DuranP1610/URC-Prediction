import math

HFA = 6.0     # home advantage, in points
K = 0.10      # learning rate
SIGMA = 14.1  # spread of real margins around the prediction (points)
CAP = 30      # cap blowouts
SHRINK = 0.6
CARRY = 0.7  # pull last season's form back toward average

def seed_rating(pf, pa, played, shrink=SHRINK):
    return shrink * (pf - pa) / played

def predict_margin(r_home, r_away, hfa=HFA):
    return r_home + hfa - r_away

def win_prob(margin, sigma=SIGMA):
    """P(home wins) = Normal CDF of predicted margin."""
    return 0.5 * (1 + math.erf(margin / (sigma * math.sqrt(2))))

def update(r_home, r_away, home_score, away_score, k=K, hfa=HFA, cap=CAP):
    actual = max(-cap, min(cap, home_score - away_score))
    surprise = actual - predict_margin(r_home, r_away, hfa)
    return r_home + k * surprise, r_away - k * surprise

DEFAULTS = {"hfa": HFA, "k": K, "sigma": SIGMA, "cap": CAP}


def replay(matches, seeds, params=None):
    """
    matches: iterable of rows with .date .home .away .home_score .away_score
    seeds:   {team: starting rating}
    params:  overrides for DEFAULTS, e.g. {"k": 0.2}

    Returns (final_ratings, history). Each history row holds the ratings
    and prediction from BEFORE that match was played.
    """
    p = {**DEFAULTS, **(params or {})}
    r = dict(seeds)
    history = []
    for m in matches:
        rh, ra = r.get(m.home, 0.0), r.get(m.away, 0.0)   # unknown team -> average
        margin = predict_margin(rh, ra, p["hfa"])
        history.append({
            "date": m.date, "home": m.home, "away": m.away,
            "elo_home": rh, "elo_away": ra,
            "pred_margin": margin,
            "p_home_win": win_prob(margin, p["sigma"]),
            "actual_margin": m.home_score - m.away_score,
        })
        r[m.home], r[m.away] = update(rh, ra, m.home_score, m.away_score, p["k"], p["hfa"], p["cap"])
    return r, history