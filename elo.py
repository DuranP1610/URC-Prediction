import math

HFA = 4.0     # home advantage, in points
K = 0.15      # learning rate
SIGMA = 13.5  # spread of real margins around the prediction (points)
CAP = 30      # cap blowouts
SHRINK = 0.6  # pull last season's form back toward average

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
