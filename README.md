# URC Rugby Predictor

Predicting United Rugby Championship 2026/27 match outcomes.

**Status:** 
- Elo baseline complete. 
- ML model + ensemble in progress.

## Model A: points-based Elo
- Ratings in points (+5 = 5 points better than an average team)
- Predicted margin = home rating + home advantage − away rating
- Win probability from a normal distribution around the predicted margin
- Seeded from 2025/26 points difference, updated after every match
- Predictions logged before kickoff each round (`data/predictions_log.csv`)

## Usage
    python urc.py predict <round>
    python urc.py result <round> <home> <home_score> <away> <away_score>
    python urc.py ratings
    python urc.py history
