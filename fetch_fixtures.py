"""Fetch the full 2026-27 URC fixture list from Wikipedia -> data/season_fixtures.csv

python fetch_fixtures.py
"""
import re
from pathlib import Path

import mwparserfromhell
import pandas as pd

from scrape_history import (HEADING, add_missing_year, clean, clean_date, fetch_wikitext,
                            fix_season_year, normalise_team, param)

SEASON = "2026–27"                       # en dash, like Wikipedia
OUT = Path("data/season_fixtures.csv")
ROUND = re.compile(r"Round\s+(\d+)", re.IGNORECASE)


def parse_fixtures(wikitext):
    """Every rugbybox under a 'Round N' heading, played or not."""
    pieces = HEADING.split(wikitext)
    sections = [("", pieces[0])] + [(pieces[i + 1], pieces[i + 2]) for i in range(1, len(pieces) - 2, 3)]
    rows = []
    for heading, text in sections:
        rnd = ROUND.search(clean(heading))
        if not rnd:                      # skips play-offs (teams not known yet)
            continue
        for tpl in mwparserfromhell.parse(text).filter_templates():
            if not str(tpl.name).strip().lower().startswith("rugbybox"):
                continue
            home = normalise_team(param(tpl, "home", "team1"))
            away = normalise_team(param(tpl, "away", "team2"))
            if home and away:
                rows.append({
                    "round": int(rnd.group(1)),
                    "date_raw": clean_date(tpl.get("date").value) if tpl.has("date") else "",
                    "home": home, "away": away,
                })
    return pd.DataFrame(rows)


def main():
    df = parse_fixtures(fetch_wikitext(SEASON))
    season = SEASON.replace("–", "-")
    df["date"] = pd.to_datetime([add_missing_year(d, season) for d in df.date_raw],
                                errors="coerce", format="mixed", dayfirst=True)
    df["date"] = [fix_season_year(d, season) for d in df.date]
    df = df.drop(columns="date_raw").sort_values(["round", "date"])
    df["date"] = df["date"].dt.date
    df.to_csv(OUT, index=False)

    # sanity checks: 18 rounds x 8 games, every team 18 games (9 home)
    games = pd.concat([df.home, df.away]).value_counts()
    print(f"Saved {len(df)} fixtures to {OUT}  (expected 144)")
    print(f"Rounds: {df['round'].nunique()} (expected 18), games per round: {sorted(df['round'].value_counts().unique())}")
    print(f"Games per team: {sorted(games.unique())} (expected [18])")
    print(f"Home games per team: {sorted(df.home.value_counts().unique())} (expected [9])")
    print(f"Dates missing: {df['date'].isna().sum()}")


if __name__ == "__main__":
    main()