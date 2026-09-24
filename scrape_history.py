"""Scrape URC match results from Wikipedia into data/history.csv

python scrape_history.py
"""
import re
from pathlib import Path

import mwparserfromhell
import pandas as pd
import requests

SEASONS = ["2021–22", "2022–23", "2023–24", "2024–25", "2025–26"]   # en dash, like Wikipedia
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "urc-elo-student-project/0.1 (github.com/DuranP1610)"}
OUT = Path("data/history.csv")

# map any spelling Wikipedia uses -> our short names (checked by 'keyword in name')
TEAM_KEYWORDS = {
    "Benetton": "Benetton", "Bulls": "Bulls", "Cardiff": "Cardiff", "Connacht": "Connacht",
    "Dragons": "Dragons", "Edinburgh": "Edinburgh", "Glasgow": "Glasgow", "Leinster": "Leinster",
    "Lions": "Lions", "Munster": "Munster", "Ospreys": "Ospreys", "Scarlets": "Scarlets",
    "Sharks": "Sharks", "Stormers": "Stormers", "Ulster": "Ulster", "Zebre": "Zebre",
}
HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$", re.MULTILINE)
SCORE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)")


def fetch_wikitext(season):
    params = {"action": "parse", "page": f"{season} United Rugby Championship",
              "prop": "wikitext", "format": "json", "formatversion": 2}
    resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["parse"]["wikitext"]


def clean(value):
    """Wiki markup -> plain text: '[[Leinster Rugby|Leinster]]' -> 'Leinster'."""
    return mwparserfromhell.parse(str(value)).strip_code().strip()


def normalise_team(name):
    for key, short in TEAM_KEYWORDS.items():
        if key.lower() in name.lower():
            return short
    return None

def clean_date(value):
    code = mwparserfromhell.parse(str(value))
    text = code.strip_code().strip()
    for t in code.filter_templates():
        args = [str(p.value).strip() for p in t.params if not p.showkey]
        if len(args) >= 3 and all(a.isdigit() for a in args[:3]):
            return f"{args[0]}-{int(args[1]):02d}-{int(args[2]):02d}"
        if args:
            text = args[0]
            break
    text = re.sub(r"\(.*?\)", "", text)          # drop notes like "(rearranged)"
    text = text.replace("Match ", "March ")     # Wikipedia typo
    return text.strip()

LATE_MONTHS = ("August", "September", "October", "November", "December")


def add_missing_year(date_raw, season):
    """'17 November' in season 2023-24 -> '17 November 2023'."""
    if re.search(r"\d{4}", date_raw):
        return date_raw
    first_year = int(season[:4])
    year = first_year if any(m in date_raw for m in LATE_MONTHS) else first_year + 1
    return f"{date_raw} {year}"

def fix_season_year(date, season):
    """Force the year to match the season: Aug-Dec -> first year, Jan-Jul -> second year."""
    if pd.isna(date):
        return date
    first_year = int(season[:4])
    expected = first_year if date.month >= 8 else first_year + 1
    if date.year != expected:
        print(f"  fixed year: {date.date()} -> {date.replace(year=expected).date()} ({season})")
        return date.replace(year=expected)
    return date


def param(tpl, *names):
    for n in names:
        if tpl.has(n):
            return clean(tpl.get(n).value)
    return ""


def parse_season(season, wikitext):
    """Split the page by headings so every match knows which section (round) it sits in."""
    rows, unknown = [], set()
    pieces = HEADING.split(wikitext)          # [text, '==', heading, text, '==', heading, text, ...]
    sections = [("", pieces[0])] + [(pieces[i + 1], pieces[i + 2]) for i in range(1, len(pieces) - 2, 3)]

    for heading, text in sections:
        for tpl in mwparserfromhell.parse(text).filter_templates():
            if not str(tpl.name).strip().lower().startswith("rugbybox"):
                continue
            home_raw, away_raw = param(tpl, "home", "team1"), param(tpl, "away", "team2")
            home, away = normalise_team(home_raw), normalise_team(away_raw)
            score = SCORE.search(param(tpl, "score"))
            if not (home and away):
                unknown.update(n for n, t in [(home_raw, home), (away_raw, away)] if not t)
                continue
            if not score:        # postponed / cancelled / not played
                continue
            rows.append({
                "season": season.replace("–", "-"),
                "section": clean(heading),
                "date_raw": clean_date(tpl.get("date").value) if tpl.has("date") else "",
                "home": home, "away": away,
                "home_score": int(score.group(1)), "away_score": int(score.group(2)),
                "venue": param(tpl, "stadium", "venue"),
            })
    return rows, unknown


def main():
    all_rows = []
    for season in SEASONS:
        rows, unknown = parse_season(season, fetch_wikitext(season))
        print(f"{season}: {len(rows)} matches" + (f"  | unrecognised names: {sorted(unknown)}" if unknown else ""))
        all_rows += rows

    df = pd.DataFrame(all_rows)
    df["date_raw"] = [add_missing_year(d, s) for d, s in zip(df.date_raw, df.season)]
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce", format="mixed", dayfirst=True)
    df["date"] = [fix_season_year(d, s) for d, s in zip(df.date, df.season)]
    print("Date range per season:\n", df.groupby("season")["date"].agg(["min", "max"]))
    bad = df["date"].isna().sum()
    if bad:
        print(f"WARNING: {bad} dates didn't parse, e.g. {df.loc[df.date.isna(), 'date_raw'].head(3).tolist()}")

    df = df.sort_values("date").drop(columns="date_raw")
    df["date"] = df["date"].dt.date
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} matches to {OUT}")
    print(df.head().to_string(index=False))




if __name__ == "__main__":
    main()