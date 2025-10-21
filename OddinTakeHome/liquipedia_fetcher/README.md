# MLBB Tournament Dataset Creator

Scrapes Mobile Legends: Bang Bang tournament data from Liquipedia and creates structured CSV datasets.

---

## What It Does

Fetches tournament data from Liquipedia URLs and outputs CSV with:
- Match metadata (tournament, stage, date, teams)
- Game statistics (winner, duration)
- Hero picks and bans
- Blue/red side assignments (when available)


**Features:**
- Automatic deduplication via hash-based MAP IDs
- Processes main tournament pages + all subpages
---

## Usage

```bash
pip install -r requirements.txt

python mlbb_dataset_creator.py tournament_urls.txt
```

**Output:** `mlbb_data.csv`

## Tournament URLs Format

File contains Liquipedia URLs, one per line:

```
https://liquipedia.net/mobilelegends/M6_World_Championship
https://liquipedia.net/mobilelegends/MSC/2024
https://liquipedia.net/mobilelegends/MPL/Indonesia/Season_14
# Comments with #
```

Each URL represents a specific tournament instance. The script fetches all games from that tournament.

---

## How It Works

1. **Fetch Pages:** Reads URLs, fetches tournament page + all subpages via Liquipedia API
2. **Parse Data:** Extracts match templates from wiki markup, parses teams, dates, results, picks/bans
3. **Generate Hash:** Creates unique MAP ID for each game (teams + date + picks + bans + winner + duration)
4. **Save CSV:** Removes duplicates, saves dataset with statistics

**Deduplication:**
```python
# Same game from multiple pages = same hash = auto-dedup
hash_input = f"{teams}|{date}|{game_number}|{picks}|{bans}|{winner}|{duration}"
map_id = f"MAP{md5(hash_input)[:12]}"
```

---

## Dataset Schema

| Column | Description |
|--------|-------------|
| `map_id` | Unique game identifier |
| `game_number` | Game number in match (1, 2, 3...) |
| `tournament_name` | Tournament name |
| `tier` | Tournament tier (S-Tier, A-Tier, etc.) |
| `tournament_stage` | Stage (Playoffs, Grand Finals, etc.) |
| `match_date` | Date (YYYY-MM-DD) |
| `team1` / `team2` | Team names |
| `winner` | Winner (1 = team1, 2 = team2, 0 = draw) |
| `duration` | Game duration (15m 43s) |
| `team1_picks` / `team2_picks` | Hero picks (comma-separated) |
| `team1_bans` / `team2_bans` | Hero bans (comma-separated) |
| `team1_side` / `team2_side` | Side (blue/red, if available) |



