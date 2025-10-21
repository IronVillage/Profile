"""
MLBB Tournament Dataset Creator

Fetches tournament data from Liquipedia and creates a comprehensive dataset.

Each URL in the input file represents a specific tournament instance (e.g., "MPL Season 14").
The script fetches all games from each tournament URL - no filtering by date or year.

Output:
- mlbb_data.csv - Complete dataset with all game data (including side information when available)
"""

import requests
import pandas as pd
import re
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional


class MLBBDatasetCreator:
    """
    Unified fetcher for MLBB tournament data from Liquipedia
    
    How it works:
    - Each URL = one specific tournament instance (e.g., MPL Indonesia Season 14)
    - Fetches ALL games from each tournament URL
    - No date/year filtering - each URL is already a specific tournament
    - Deduplication via map_id hash (teams + date + picks + bans + winner + duration)
    """
    
    def __init__(self):
        self.base_url = "https://liquipedia.net/mobilelegends/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MLBBDatasetCreator/1.0',
            'Accept-Encoding': 'gzip'
        })
        self.all_games = []
        
    def _rate_limit(self):
        """0.5s delay between requests"""
        time.sleep(0.5)
    
    def fetch_page_content(self, page_title: str) -> Optional[str]:
        """Fetch raw wiki content for a page"""
        self._rate_limit()
        
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main"
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                if "revisions" in page_data:
                    return page_data["revisions"][0]["slots"]["main"]["*"]
        except Exception as e:
            print(f"    [ERROR] Failed to fetch {page_title}: {e}", flush=True)
        
        return None
    
    def get_all_subpages(self, base_page: str) -> List[str]:
        """Get all subpages for a tournament"""
        self._rate_limit()
        
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "apprefix": base_page + "/",
            "aplimit": 500
        }
        
        subpages = []
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for page_info in data.get("query", {}).get("allpages", []):
                page_title = page_info.get("title", "")
                # Skip Statistics pages (no game data)
                if '/Statistics' not in page_title:
                    subpages.append(page_title)
        except Exception as e:
            print(f"    [ERROR] Failed to fetch subpages: {e}", flush=True)
        
        return subpages
    
    def extract_tournament_name(self, content: str, page_title: str) -> str:
        """Extract clean tournament name from infobox"""
        pattern = r'\|name\s*=\s*([^\n|]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove wiki formatting
            name = re.sub(r'\{\{[^\}]*\}\}', '', name)
            name = re.sub(r'\[\[([^\]|]+)\|?[^\]]*\]\]', r'\1', name)
            return name.strip()
        
        return page_title.replace('_', ' ')
    
    def get_tier(self, content: str) -> Optional[str]:
        """Extract tournament tier"""
        tier_match = re.search(r'\|liquipediatier\s*=\s*(\d+)', content, re.IGNORECASE)
        if tier_match:
            tier_num = tier_match.group(1)
            tier_map = {'1': 'S-Tier', '2': 'A-Tier', '3': 'B-Tier', '4': 'C-Tier'}
            return tier_map.get(tier_num, f'Tier-{tier_num}')
        return None
    
    def has_match_data(self, content: str) -> bool:
        """Check if page has match data (not a redirect or empty page)"""
        if not content or len(content) < 100:
            return False
        # Check for match templates
        return bool(re.search(r'\{\{Match', content, re.IGNORECASE))
    
    def parse_date(self, date_str: str) -> Optional[str]:
        """Parse date to YYYY-MM-DD format - only if complete date with year is present"""
        if not date_str or pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        # Remove HTML comments and wiki templates
        date_str = re.sub(r'<!--.*?-->', '', date_str)
        date_str = re.sub(r'\{\{[^\}]*\}\}', '', date_str)
        
        # Remove ordinal suffixes
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        date_str = re.sub(r'\s*,\s*', ', ', date_str)
        
        # Already in YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # Remove time portions
        date_str = re.sub(r'\s*-\s*\d{1,2}:\d{2}.*$', '', date_str)
        date_str = re.sub(r'\s+\d{1,2}:\d{2}.*$', '', date_str)
        date_str = date_str.strip()
        
        # Only parse dates that include year
        formats = [
            '%B %d, %Y', '%B %d,%Y', '%B %d %Y',
            '%b %d, %Y', '%b %d,%Y', '%b %d %Y',
            '%d %B %Y', '%d %b %Y',
            '%Y-%m-%d', '%d.%m.%Y',
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no year present, return None (no fallback)
        return None
    
    def format_duration(self, duration_str: str) -> str:
        """Format game duration to XXm YYs"""
        if not duration_str:
            return ""
        
        duration_str = str(duration_str).strip()
        
        if re.match(r'^\d+m\s+\d+s$', duration_str):
            return duration_str
        
        time_match = re.match(r'^(\d+):(\d+)', duration_str)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            return f"{minutes}m {seconds}s"
        
        return duration_str
    
    def create_map_hash(self, team1: str, team2: str, match_date: str, game_number: int, 
                        team1_picks: str = "", team2_picks: str = "", 
                        team1_bans: str = "", team2_bans: str = "",
                        winner: int = 0, duration: str = "") -> str:
        """Create unique hash for deduplication - same game = same hash"""
        teams = tuple(sorted([team1.strip().lower(), team2.strip().lower()]))
        date_str = str(match_date) if match_date else "no_date"
        
        def normalize_heroes(hero_str):
            if not hero_str or hero_str == "nan":
                return ""
            heroes = [h.strip().lower() for h in str(hero_str).split(',') if h.strip()]
            return '|'.join(sorted(heroes))
        
        picks1 = normalize_heroes(team1_picks)
        picks2 = normalize_heroes(team2_picks)
        bans1 = normalize_heroes(team1_bans)
        bans2 = normalize_heroes(team2_bans)
        
        # Hash combines all game-identifying data
        hash_input = f"{teams[0]}|{teams[1]}|{date_str}|{game_number}|{picks1}|{picks2}|{bans1}|{bans2}|{winner}|{duration}"
        hash_obj = hashlib.md5(hash_input.encode('utf-8'))
        return hash_obj.hexdigest()[:12]
    
    def parse_games(self, content: str, tournament_info: Dict) -> List[Dict]:
        """Parse all games from page content"""
        games = []
        
        if not content:
            return games
        
        # Find all match templates
        match_patterns = [
            r'\|M\d+\s*=\s*\{\{Match(.*?)(?=\|M\d+\s*=|\|R\d+|\}\}$|\Z)',
            r'\|R\d+M\d+\s*=\s*\{\{Match(.*?)(?=\|R\d+M\d+\s*=|\|M\d+\s*=|\}\}$|\Z)',
        ]
        
        all_matches = []
        for pattern in match_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            all_matches.extend(matches)
        
        # Deduplicate matches
        seen = set()
        unique_matches = []
        for match in all_matches:
            key = match[:100]
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        for match_content in unique_matches:
            # Extract match date (skip games without dates)
            date_match = re.search(r'\|date\s*=\s*([^\n|]+)', match_content, re.IGNORECASE)
            match_date = self.parse_date(date_match.group(1)) if date_match else None
            
            if not match_date:
                continue
            
            # Extract teams
            team1_match = re.search(r'\|opponent1\s*=\s*\{\{TeamOpponent\|([^}|]+)', match_content, re.IGNORECASE)
            team2_match = re.search(r'\|opponent2\s*=\s*\{\{TeamOpponent\|([^}|]+)', match_content, re.IGNORECASE)
            
            team1 = team1_match.group(1).strip() if team1_match else "Unknown"
            team2 = team2_match.group(1).strip() if team2_match else "Unknown"
            
            # Find all maps/games
            map_pattern = r'\|map(\d+)\s*=\s*\{\{Map(.*?)\}\}'
            maps = re.findall(map_pattern, match_content, re.DOTALL | re.IGNORECASE)
            
            for map_num, map_content in maps:
                game_number = int(map_num)
                
                # Extract winner
                winner_match = re.search(r'\|winner\s*=\s*(\d+)', map_content, re.IGNORECASE)
                winner = int(winner_match.group(1)) if winner_match else 0
                
                # Extract duration
                duration_match = re.search(r'\|length\s*=\s*([^\n|]+)', map_content, re.IGNORECASE)
                duration = self.format_duration(duration_match.group(1)) if duration_match else ""
                
                # Extract hero picks and bans
                team1_picks_list = []
                team1_bans_list = []
                team2_picks_list = []
                team2_bans_list = []
                
                for team_num in [1, 2]:
                    pick_pattern = rf'\|t{team_num}h(\d+)\s*=\s*([^\n|]+)'
                    picks = re.findall(pick_pattern, map_content, re.IGNORECASE)
                    picks = [p[1].strip() for p in sorted(picks, key=lambda x: int(x[0])) if p[1].strip()]
                    
                    ban_pattern = rf'\|t{team_num}b(\d+)\s*=\s*([^\n|]+)'
                    bans = re.findall(ban_pattern, map_content, re.IGNORECASE)
                    bans = [b[1].strip() for b in sorted(bans, key=lambda x: int(x[0])) if b[1].strip()]
                    
                    if team_num == 1:
                        team1_picks_list = picks
                        team1_bans_list = bans
                    else:
                        team2_picks_list = picks
                        team2_bans_list = bans
                
                team1_picks_str = ', '.join(team1_picks_list) if team1_picks_list else ""
                team1_bans_str = ', '.join(team1_bans_list) if team1_bans_list else ""
                team2_picks_str = ', '.join(team2_picks_list) if team2_picks_list else ""
                team2_bans_str = ', '.join(team2_bans_list) if team2_bans_list else ""
                
                # Create unique map ID
                map_hash = self.create_map_hash(
                    team1, team2, match_date, game_number,
                    team1_picks_str, team2_picks_str,
                    team1_bans_str, team2_bans_str,
                    winner, duration
                )
                map_id = f"MAP{map_hash}"
                
                # Extract side information (if available)
                team1_side_match = re.search(r'\|team1side\s*=\s*([^\n|]+)', map_content, re.IGNORECASE)
                team2_side_match = re.search(r'\|team2side\s*=\s*([^\n|]+)', map_content, re.IGNORECASE)
                
                team1_side = team1_side_match.group(1).strip().lower() if team1_side_match else None
                team2_side = team2_side_match.group(1).strip().lower() if team2_side_match else None
                
                # Build game data
                game_data = {
                    'map_id': map_id,
                    'game_number': game_number,
                    'tournament_name': tournament_info['tournament_name'],
                    'tier': tournament_info['tier'],
                    'tournament_stage': tournament_info['tournament_stage'],
                    'match_date': match_date,
                    'team1': team1,
                    'team2': team2,
                    'winner': winner,
                    'duration': duration,
                }
                
                # Add picks/bans if present
                if team1_picks_str:
                    game_data['team1_picks'] = team1_picks_str
                if team1_bans_str:
                    game_data['team1_bans'] = team1_bans_str
                if team2_picks_str:
                    game_data['team2_picks'] = team2_picks_str
                if team2_bans_str:
                    game_data['team2_bans'] = team2_bans_str
                
                # Add side information if present
                if team1_side:
                    game_data['team1_side'] = team1_side
                if team2_side:
                    game_data['team2_side'] = team2_side
                
                games.append(game_data)
        
        return games
    
    def process_tournament_url(self, url: str, index: int, total: int) -> int:
        """Process a single tournament URL and return number of games found"""
        if '/mobilelegends/' not in url:
            print(f"[{index+1}/{total}] Invalid URL format: {url}", flush=True)
            return 0
        
        page_title = url.split('/mobilelegends/')[-1]
        print(f"\n[{index+1}/{total}] {page_title}...", end=" ", flush=True)
        
        # Fetch main page (just to get tournament info - matches might be on subpages)
        main_content = self.fetch_page_content(page_title)
        if not main_content:
            print(f"skipped (no content)", flush=True)
            return 0
        
        # Extract tournament info
        tournament_name = self.extract_tournament_name(main_content, page_title)
        tier = self.get_tier(main_content)
        
        games_found = 0
        
        # Get all subpages (playoffs, groups, etc.)
        subpages = self.get_all_subpages(page_title)
        
        # Process main page + all subpages
        all_pages = [(page_title, "Main", main_content)] + [(sp, sp.replace(page_title + '/', ''), None) for sp in subpages]
        
        for liquipedia_path, stage_name, cached_content in all_pages:
            # Fetch content if not cached
            if cached_content is None:
                content = self.fetch_page_content(liquipedia_path)
                if not content:
                    continue
            else:
                content = cached_content
            
            # Skip pages without match data
            if not self.has_match_data(content):
                continue
            
            # Parse games
            tournament_info = {
                'tournament_name': tournament_name,
                'tier': tier,
                'tournament_stage': stage_name
            }
            
            games = self.parse_games(content, tournament_info)
            
            if games:
                self.all_games.extend(games)
                games_found += len(games)
        
        print(f"{games_found} games", flush=True)
        return games_found
    
    def fetch_all(self, urls_file: str, output_prefix: str = "mlbb_data"):
        """Fetch all tournaments from URL list and save results"""
        print("="*80, flush=True)
        print("MLBB Dataset Creator", flush=True)
        print("="*80, flush=True)
        
        # Read URLs
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"\nLoaded {len(urls)} tournament URLs\n", flush=True)
        
        # Process each tournament
        for i, url in enumerate(urls):
            try:
                self.process_tournament_url(url, i, len(urls))
            except KeyboardInterrupt:
                print("\n\n[!] Interrupted by user", flush=True)
                break
            except Exception as e:
                print(f"\n[ERROR] {e}", flush=True)
                continue
        
        # Save dataset
        if self.all_games:
            df = pd.DataFrame(self.all_games)
            
            # Remove duplicates by map_id
            before_dedup = len(df)
            df = df.drop_duplicates(subset=['map_id'], keep='first')
            after_dedup = len(df)
            
            # Save output file
            output_file = f"{output_prefix}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            # Check side data coverage
            if 'team1_side' in df.columns:
                coverage = (df['team1_side'].notna().sum() / len(df)) * 100
                side_info = f" ({coverage:.1f}% with side data)"
            else:
                side_info = ""
            
            print(f"\n{'='*80}", flush=True)
            print(f"[SUCCESS] Dataset created: {after_dedup} unique games{side_info}", flush=True)
            print(f"  Saved to: {output_file}", flush=True)
            print(f"  Duplicates removed: {before_dedup - after_dedup}", flush=True)
            print(f"{'='*80}", flush=True)
        else:
            print("\n[WARNING] No games found!", flush=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mlbb_dataset_creator.py <urls_file>")
        print("\nExamples:")
        print("  python mlbb_dataset_creator.py tournament_urls.txt")
        print("  python mlbb_dataset_creator.py test_urls.txt")
        sys.exit(1)
    
    urls_file = sys.argv[1]
    
    creator = MLBBDatasetCreator()
    creator.fetch_all(urls_file, "mlbb_data")

