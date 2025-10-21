import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from collections import defaultdict
import config


SYSTEM_CONTEXT = """You are an expert AI data extractor for Mobile Legends: Bang Bang (MLBB) post-game statistics screens. 
Your sole purpose is to analyze stat screen images and extract structured data in JSON format. Be precise and accurate."""

USER_PROMPT = """Analyze this MLBB post-game statistics screen and extract the following data:

**Required Fields:**
1. Team Names (abbreviations, e.g., "ECHO", "RSG", "ONIC")
2. Team Kills (integer for each team)
3. Game Duration (format: "MM:SS")
4. Team Total Gold (integer for each team, if visible)

**JSON Structure:**
{
  "game_duration": "MM:SS",
  "teams": [
    {
      "side": "left",
      "name": "TEAM_ABBR",
      "kills": 0,
      "total_gold": 0
    },
    {
      "side": "right", 
      "name": "TEAM_ABBR",
      "kills": 0,
      "total_gold": 0
    }
  ]
}

Return ONLY the JSON object, no additional text."""


def initialize_vertex_ai():
    vertexai.init(project=config.VERTEX_PROJECT_ID, location=config.VERTEX_LOCATION)
    model = GenerativeModel(model_name=config.VERTEX_MODEL_ENDPOINT)
    return model


def parse_single_image(model, image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
        full_prompt = f"{SYSTEM_CONTEXT}\n\n{USER_PROMPT}"
        
        response = model.generate_content(
            [image_part, Part.from_text(full_prompt)],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
            }
        )
        
        response_text = response.text.strip()
        
        # Sometimes Gemini wraps JSON in markdown code blocks
        if response_text.startswith('```'):
            response_text = response_text.strip('`').strip()
            if response_text.startswith('json'):
                response_text = response_text[4:].strip()
        
        parsed_data = json.loads(response_text)
        
        # Basic validation - must have kills data
        has_kills = all(
            team.get('kills') is not None 
            for team in parsed_data.get('teams', [])
        )
        
        if not has_kills:
            return None
        
        parsed_data['source_file'] = image_path.name
        return parsed_data
        
    except (json.JSONDecodeError, KeyError, IOError):
        return None


def process_image_wrapper(args):
    image_path = args
    try:
        model = initialize_vertex_ai()
        result = parse_single_image(model, image_path)
        return (image_path, result, result is not None)
    except Exception:
        return (image_path, None, False)


def parse_duration_to_seconds(duration_str):
    try:
        if not duration_str or duration_str == "null":
            return None
        parts = duration_str.strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, AttributeError):
        pass
    return None


def create_game_hash(game_json):
    """Create a unique hash for deduplication - same game = same hash"""
    try:
        teams = game_json.get('teams', [])
        if len(teams) != 2:
            return None
        
        t1, t2 = teams[0], teams[1]
        
        kills1 = t1.get('kills')
        kills2 = t2.get('kills')
        name1 = str(t1.get('name', '')).lower().strip()
        name2 = str(t2.get('name', '')).lower().strip()
        duration = parse_duration_to_seconds(game_json.get('game_duration'))
        
        if kills1 is None or kills2 is None or not name1 or not name2 or duration is None:
            return None
        
        # Hash combines team names, kills, and duration
        return f"{name1}_{kills1}_{duration}_{kills2}_{name2}"
    except (KeyError, TypeError):
        return None


def deduplicate_games(parsed_games):
    """Remove duplicate games - often multiple frames capture the same stat screen"""
    hash_groups = defaultdict(list)
    
    for game in parsed_games:
        game_hash = create_game_hash(game)
        if game_hash:
            hash_groups[game_hash].append(game)
    
    # Keep first occurrence of each unique game
    unique_games = [games[0] for games in hash_groups.values()]
    
    return unique_games


def parse_to_json():
    stat_screens_dir = Path(config.STAT_SCREENS_DIR)
    
    if not stat_screens_dir.exists() or not list(stat_screens_dir.glob('*.jpg')):
        print(f"\nError: No stat screens found in {config.STAT_SCREENS_DIR}")
        print("Run Step 3 first: python 3_classify_frames.py")
        return False
    
    print("\n" + "="*70)
    print("STEP 4: PARSING TO JSON (VERTEX AI)")
    print("="*70)
    print(f"Workers: {config.MAX_WORKERS} parallel")
    print(f"Input: {stat_screens_dir}")
    print("="*70)
    
    image_files = list(stat_screens_dir.glob("*.jpg"))
    print(f"\nParsing {len(image_files)} stat screens...\n")
    
    parsed_games = []
    failed = 0
    
    # Parse images in parallel with thread pool
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = [executor.submit(process_image_wrapper, img) for img in image_files]
        
        with tqdm(total=len(image_files), desc="Parsing") as pbar:
            for future in as_completed(futures):
                image_path, result, success = future.result()
                
                if success and result:
                    parsed_games.append(result)
                else:
                    failed += 1
                
                pbar.update(1)
    
    print(f"\nParsed: {len(parsed_games)} games")
    print(f"Failed: {failed} images")
    
    print("\nDeduplicating games...")
    unique_games = deduplicate_games(parsed_games)
    print(f"Unique games: {len(unique_games)}")
    
    # Add tournament metadata
    for game in unique_games:
        game['tournament_name'] = config.TOURNAMENT_NAME
        game['tournament_stage'] = config.TOURNAMENT_STAGE
    
    output_path = Path(config.OUTPUT_JSON)
    with open(output_path, 'w') as f:
        json.dump(unique_games, f, indent=2)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"  Unique games: {len(unique_games)}")
    print(f"  Tournament: {config.TOURNAMENT_NAME}")
    print(f"  Stage: {config.TOURNAMENT_STAGE}")
    print(f"  Output: {output_path.absolute()}")
    print("="*70)
    
    if unique_games:
        print("\nSample game:")
        sample = unique_games[0]
        print(f"  {sample['teams'][0]['name']} vs {sample['teams'][1]['name']}")
        print(f"  Kills: {sample['teams'][0]['kills']}-{sample['teams'][1]['kills']}")
        print(f"  Duration: {sample['game_duration']}")
    
    return True


def main():
    success = parse_to_json()
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main()
