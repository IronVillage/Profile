# MLBB YouTube to JSON Pipeline

Automated extraction of Mobile Legends: Bang Bang game statistics from YouTube tournament VODs.

Converts tournament videos into structured JSON data using motion detection, CNN classification, and AI-powered OCR.

---

## What It Does

```
YouTube URL → 9-hour tournament stream
    ↓
games.json with structured data
```

**Example output:**
```json
[
  {
    "tournament_name": "MPL - Cambodia - Season 6",
    "tournament_stage": "Grand Finals",
    "game_duration": "15:43",
    "teams": [
      {
        "side": "left",
        "name": "RRQ",
        "kills": 18,
        "total_gold": 45234
      },
      {
        "side": "right",
        "name": "ECHO",
        "kills": 12,
        "total_gold": 38921
      }
    ]
  }
]
```



#this is meant to be a proof of concept, the other version was more complex.

Configure your vision AI provider (current implementation uses Vertex AI, easily swappable for OpenAI or similar).

## Usage

### 1. Configure

Edit `config.py`:
```python
TOURNAMENT_NAME = "MPL - Cambodia - Season 6"
TOURNAMENT_STAGE = "Grand Finals"
```

### 2. Run Pipeline

```bash
python 1_download_video.py "https://youtube.com/watch?v=VIDEO_ID"
python 2_extract_frames.py
python 3_classify_frames.py
python 4_parse_to_json.py
```

### 3. Output

Results saved to `games.json`



## Configuration

All settings in `config.py`:

```python
# Tournament metadata
TOURNAMENT_NAME = "MPL - Cambodia - Season 6"
TOURNAMENT_STAGE = "Grand Finals"

# Frame extraction
FRAME_SKIP = 8                  # Analyze every 8th frame
MOTION_THRESHOLD = 15.0         # Lower = stricter
MIN_STATIC_DURATION = 26        # ~0.87 seconds

# CNN classification
BATCH_SIZE = 64
CLASSIFICATION_THRESHOLD = 0.5
