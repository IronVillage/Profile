# Tournament metadata
TOURNAMENT_NAME = "MPL - Cambodia - Season 6"
TOURNAMENT_STAGE = "Grand Finals"

# Paths
VIDEO_OUTPUT = "video.mp4"
FRAMES_DIR = "frames/"
STAT_SCREENS_DIR = "stat_screens/"
OUTPUT_JSON = "games.json"
CNN_MODEL_PATH = "models/stat_screen_classifier.pth"

# Video download
VIDEO_QUALITY = 360  # 360p for speed
YOUTUBE_COOKIES = "youtube_cookies.txt"

# Frame extraction (motion detection)
FRAME_SKIP = 8  # Analyze every 8th frame
MOTION_THRESHOLD = 15.0  # Lower = stricter static detection
MIN_STATIC_DURATION = 26  # ~0.87 seconds at 30fps
DOWNSCALE_WIDTH = 160  # Downscale for fast motion analysis

# CNN classification
BATCH_SIZE = 64
CLASSIFICATION_THRESHOLD = 0.5

# Vertex AI
VERTEX_PROJECT_ID = "1020774583673"
VERTEX_LOCATION = "us-central1"
VERTEX_MODEL_ENDPOINT = "projects/1020774583673/locations/us-central1/endpoints/8984112809092579328"
MAX_WORKERS = 10
