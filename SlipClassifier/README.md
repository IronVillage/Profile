# Bet Slip Classification System

An ensemble ML model for classifying sports betting slip images. Built for esports betting analysis at scale.

## Architecture

The system uses three specialized models combined through a meta-classifier:

1. **Text Analysis (BERT + Logistic Regression)**: Extracts semantic meaning from OCR text
2. **Pattern Detection (Random Forest)**: Identifies key betting indicators and timestamps  
3. **Visual Analysis (CNN/ResNet-18)**: Processes raw image features

The meta-model combines these predictions for final classification with ~95% accuracy on validation data.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (system dependency)
# Ubuntu/Debian: apt-get install tesseract-ocr
# MacOS: brew install tesseract
# Windows: Download from GitHub releases
```

## Usage

```bash
# Basic usage
python MetaFilter.py

# Custom directories and threshold
python MetaFilter.py --src ./images --passdir ./approved --rejectdir ./rejected --thresh 0.15
```

## Docker Deployment

The container runs continuously, checking for new images every 20 seconds:

```bash
# Build
docker build -t betslip-filter .

# Run with volume mounts
docker run -v /path/to/images:/input -v /path/to/output:/app betslip-filter
```

The Dockerfile uses a simple bash loop for 24/7 operation - clean and reliable.

## Model Training

Models were trained on 10,000+ manually labeled bet slip images from sources. The ensemble approach handles:

- Varying image quality and formats
- Multiple sportsbook layouts
- Live vs pre-game bets
- Different text extraction confidence levels

## Performance

- Processes ~100 images/minute on CPU
- 95% classification accuracy
- Automatic GCS upload for approved slips
- Handles OCR failures gracefully

