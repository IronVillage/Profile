import os
import re
import shutil
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import pytesseract
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModel
import joblib


class BetSlipFilter:
    """ML-based bet slip image classifier using ensemble models."""
    
    def __init__(self, models_dir="models", threshold=0.10):
        self.models_dir = Path(models_dir)
        self.threshold = threshold
        self.device = torch.device("cpu")
        self.logger = self._setup_logger()
        
        # Model placeholders
        self.bert_tokenizer = None
        self.bert_model = None
        self.logreg_model = None
        self.rf_model = None
        self.cnn_model = None
        self.meta_model = None
        
        # Image transforms for CNN
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self._configure_tesseract()
        self._load_models()
    
    def _setup_logger(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_dir / "meta_filter_activity.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f"META_FILTER_BATCH-{int(datetime.now().timestamp())}")
    
    def log_metric(self, component, metric_type, value, metadata=None):
        """Log structured metrics for dashboard parsing."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": "crowdslips",
            "component": component,
            "metric_type": metric_type,
            "value": value,
            "metadata": metadata or {}
        }
        print(f"METRIC: {json.dumps(log_data)}")
    
    def get_source_info(self, filename):
        """Extract source platform and channel info from filename."""
        filename_lower = filename.lower()
        parts = filename_lower.split('_')
        
        if len(parts) >= 3 and parts[0] in ['discord', 'twitter', 'reddit']:
            return {
                'source': parts[0],
                'channel_handle': parts[1],
                'identifier': '_'.join(parts[2:]).split('.')[0]
            }
        
        # Fallback
        if filename_lower.startswith('discord_'):
            source = 'discord'
        elif filename_lower.startswith('twitter_'):
            source = 'twitter'
        elif filename_lower.startswith('reddit_'):
            source = 'reddit'
        else:
            source = 'unknown'
        
        return {
            'source': source,
            'channel_handle': 'unknown',
            'identifier': filename.split('.')[0]
        }
    
    def _configure_tesseract(self):
        """Configure Tesseract OCR path if needed."""
        if os.name == 'nt':  # Windows
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def _load_models(self):
        """Load all ML models."""
        try:
            # BERT for text embeddings
            self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.bert_model.to(self.device).eval()
            
            # Sklearn models
            self.logreg_model = joblib.load(self.models_dir / "logreg_embed_model.pkl")
            self.rf_model = joblib.load(self.models_dir / "rf_model.pkl")
            self.meta_model = joblib.load(self.models_dir / "meta_model.pkl")
            
            # CNN model
            self.cnn_model = resnet18(weights=None)
            self.cnn_model.fc = torch.nn.Linear(self.cnn_model.fc.in_features, 1)
            self.cnn_model.load_state_dict(
                torch.load(self.models_dir / "cnn_model.pth", map_location=self.device)
            )
            self.cnn_model.to(self.device).eval()
            
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_text(self, image_path):
        """Extract text from image using OCR."""
        try:
            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            text_blocks = [t.strip() for t in data["text"] if t.strip()]
            text = " ".join(text_blocks).lower()
            
            confidences = [float(c) for c in data["conf"] if float(c) >= 0]
            confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            if not text:
                text = pytesseract.image_to_string(img).strip().lower()
            
            return text, confidence
        except Exception as e:
            self.logger.warning(f"OCR failed for {image_path}: {e}")
            return "", 0.0
    
    def get_text_features(self, text):
        """Extract rule-based features from text."""
        features = [
            len(text) < 50,
            bool(re.search(r'\b(starts?\s+in|starting\s+soon)\b', text, re.I)),
            bool(re.search(r'\b(final|live|q[1-4])\b', text, re.I)),
            bool(re.search(r'\b(final|live)\b', text, re.I)) and not bool(re.search(r'\bstarts?\b', text, re.I)),
            bool(re.search(r'\b(more|less|higher|lower|under|over)\b', text, re.I)),
            bool(re.search(r'\d{1,2}:\d{2}|\b(mon|tue|wed|thu|fri|sat|sun)\b', text, re.I)),
        ]
        return np.array(features, dtype=float)
    
    def predict_bert(self, text):
        """Get BERT-based prediction."""
        if not text:
            return 0.0
        
        try:
            with torch.no_grad():
                inputs = self.bert_tokenizer(
                    text, truncation=True, padding=True, 
                    max_length=128, return_tensors="pt"
                ).to(self.device)
                
                embedding = self.bert_model(**inputs).last_hidden_state[:, 0, :]
                embedding = embedding.cpu().numpy()
                
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                
                return float(self.logreg_model.predict_proba(embedding)[0, 1])
        except Exception:
            return 0.0
    
    def predict_rf(self, text, confidence):
        """Get Random Forest prediction."""
        try:
            text_features = self.get_text_features(text)
            features = np.append(text_features, confidence).reshape(1, -1)
            return float(self.rf_model.predict_proba(features)[0, 1])
        except Exception:
            return 0.0
    
    def predict_cnn(self, image_path):
        """Get CNN prediction."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.cnn_model(image_tensor).view(-1)
                return float(torch.sigmoid(logits).cpu().numpy()[0])
        except Exception:
            return 0.0
    
    def classify(self, image_path):
        """Classify image using ensemble of models."""
        # OCR
        text, confidence = self.extract_text(image_path)
        
        # Individual model predictions
        bert_prob = self.predict_bert(text)
        rf_prob = self.predict_rf(text, confidence)
        cnn_prob = self.predict_cnn(image_path)
        
        # Meta model combination
        try:
            meta_features = np.array([[bert_prob, rf_prob, cnn_prob]])
            final_prob = float(self.meta_model.predict_proba(meta_features)[0, 1])
        except Exception:
            final_prob = np.mean([bert_prob, rf_prob, cnn_prob])
        
        return {
            'probability': final_prob,
            'passed': final_prob >= self.threshold,
            'scores': {
                'bert': bert_prob,
                'rf': rf_prob,
                'cnn': cnn_prob
            }
        }
    
    def upload_to_gcs(self, file_path, blob_name):
        """Upload file to Google Cloud Storage."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket('crowdslips-images')
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            self.logger.info(f"Uploaded {blob_name} to GCS")
            return f"gs://crowdslips-images/{blob_name}"
        except Exception as e:
            self.logger.warning(f"GCS upload failed for {blob_name}: {e}")
            return None
    
    def process_batch(self, source_dir, pass_dir, reject_dir):
        """Process all images in source directory."""
        source_dir = Path(source_dir)
        pass_dir = Path(pass_dir)
        reject_dir = Path(reject_dir)
        
        # Create output directories
        pass_dir.mkdir(parents=True, exist_ok=True)
        reject_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        metrics = {
            "start_time": datetime.now().isoformat(),
            "total_files_scanned": 0,
            "images_processed": 0,
            "images_passed_model": 0,
            "images_rejected_model": 0,
            "images_gcs_uploaded": 0,
            "ocr_errors_catastrophic": 0,
            "gcs_upload_errors": 0,
            "error_details": []
        }
        
        source_metrics = {
            'discord': {'processed': 0, 'approved': 0, 'rejected': 0},
            'twitter': {'processed': 0, 'approved': 0, 'rejected': 0},
            'reddit': {'processed': 0, 'approved': 0, 'rejected': 0},
            'unknown': {'processed': 0, 'approved': 0, 'rejected': 0}
        }
        
        # Clean up GIFs and collect images
        image_extensions = {'.png', '.jpg', '.jpeg'}
        images = []
        gifs_deleted = 0
        
        for file in source_dir.iterdir():
            if file.suffix.lower() == '.gif':
                try:
                    file.unlink()
                    gifs_deleted += 1
                except Exception as e:
                    self.logger.warning(f"Could not delete {file.name}: {e}")
            elif file.suffix.lower() in image_extensions:
                images.append(file)
        
        if gifs_deleted > 0:
            self.logger.info(f"Deleted {gifs_deleted} GIF files")
        
        # Sort by modification time
        images.sort(key=lambda x: x.stat().st_mtime)
        
        metrics["total_files_scanned"] = len(images)
        self.logger.info(f"Found {len(images)} images to process")
        
        # Process each image
        for image_path in images:
            filename = image_path.name
            self.logger.info(f"Processing: {filename}")
            metrics["images_processed"] += 1
            
            # Track source
            source_info = self.get_source_info(filename)
            source = source_info['source']
            source_metrics[source]['processed'] += 1
            
            # Classify
            result = self.classify(image_path)
            
            self.logger.info(
                f"{filename}: BERT={result['scores']['bert']:.3f}, "
                f"RF={result['scores']['rf']:.3f}, CNN={result['scores']['cnn']:.3f} "
                f"-> Meta={result['probability']:.3f}"
            )
            
            if result['passed']:
                metrics["images_passed_model"] += 1
                source_metrics[source]['approved'] += 1
                
                # Log approval metric
                self.log_metric("meta_filter", "image_approved", 1, {
                    "source": source_info['source'],
                    "channel_handle": source_info['channel_handle'],
                    "filename": filename,
                    "probability": round(result['probability'], 3)
                })
                
                # Upload to GCS
                gcs_uri = self.upload_to_gcs(image_path, filename)
                if gcs_uri:
                    metrics["images_gcs_uploaded"] += 1
                    destination = pass_dir / filename
                else:
                    metrics["gcs_upload_errors"] += 1
                    metrics["error_details"].append(f"GCS upload failed: {filename}")
                    destination = reject_dir / f"gcs_fail_{filename}"
                
                self.logger.info(f"{filename} - PASS (Prob: {result['probability']:.3f})")
            else:
                metrics["images_rejected_model"] += 1
                source_metrics[source]['rejected'] += 1
                destination = reject_dir / filename
                
                # Log rejection metric
                self.log_metric("meta_filter", "image_rejected", 1, {
                    "source": source_info['source'],
                    "channel_handle": source_info['channel_handle'],
                    "filename": filename,
                    "probability": round(result['probability'], 3)
                })
                
                self.logger.info(f"{filename} - REJECT (Prob: {result['probability']:.3f})")
            
            # Move file
            try:
                shutil.move(str(image_path), str(destination))
            except Exception as e:
                self.logger.error(f"Error moving {filename}: {e}")
                metrics["error_details"].append(f"Move error: {filename}")
        
        # Final metrics
        metrics["end_time"] = datetime.now().isoformat()
        metrics["duration_seconds"] = round(
            (datetime.fromisoformat(metrics["end_time"]) - 
             datetime.fromisoformat(metrics["start_time"])).total_seconds(), 2
        )
        
        # Log source breakdown
        for source, data in source_metrics.items():
            if data['processed'] > 0:
                self.log_metric("meta_filter", "source_breakdown", data['processed'], {
                    "source": source,
                    "approved": data['approved'],
                    "rejected": data['rejected'],
                    "approval_rate": round(data['approved'] / data['processed'], 3)
                })
        
        # Log batch complete
        self.log_metric("meta_filter", "batch_complete", metrics["images_processed"], {
            "total_approved": metrics["images_passed_model"],
            "total_rejected": metrics["images_rejected_model"],
            "gcs_uploaded": metrics["images_gcs_uploaded"],
            "duration_seconds": metrics["duration_seconds"]
        })
        
        # Print summary
        self.logger.info("\n--- Meta Filter Metrics ---")
        for key, value in metrics.items():
            if key != "error_details":
                self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        if metrics["error_details"]:
            self.logger.info(f"Errors: {len(metrics['error_details'])}")
            for err in metrics["error_details"][:5]:
                self.logger.info(f"  - {err}")
        
        self.logger.info("Processing complete")
        return metrics
    


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ML-based bet slip image filter")
    parser.add_argument("--src", default="../images/", help="Source directory")
    parser.add_argument("--passdir", default="../approved/", help="Approved images directory")
    parser.add_argument("--rejectdir", default="../blocked/", help="Rejected images directory")
    parser.add_argument("--thresh", type=float, default=0.10, help="Decision threshold")
    
    args = parser.parse_args()
    
    filter = BetSlipFilter(threshold=args.thresh)
    filter.process_batch(args.src, args.passdir, args.rejectdir)


if __name__ == "__main__":
    main()