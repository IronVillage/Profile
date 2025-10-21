import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
import config


def load_model(device):
    model_path = Path(config.CNN_MODEL_PATH)
    
    if not model_path.exists():
        print(f"\nError: Model not found at {config.CNN_MODEL_PATH}")
        print("\nExpected model file:")
        print(f"  {model_path.absolute()}")
        return None
    
    # ResNet18 modified for binary classification (stat screen or not)
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    return model


def classify_frames():
    frames_dir = Path(config.FRAMES_DIR)
    stat_screens_dir = Path(config.STAT_SCREENS_DIR)
    
    if not frames_dir.exists() or not list(frames_dir.glob('*.jpg')):
        print(f"\nError: No frames found in {config.FRAMES_DIR}")
        print("Run Step 2 first: python 2_extract_frames.py")
        return False
    
    stat_screens_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 3: CLASSIFYING FRAMES (CNN)")
    print("="*70)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Threshold: {config.CLASSIFICATION_THRESHOLD}")
    print("="*70)
    
    print("\nLoading CNN model...")
    model = load_model(device)
    if model is None:
        return False
    
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_files = list(frames_dir.glob('*.jpg'))
    print(f"\nClassifying {len(image_files)} frames...\n")
    
    stat_count = 0
    
    # Process in batches for GPU efficiency
    for i in tqdm(range(0, len(image_files), config.BATCH_SIZE), desc="Processing"):
        batch_files = image_files[i:i+config.BATCH_SIZE]
        batch_images = []
        valid_files = []
        
        for img_path in batch_files:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            batch_images.append(image_tensor)
            valid_files.append(img_path)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # Copy stat screens to output folder
        for img_path, probability in zip(valid_files, probabilities):
            if probability >= config.CLASSIFICATION_THRESHOLD:
                dest = stat_screens_dir / img_path.name
                shutil.copy2(img_path, dest)
                stat_count += 1
    
    print("\n" + "="*70)
    print(f"CLASSIFICATION COMPLETE")
    print(f"  Total frames: {len(image_files)}")
    print(f"  Stat screens: {stat_count} ({stat_count/len(image_files)*100:.1f}%)")
    print(f"  Output: {stat_screens_dir}")
    print("="*70)
    
    return True


def main():
    success = classify_frames()
    
    if success:
        print(f"\nNext step: python 4_parse_to_json.py")
    else:
        exit(1)


if __name__ == '__main__':
    main()
