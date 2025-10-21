import cv2
import numpy as np
from pathlib import Path
import config


def extract_frames():
    video_path = Path(config.VIDEO_OUTPUT)
    frames_dir = Path(config.FRAMES_DIR)
    
    if not video_path.exists():
        print(f"\nError: {config.VIDEO_OUTPUT} not found")
        print("Run Step 1 first: python 1_download_video.py <url>")
        return False
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING FRAMES (MOTION DETECTION)")
    print("="*70)
    print(f"Video: {video_path.name}")
    print(f"Frame skip: {config.FRAME_SKIP} (analyze every {config.FRAME_SKIP}th frame)")
    print(f"Motion threshold: {config.MOTION_THRESHOLD}")
    print(f"Min static duration: {config.MIN_STATIC_DURATION} frames")
    print("="*70)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info: {total_frames} frames @ {fps:.1f} fps")
    print("Analyzing motion...\n")
    
    prev_frame_gray = None
    motion_scores = []
    frame_indices = []
    frame_idx = 0
    processed = 0
    
    # Pass 1: Find motion scores for each frame
    # Downscaled and skipped for speed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % config.FRAME_SKIP == 0:
            # Resize to 160px width for fast processing
            frame_small = cv2.resize(frame, (config.DOWNSCALE_WIDTH, 
                                            int(config.DOWNSCALE_WIDTH * frame.shape[0] / frame.shape[1])))
            frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            if prev_frame_gray is not None:
                # Motion = average pixel difference between frames
                diff = np.abs(frame_gray.astype(float) - prev_frame_gray.astype(float))
                motion_score = np.mean(diff)
                
                motion_scores.append(motion_score)
                frame_indices.append(frame_idx)
            
            prev_frame_gray = frame_gray
            processed += 1
            
            if processed % 5000 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.0f}%")
        
        frame_idx += 1
    
    cap.release()
    print(f"\nMotion analysis complete ({processed} frames analyzed)")
    
    # Pass 2: Find periods where motion stays low for a while
    # Stat screens appear at the end of games and stay static
    print(f"\nFinding static scenes (motion < {config.MOTION_THRESHOLD})...")
    
    static_periods = []
    current_static_start = None
    current_static_frames = []
    
    for idx, (frame_num, score) in enumerate(zip(frame_indices, motion_scores)):
        if score < config.MOTION_THRESHOLD:
            if current_static_start is None:
                current_static_start = frame_num
            current_static_frames.append((frame_num, score))
        else:
            # Motion detected - save static period if it lasted long enough
            if current_static_start is not None and len(current_static_frames) >= config.MIN_STATIC_DURATION:
                # Pick the most static frame from the period
                most_static_idx = np.argmin([s for _, s in current_static_frames])
                best_frame, best_score = current_static_frames[most_static_idx]
                static_periods.append((best_frame, best_score))
            
            current_static_start = None
            current_static_frames = []
    
    # Don't forget the last period if video ends on a static scene
    if current_static_start is not None and len(current_static_frames) >= config.MIN_STATIC_DURATION:
        most_static_idx = np.argmin([s for _, s in current_static_frames])
        best_frame, best_score = current_static_frames[most_static_idx]
        static_periods.append((best_frame, best_score))
    
    print(f"Found {len(static_periods)} static scenes")
    
    # Pass 3: Go back and extract those frames at full resolution
    print(f"\nExtracting frames at full resolution...")
    cap = cv2.VideoCapture(str(video_path))
    
    for idx, (frame_num, motion_score) in enumerate(static_periods):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Filename includes timestamp for debugging
            timestamp_sec = frame_num / fps
            filename = f"frame_{idx:04d}_{int(timestamp_sec)}s.jpg"
            filepath = frames_dir / filename
            cv2.imwrite(str(filepath), frame)
        
        if idx % 50 == 0 and idx > 0:
            print(f"  Extracted {idx}/{len(static_periods)} frames...")
    
    cap.release()
    
    print("\n" + "="*70)
    print(f"EXTRACTION COMPLETE: {len(static_periods)} frames")
    print(f"  Output: {frames_dir}")
    print("="*70)
    
    return True


def main():
    success = extract_frames()
    
    if success:
        print(f"\nNext step: python 3_classify_frames.py")
    else:
        exit(1)


if __name__ == '__main__':
    main()
