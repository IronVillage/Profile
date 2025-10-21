import sys
import subprocess
from pathlib import Path
import config


def download_video(url):
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING VIDEO")
    print("="*70)
    print(f"URL: {url}")
    print(f"Output: {config.VIDEO_OUTPUT}")
    print(f"Quality: {config.VIDEO_QUALITY}p (video only, no audio)")
    print("="*70)
    
    cmd = [
        'yt-dlp',
        '-f', f'best[height<={config.VIDEO_QUALITY}]/best',
        '-o', config.VIDEO_OUTPUT,
        url,
        '--no-playlist',
    ]
    
    # Add cookies if available (for age-restricted or private videos)
    cookies_path = Path(config.YOUTUBE_COOKIES)
    if cookies_path.exists():
        cmd.extend(['--cookies', str(cookies_path)])
        print(f"Using cookies: {config.YOUTUBE_COOKIES}")
    
    # Use aria2c for parallel download (16 connections)
    cmd.extend([
        '--downloader', 'aria2c',
        '--downloader-args', 'aria2c:-x 16 -s 16 -k 1M'
    ])
    
    print("\nDownloading...")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        return True
        
    except subprocess.CalledProcessError:
        print("\n" + "="*70)
        print("DOWNLOAD FAILED")
        print("="*70)
        print("\nPossible solutions:")
        print("1. Export YouTube cookies:")
        print("   - Install 'Get cookies.txt LOCALLY' browser extension")
        print("   - Go to YouTube.com (logged in)")
        print("   - Export cookies and save as: youtube_cookies.txt")
        print("2. Check if URL is valid and video is public")
        return False


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python 1_download_video.py <youtube_url>")
        print("\nExample:")
        print('  python 1_download_video.py "https://www.youtube.com/watch?v=FcmQcRCzjIw"')
        sys.exit(1)
    
    url = sys.argv[1]
    success = download_video(url)
    
    if success:
        print(f"\nNext step: python 2_extract_frames.py")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
