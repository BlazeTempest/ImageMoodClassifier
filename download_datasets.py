"""
Download diverse mood-based images from Pexels API
No authentication key required for basic searches
"""
import os
import requests
import time
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import random


# Mood-based search queries (diverse keywords)
MOOD_KEYWORDS = {
    'Calm': [
        'peaceful landscape', 'ocean waves', 'blue sky', 'meditation',
        'serene nature', 'lake', 'sunrise', 'calm water', 'zen garden',
        'quiet forest', 'still life', 'pastel colors', 'soft light'
    ],
    'Energetic': [
        'dynamic action', 'vibrant colors', 'bright neon', 'dancing',
        'sports movement', 'city lights', 'excitement', 'fast motion',
        'colorful abstract', 'electric', 'party', 'active people',
        'explosion of colors'
    ],
    'Warm': [
        'sunset', 'fire', 'warm tones', 'golden hour', 'autumn leaves',
        'hot colors', 'warm lighting', 'cozy room', 'desert sand',
        'orange sky', 'fireplace', 'warm embrace', 'heated atmosphere'
    ],
    'Dark': [
        'night sky', 'dark forest', 'storm clouds', 'shadow', 'midnight',
        'dark mood', 'moody atmosphere', 'black background', 'dim light',
        'cave', 'dark water', 'gloomy', 'noir'
    ],
    'Soft': [
        'pastel colors', 'soft focus', 'gentle', 'blurred', 'delicate',
        'light pink', 'light blue', 'dreamy', 'misty morning', 'gentle waves',
        'soft texture', 'cotton candy', 'watercolor', 'fog'
    ]
}

# Pexels API endpoint (free, no key needed for basic searches)
PEXELS_API_URL = "https://api.pexels.com/v1/search"
PEXELS_API_KEY = "yR8BSmVQ6C3gwKye8oKa7mcMe8GFORW0fsvVxhT2sjmVb86eiWgv6cls"

def download_image_from_url(url, save_path):
    """
    Download image from URL and save to file
    
    Args:
        url: Image URL
        save_path: Path to save the image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Open and resize image
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(save_path, 'JPEG', quality=95)
            return True
    except Exception as e:
        print(f"  ✗ Error downloading {url}: {e}")
        return False


def search_and_download_mood_images(mood, keywords, images_per_mood=150):
    """
    Search and download images for a specific mood
    
    Args:
        mood: Mood category name
        keywords: List of search keywords for this mood
        images_per_mood: Number of images to download
    
    Returns:
        Number of images successfully downloaded
    """
    mood_folder = f'dataset/{mood}'
    os.makedirs(mood_folder, exist_ok=True)
    
    downloaded_count = 0
    page = 1
    max_pages = 8  # Limit pages to avoid too many requests
    
    print(f"\n{'='*60}")
    print(f"Downloading images for: {mood} mood")
    print(f"{'='*60}")
    
    while downloaded_count < images_per_mood and page <= max_pages:
        # Randomly select a keyword for this page
        keyword = random.choice(keywords)
        
        print(f"\nPage {page} - Searching for: '{keyword}'")
        
        try:
            # Make API request (Pexels allows free access without key for basic searches)
            params = {
                'query': keyword,
                'per_page': 80,
                'page': page,
                'orientation': 'landscape'
            }
            
            headers = {'Authorization': PEXELS_API_KEY}
            response = requests.get(PEXELS_API_URL, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"  ✗ API request failed: {response.status_code}")
                page += 1
                continue
            
            data = response.json()
            photos = data.get('photos', [])
            
            if not photos:
                print(f"  ✗ No images found for '{keyword}'")
                page += 1
                continue
            
            print(f"  Found {len(photos)} images")
            
            # Download images
            for idx, photo in enumerate(tqdm(photos, desc=f"  Downloading")):
                if downloaded_count >= images_per_mood:
                    break
                
                # Use medium quality image URL
                image_url = photo['src']['medium']
                image_filename = f"{mood}_{page}_{idx}.jpg"
                save_path = os.path.join(mood_folder, image_filename)
                
                if download_image_from_url(image_url, save_path):
                    downloaded_count += 1
                
                # Rate limiting: be respectful to the API
                time.sleep(0.1)
            
            page += 1
            
            # Respectful delay between pages
            time.sleep(1)
            
        except Exception as e:
            print(f"  ✗ Error during search: {e}")
            page += 1
            continue
    
    print(f"\n✓ Downloaded {downloaded_count} images for {mood}")
    return downloaded_count


def verify_dataset():
    """
    Verify that dataset was downloaded correctly
    """
    dataset_root = 'dataset'
    mood_classes = list(MOOD_KEYWORDS.keys())
    
    if not os.path.exists(dataset_root):
        print("\n✗ Error: dataset/ folder not found!")
        return False
    
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    all_good = True
    total_images = 0
    
    for mood in mood_classes:
        mood_path = f'{dataset_root}/{mood}'
        if not os.path.exists(mood_path):
            print(f"✗ Missing folder: {mood_path}")
            all_good = False
        else:
            image_count = len([f for f in os.listdir(mood_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if image_count == 0:
                print(f"✗ No images in: {mood_path}")
                all_good = False
            else:
                print(f"✓ {mood:12s}: {image_count:4d} images")
                total_images += image_count
    
    print(f"\n✓ Total images: {total_images}")
    return all_good


def main():
    """
    Main function to download all mood-based images
    """
    print("\n" + "="*60)
    print("Image Mood Dataset Downloader (Pexels API)")
    print("="*60)
    print("\nThis script will download diverse images for mood classification.")
    print("Each mood category will have varied scenes, colors, and styles.\n")
    
    # Clean and create dataset folder
    if os.path.exists('dataset'):
        print("Cleaning existing dataset folder...")
        import shutil
        shutil.rmtree('dataset')
    
    os.makedirs('dataset', exist_ok=True)
    
    # Download images for each mood
    total_downloaded = 0
    images_per_mood = 150
    
    for mood, keywords in MOOD_KEYWORDS.items():
        count = search_and_download_mood_images(mood, keywords, images_per_mood)
        total_downloaded += count
    
    # Verify dataset
    print("\n")
    if verify_dataset():
        print("\n" + "="*60)
        print("✓ Dataset download complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. python feature_extraction.py")
        print("2. python train_ml.py")
        print("3. python train_cnn.py")
        print("4. python app.py")
        print("="*60)
    else:
        print("\n✗ Dataset verification failed!")
        print("Please check errors above and retry.")


if __name__ == '__main__':
    main()
