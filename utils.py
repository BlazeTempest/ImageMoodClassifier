"""
Image loading and preprocessing utilities
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image using OpenCV
    
    Args:
        image_path: Path to image file
        target_size: Tuple of (width, height) for resizing
    
    Returns:
        Tuple of (original_rgb, resized_rgb, hsv, grayscale)
    """
    # Load image in BGR format
    img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    return img_rgb, img_resized, img_hsv, img_gray


def extract_hsv_histogram(hsv_image, bins=(8, 8, 8)):
    """
    Extract HSV color histogram features
    
    Args:
        hsv_image: HSV format image
        bins: Number of bins for each HSV channel
    
    Returns:
        Flattened histogram array
    """
    # Calculate 3D histogram for HSV channels
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, 
                        [0, 180, 0, 256, 0, 256])
    
    # Normalize histogram
    cv2.normalize(hist, hist)
    
    # Flatten to 1D array
    return hist.flatten()


def extract_lbp_features(gray_image, num_points=24, radius=8):
    """
    Extract Local Binary Pattern texture features using scikit-image
    
    Args:
        gray_image: Grayscale image
        num_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
    
    Returns:
        LBP histogram feature vector
    """
    # Compute LBP
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    
    # Calculate histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_dominant_colors(rgb_image, n_colors=3):
    """
    Extract dominant colors using KMeans clustering
    
    Args:
        rgb_image: RGB format image
        n_colors: Number of dominant colors to extract
    
    Returns:
        Array of dominant RGB colors
    """
    # Reshape image to list of pixels
    pixels = rgb_image.reshape(-1, 3)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors


def extract_all_features(image_path):
    """
    Extract all features from an image
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary containing all features
    """
    # Load and preprocess
    _, img_resized, img_hsv, img_gray = load_and_preprocess_image(image_path)
    
    # Extract HSV histogram
    hsv_hist = extract_hsv_histogram(img_hsv)
    
    # Extract LBP features
    lbp_hist = extract_lbp_features(img_gray)
    
    # Extract dominant colors
    dominant_colors = extract_dominant_colors(img_resized)
    
    # Combine all features into single vector
    feature_vector = np.concatenate([hsv_hist, lbp_hist])
    
    return {
        'features': feature_vector,
        'dominant_colors': dominant_colors,
        'hsv_hist_length': len(hsv_hist),
        'lbp_hist_length': len(lbp_hist)
    }


def load_image_from_bytes(image_bytes, target_size=(224, 224)):
    """
    Load image from bytes (for Flask file upload)
    
    Args:
        image_bytes: Image data in bytes
        target_size: Tuple of (width, height)
    
    Returns:
        Tuple of (resized_rgb, hsv, grayscale)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image from bytes")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Convert to HSV
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    return img_resized, img_hsv, img_gray
