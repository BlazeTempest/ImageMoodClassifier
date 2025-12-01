"""
Build feature dataset from image folders
Run this script to generate features.csv from dataset/
"""
import os
import pandas as pd
import numpy as np
from utils import extract_all_features
from tqdm import tqdm


def build_feature_dataset(dataset_path='dataset', output_csv='features.csv'):
    """
    Extract features from all images in dataset folders
    
    Args:
        dataset_path: Root path containing mood class folders
        output_csv: Output CSV file path
    """
    mood_classes = ['Calm', 'Energetic', 'Warm', 'Dark', 'Soft']
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    print("Extracting features from dataset...")
    
    for mood_class in mood_classes:
        class_path = os.path.join(dataset_path, mood_class)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist. Skipping.")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Processing {len(image_files)} images from {mood_class}...")
        
        for img_file in tqdm(image_files, desc=mood_class):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Extract features
                result = extract_all_features(img_path)
                feature_vector = result['features']
                
                all_features.append(feature_vector)
                all_labels.append(mood_class)
                all_filenames.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Convert to DataFrame
    features_array = np.array(all_features)
    
    # Create column names
    feature_columns = [f'feature_{i}' for i in range(features_array.shape[1])]
    
    df = pd.DataFrame(features_array, columns=feature_columns)
    df['label'] = all_labels
    df['filename'] = all_filenames
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nFeature extraction complete!")
    print(f"Total samples: {len(df)}")
    print(f"Features per sample: {features_array.shape[1]}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Saved to {output_csv}")
    
    return df


if __name__ == '__main__':
    # Create sample dataset structure if it doesn't exist
    dataset_path = 'dataset'
    mood_classes = ['Calm', 'Energetic', 'Warm', 'Dark', 'Soft']
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        for mood in mood_classes:
            os.makedirs(os.path.join(dataset_path, mood), exist_ok=True)
        print(f"Created dataset structure at {dataset_path}/")
        print("Please add images to each mood folder before running this script.")
    else:
        # Build feature dataset
        df = build_feature_dataset(dataset_path=dataset_path, 
                                   output_csv='features.csv')
