"""
Train machine learning model (Random Forest) for mood classification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


def train_ml_model(features_csv='trainedModels/features.csv', 
                   model_path='trainedModels/mood_classifier_rf.pkl',
                   label_encoder_path='trainedModels/label_encoder.pkl'):
    """
    Train Random Forest classifier on extracted features
    
    Args:
        features_csv: Path to features CSV file
        model_path: Path to save trained model
        label_encoder_path: Path to save label encoder
    """
    print("Loading features from CSV...")
    df = pd.read_csv(features_csv)
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    print(f"\nTop 10 important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx in top_indices:
        print(f"  feature_{idx}: {feature_importance[idx]:.4f}")
    
    # Save model and label encoder
    joblib.dump(rf_model, model_path)
    joblib.dump(label_encoder, label_encoder_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {label_encoder_path}")
    
    return rf_model, label_encoder


if __name__ == '__main__':
    import os
    
    if not os.path.exists('trainedModels/features.csv'):
        print("Error: features.csv not found!")
        print("Please run feature_extraction.py first to generate the feature dataset.")
    else:
        model, encoder = train_ml_model()
