Image Mood Classifier – How to Run

1. Clone the Project
--------------------
Download or clone your project to your desired folder:
    git clone https://github.com/your-username/image-mood-classifier.git
    cd image-mood-classifier

2. Set Up Python Environment
----------------------------
Create and activate a virtual environment:
    python -m venv venv
    # For Windows:
    venv\Scripts\activate
    # For Mac/Linux:
    source venv/bin/activate

3. Install Required Libraries
----------------------------
Install dependencies:
    pip install -r requirements.txt
(Requires Python 3.12 or compatible.)

4. Prepare the Dataset
----------------------
Download and organize the dataset:
    python download_dataset.py
(Enter your Pexels API key if prompted.)

5. Extract Image Features
-------------------------
Build the feature dataset:
    python feature_extraction.py
(Creates features.csv for ML step.)

6. Train the Models
-------------------
Train models sequentially:
    python train_ml.py     # Random Forest – outputs .pkl
    python train_cnn.py    # MobileNetV2 CNN – outputs .h5

7. Launch the Flask App
-----------------------
Start the web backend:
    python app.py
Look for:
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

8. Use the Web Interface
------------------------
- Open your browser at http://localhost:5000
- Upload any image (jpg, png, bmp; ≤16MB)
- View mood prediction & color result

Notes & Troubleshooting
-----------------------
- Ensure .pkl and .h5 models are in the project directory before running app.py
- For GPU training: verify your TensorFlow and drivers
- Edit config.py for custom settings if needed

Done! Your app is ready for local testing and demo.
