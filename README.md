## Brain Tumor Detection using CNN:

This is a web-based application that detects brain tumors from MRI images using a Convolutional Neural Network (CNN). Built with Streamlit, this app allows users to upload brain scans and receive real-time predictions about the presence of a tumor.


## Dataset Link:
 https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

## Features:

 Upload an MRI image and receive a tumor/no-tumor prediction.

 Clean and interactive UI built with Streamlit.

 CNN-based deep learning model trained on MRI scan datasets.

 Preprocessing and resizing images handled seamlessly.

 Model accuracy ~98% on test data.

 Model saved using .h5 format and loaded in real-time.

 Includes confidence score for each prediction.

## How to Run:

1. Clone the Repository

git clone https://github.com/yourusername/Brain-Tumor-Detection.git

cd Brain-Tumor-Detection

2. (Optional) Create & Activate a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS/Linux

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the App

streamlit run app.py

##  Model Overview:

Model Type: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Input: MRI brain image (JPG, PNG)

Output: Tumor / No Tumor + with confidence score

File Format: Model saved as brain_tumor_cnn_model.h5

📈 Performance:
✅ Accuracy: 88%

📉 Loss: 0.02

📊 Evaluated on unseen validation and test datasets


## Required Packages:

streamlit

tensorflow

numpy

Pillow

scikit-learn 

## Install with:

pip install -r requirements.txt

## Future Improvements:

🔍 Add Grad-CAM visualization for explainable AI

📦 Deploy on Streamlit Cloud, Heroku, or Render

🔄 Allow users to upload datasets and retrain the model

📘 Show detailed prediction logs

🧾 Export prediction as PDF report

## License:

This project is licensed under the MIT License.

## Author:

Developed by Sathish B
GitHub: https://github.com/SathishB-1/Brain-Tumor-Detection-using-CNN

## User Interface:
 ![img Alt]()