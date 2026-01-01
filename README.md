🐶🐱 Cat vs Dog Classifier

A Deep Learning–based Cat vs Dog Image Classifier built using Transfer Learning (MobileNetV2) and deployed with Streamlit.

🚀 Demo

Upload an image and the model predicts whether it is a Cat or a Dog with confidence.

🧠 Model Details

- Architecture: MobileNetV2 (pretrained on ImageNet)

- Technique: Transfer Learning

- Input size: 224 × 224

- Output: Binary classification (Cat / Dog)

- Accuracy: ~98% validation accuracy

🛠️ Tech Stack

- Python

- TensorFlow / Keras

- NumPy

- Streamlit

- Jupyter Notebook

📁 Project Structure

cat-vs-dog-classifier/
│
├── app.py                # Streamlit web app
├── train_model.ipynb     # Model training notebook
├── model_tf.keras        # Trained model
├── .gitignore
└── README.md

▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

⚠️ Notes on Predictions

Some incorrect predictions may occur due to:

- Lighting conditions

- Background noise

- Dataset bias

- These can be improved with stronger data augmentation and fine-tuning.

👤 Author

Hem Modi

