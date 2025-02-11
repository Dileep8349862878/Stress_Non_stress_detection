from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import streamlit as st
import requests
from PIL import Image


app = Flask(__name__)

# Load the trained model
model = load_model('stress_detection_model.h5')

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    """Preprocess the image for model prediction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"Error": "No file provided"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"Error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)

    # Model prediction
    prediction = model.predict(img_array)
    result = "Stress" if prediction[0][0] > 0.5 else "Not Stress"

    # Clean up
    os.remove(file_path)

    return jsonify({'prediction': result})




# Streamlit app
st.title('Stress Detection from Image')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Send the image to the Flask app
    files = {'file': uploaded_file.getvalue()}
    try:
        response = requests.post('http://127.0.0.1:5000/predict', files=files)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: {result['prediction']}")
        else:
            st.write(f"Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the server. Ensure the Flask app is running.")



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)  # Run Flask server
