import os
import json
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Page configuration (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Plant Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# 2. Load class indices
try:
    with open(class_indices_path) as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)
except FileNotFoundError as e:
    st.error(f"Class indices file missing: {str(e)}")
    st.stop()

# 3. Model loading with cache
@st.cache_resource
def load_model(model_path, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    try:
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"""
        Failed to load weights. Please verify:
        1. File exists at: {model_path}
        2. Architecture matches exactly
        3. TensorFlow version is compatible
        Error: {str(e)}
        """)
        st.stop()

# Load the model
model = load_model(model_path, num_classes)

# 4. Image processing
def load_and_preprocess_image(image_data, target_size=(224, 224)):
    try:
        img = Image.open(image_data).convert('RGB').resize(target_size)
        return np.expand_dims(np.array(img), axis=0).astype('float32') / 255.0
    except Exception as e:
        st.error(f"Image error: {str(e)}")
        return None

# 5. Prediction function
def predict_image_class(model, image_data, class_indices):
    start = time.time()
    try:
        img_array = load_and_preprocess_image(image_data)
        if img_array is None:
            return None, 0, 0
            
        preds = model.predict(img_array, verbose=0)
        elapsed = time.time() - start
        
        pred_idx = np.argmax(preds[0])
        confidence = np.max(preds) * 100
        class_name = class_indices.get(str(pred_idx), "Unknown")
        
        return class_name, confidence, round(elapsed, 2)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0, 0

# 6. Main App
def main():
    st.title('🌿 Plant Disease Classifier')
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background: #f0f2f6;
        color: #333333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card h3 {
        color: #2e8b57;
        margin-top: 0;
    }
    .prediction-card p {
        margin: 8px 0;
        font-size: 16px;
    }
    .high-confidence {
        color: #2e8b57;
        font-weight: bold;
    }
    .medium-confidence {
        color: #ffa500;
        font-weight: bold;
    }
    .low-confidence {
        color: #ff4500;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Leaf Image")
        uploaded_image = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Upload a clear photo of a plant leaf for disease diagnosis"
        )
        
        if uploaded_image:
            st.image(
                Image.open(uploaded_image).resize((300, 300)),
                caption="Uploaded Image",
                use_column_width=True
            )

    with col2:
        if uploaded_image:
            if st.button('🔍 Analyze', type="primary", use_container_width=True):
                with st.spinner('Analyzing leaf image...'):
                    class_name, confidence, elapsed = predict_image_class(
                        model, uploaded_image, class_indices
                    )
                
                if class_name:
                    # Display results
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Diagnosis Results</h3>
                        <p><b>Condition:</b> <span style="color: #2e8b57">{class_name}</span></p>
                        <p><b>Confidence:</b> <span style="font-weight:bold">{confidence:.1f}%</span></p>
                        <p><b>Processing Time:</b> {elapsed}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence visualization
                    st.progress(int(confidence))
                    
                    # Confidence message with custom styling
                    if confidence > 70:
                        st.markdown('<p class="high-confidence">✓ High confidence (Reliable diagnosis)</p>', 
                                  unsafe_allow_html=True)
                    elif confidence > 40:
                        st.markdown('<p class="medium-confidence">↻ Moderate confidence (Consider verification)</p>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="low-confidence">⚠ Low confidence (Upload clearer image)</p>', 
                                  unsafe_allow_html=True)
                    
                    # Additional recommendations
                    with st.expander("💡 Recommendations"):
                        if confidence > 70:
                            st.success("Suggested treatment options would appear here")
                        elif confidence > 40:
                            st.warning("The diagnosis is uncertain. Please provide additional photos from different angles.")
                        else:
                            st.error("Unable to make reliable diagnosis. Please ensure:")
                            st.write("- The leaf is well-lit and in focus")
                            st.write("- The image shows clear symptoms")
                            st.write("- The leaf fills most of the frame")

if __name__ == "__main__":
    main()