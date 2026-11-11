import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="EcoVision AI | Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #071713, #0B211B, #102820);
    color: #F0FDF4;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header */

.hero {
    background: linear-gradient(120deg, #123D2D, #0D2921);
    border: 1px solid #245C43;
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    color: #86EFAC;
    margin-bottom: 8px;
}

.hero p {
    font-size: 16px;
    color: #C4DCCB;
}

/* Section headings */

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #86EFAC;
    margin: 22px 0 15px 0;
}

/* Cards */

.info-card {
    background: rgba(20, 55, 41, 0.85);
    border: 1px solid #285C43;
    border-radius: 17px;
    padding: 22px;
    min-height: 135px;
}

.info-card h3 {
    color: #A7F3D0;
    font-size: 18px;
}

.info-card p {
    color: #D1E7D8;
    font-size: 14px;
}

/* Result */

.result-card {
    background: linear-gradient(135deg, #14532D, #166534);
    border: 1px solid #4ADE80;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-top: 15px;
}

.result-card h2 {
    color: #DCFCE7;
    font-size: 30px;
}

.result-card h1 {
    color: white;
    font-size: 38px;
    margin: 8px 0;
}

/* Upload section */

[data-testid="stFileUploader"] {
    background: rgba(20, 55, 41, 0.55);
    border: 1px dashed #4ADE80;
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(90deg, #16A34A, #22C55E);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: 700;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #15803D, #16A34A);
    border: 1px solid #86EFAC;
    transform: translateY(-2px);
}

/* Footer */

.footer {
    text-align: center;
    color: #86A994;
    padding: 25px;
    font-size: 13px;
}

hr {
    border-color: #285C43;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# MODEL CONFIGURATION
# --------------------------------

MODEL_PATH = "best_waste_model.pth"

CLASS_NAMES = [
    "Organic",
    "Recyclable",
    "Non-Recyclable"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Supports either a state_dict or a checkpoint dictionary
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------
# IMAGE TRANSFORMATION
# Must match training preprocessing
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------

def predict_image(image, model):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    confidence, predicted_index = torch.max(
        probabilities, dim=1
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    confidence = confidence.item() * 100

    probability_values = probabilities[0].cpu().tolist()

    return (
        predicted_class,
        confidence,
        probability_values,
        inference_time
    )


# --------------------------------
# HEADER
# --------------------------------

st.markdown("""
<div class="hero">

    <h1>♻️ EcoVision AI</h1>

    <p>
        AI-Powered Waste Classification System
    </p>

    <p>
        Upload an image and let Artificial Intelligence
        identify its waste category.
    </p>

</div>
""", unsafe_allow_html=True)


# --------------------------------
# CATEGORY INFORMATION
# --------------------------------

st.markdown(
    '<div class="section-title">🌱 Waste Categories</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🍃 Organic</h3>
        <p>Biodegradable waste such as food scraps,
        fruit peels, vegetables and leaves.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>♻️ Recyclable</h3>
        <p>Materials that can be processed and reused,
        such as paper, cans and suitable plastics.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>🚯 Non-Recyclable</h3>
        <p>Waste that cannot be recycled through
        the intended recycling process.</p>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------
# IMAGE UPLOAD
# --------------------------------

st.markdown(
    '<div class="section-title">📸 Upload Waste Image</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1], gap="large")

with left:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload a clear image of a waste item."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    else:
        st.markdown("""
        <div class="info-card">
            <h3>📂 Image Upload</h3>
            <p>Choose a JPG, JPEG, PNG or WEBP image
            from your device.</p>
            <p>For better predictions, upload a clear
            image with the waste item visible.</p>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------
# PREDICTION
# --------------------------------

with right:

    st.markdown(
        '<div class="section-title">🤖 AI Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Our trained MobileNetV2 model analyzes "
        "the uploaded image and predicts its category."
    )

    if uploaded_file:

        if st.button("🔍 Classify Waste", use_container_width=True):

            try:

                with st.spinner("AI is analyzing your image..."):

                    model = load_model()

                    (
                        predicted_class,
                        confidence,
                        probabilities,
                        inference_time
                    ) = predict_image(image, model)

                st.session_state["prediction"] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "time": inference_time
                }

            except FileNotFoundError:
                st.error(
                    "Model file not found. "
                    "Place best_waste_model.pth beside app.py."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:
        st.info("Upload an image to start classification.")


# --------------------------------
# DISPLAY RESULTS
# --------------------------------

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    predicted_class = result["class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    inference_time = result["time"]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Classification Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns([1.2, 1])

    with result_col1:

        category_icons = {
            "Organic": "🍃",
            "Recyclable": "♻️",
            "Non-Recyclable": "🚯"
        }

        st.markdown(f"""
        <div class="result-card">
            <h2>Predicted Category</h2>
            <h1>{category_icons[predicted_class]} {predicted_class}</h1>
            <p style="color:#DCFCE7;font-size:17px;">
                Confidence: <b>{confidence:.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with result_col2:

        st.markdown("""
        <div class="info-card">
            <h3>⚡ Model Information</h3>
            <p><b>Model:</b> MobileNetV2</p>
            <p><b>Framework:</b> PyTorch</p>
            <p><b>Input Size:</b> 224 × 224</p>
            <p><b>Device:</b> """ + str(DEVICE).upper() + """</p>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Inference Time",
            f"{inference_time:.2f} ms"
        )

    # Probability breakdown

    st.markdown(
        '<div class="section-title">📈 Category Confidence Scores</div>',
        unsafe_allow_html=True
    )

    for category, probability in zip(
        CLASS_NAMES, probabilities
    ):

        percentage = probability * 100

        st.write(
            f"**{category}** — {percentage:.2f}%"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

    st.caption(
        "Confidence scores represent the model's predicted probabilities. "
        "They do not guarantee that the classification is correct."
    )

    if st.button("🔄 Classify Another Image"):
        del st.session_state["prediction"]
        st.rerun()


# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.markdown("""
<div class="footer">
    <p>♻️ EcoVision AI | Smart Waste Classification</p>
    <p>Powered by MobileNetV2 and Deep Learning</p>
    <p>Building a cleaner and more sustainable future 🌍</p>
</div>
""", unsafe_allow_html=True)
