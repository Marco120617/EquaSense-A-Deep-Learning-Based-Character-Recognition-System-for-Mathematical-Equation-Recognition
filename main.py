# main.py
# EquaSense: A Deep Learning-Based Character Recognition System
# for Mathematical Equation Recognition

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
from PIL import Image

# ---------------------------------------------------------
# Optional OCR Engines
# ---------------------------------------------------------

PIX2TEX_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    from pix2tex.cli import LatexOCR
    PIX2TEX_AVAILABLE = True
except Exception:
    PIX2TEX_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="EquaSense",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 EquaSense")
st.subheader("A Deep Learning-Based Character Recognition System for Mathematical Equation Recognition")

st.write(
    "EquaSense recognizes mathematical equations from images and converts them "
    "into digital text or LaTeX-style format."
)


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format."""
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def preprocess_image(pil_image):
    """
    Preprocess image for OCR:
    1. Convert to grayscale
    2. Denoise
    3. Threshold
    4. Resize for better recognition
    """
    image = pil_to_cv2(pil_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    height, width = thresholded.shape

    scale_percent = 150
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)

    resized = cv2.resize(
        thresholded,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    return cv2_to_pil(resized)


def normalize_equation(text):
    """Normalize equation text for comparison."""
    if text is None:
        return ""

    text = text.lower()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("{", "")
    text = text.replace("}", "")

    return text


def levenshtein_distance(a, b):
    """Compute edit distance between two strings."""
    m = len(a)
    n = len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]


def calculate_cer(predicted, expected):
    """
    Character Error Rate:
    CER = edit distance / length of expected text
    """
    predicted = normalize_equation(predicted)
    expected = normalize_equation(expected)

    if len(expected) == 0:
        return 0

    distance = levenshtein_distance(predicted, expected)
    cer = distance / len(expected)

    return cer


def calculate_accuracy(predicted, expected):
    """Calculate character-level accuracy."""
    cer = calculate_cer(predicted, expected)
    accuracy = max(0, (1 - cer) * 100)
    return accuracy


def save_result(file_name, expected, recognized, accuracy, cer):
    """Save recognition result to CSV."""
    result_file = "equasense_results.csv"

    new_data = pd.DataFrame([{
        "Image File": file_name,
        "Expected Equation": expected,
        "Recognized Equation": recognized,
        "Accuracy (%)": round(accuracy, 2),
        "Character Error Rate": round(cer, 4),
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(result_file):
        old_data = pd.read_csv(result_file)
        final_data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        final_data = new_data

    final_data.to_csv(result_file, index=False)


# ---------------------------------------------------------
# Load OCR Model
# ---------------------------------------------------------

@st.cache_resource
def load_latex_ocr_model():
    """Load LaTeX-OCR model once."""
    if PIX2TEX_AVAILABLE:
        return LatexOCR()
    return None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("System Settings")

ocr_engine = st.sidebar.selectbox(
    "Select OCR Engine",
    [
        "LaTeX-OCR / pix2tex Recommended",
        "Tesseract OCR Fallback"
    ]
)

use_preprocessing = st.sidebar.checkbox(
    "Apply image preprocessing",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.write("**Engine Status:**")

if PIX2TEX_AVAILABLE:
    st.sidebar.success("LaTeX-OCR / pix2tex available")
else:
    st.sidebar.warning("LaTeX-OCR / pix2tex not installed")

if TESSERACT_AVAILABLE:
    st.sidebar.success("Tesseract available")
else:
    st.sidebar.warning("Tesseract not installed")


# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image containing a mathematical equation",
    type=["png", "jpg", "jpeg"]
)

expected_equation = st.text_area(
    "Enter the expected/correct equation for comparison",
    placeholder="Example: s^2 = \\frac{\\sum (x_i - \\bar{x})^2}{n - 1}"
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Image")
        st.image(original_image, use_container_width=True)

    if use_preprocessing:
        processed_image = preprocess_image(original_image)
    else:
        processed_image = original_image

    with col2:
        st.markdown("### Processed Image")
        st.image(processed_image, use_container_width=True)

    st.markdown("---")

    if st.button("Recognize Equation"):
        start_time = time.time()

        recognized_text = ""

        with st.spinner("Recognizing equation..."):
            if ocr_engine == "LaTeX-OCR / pix2tex Recommended":
                if PIX2TEX_AVAILABLE:
                    model = load_latex_ocr_model()
                    recognized_text = model(processed_image)
                else:
                    st.error(
                        "LaTeX-OCR / pix2tex is not installed. "
                        "Install it first or use Tesseract fallback."
                    )

            elif ocr_engine == "Tesseract OCR Fallback":
                if TESSERACT_AVAILABLE:
                    recognized_text = pytesseract.image_to_string(
                        processed_image,
                        config="--psm 6"
                    )
                else:
                    st.error(
                        "Tesseract OCR is not installed. "
                        "Install pytesseract and Tesseract engine first."
                    )

        end_time = time.time()
        processing_time = end_time - start_time

        if recognized_text:
            st.markdown("## Recognition Result")

            st.text_area(
                "Recognized Equation",
                value=recognized_text,
                height=120
            )

            st.write(f"**Processing Time:** {processing_time:.2f} seconds")

            if expected_equation.strip() != "":
                accuracy = calculate_accuracy(recognized_text, expected_equation)
                cer = calculate_cer(recognized_text, expected_equation)

                st.markdown("## Evaluation Result")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Accuracy", f"{accuracy:.2f}%")

                with col_b:
                    st.metric("Character Error Rate", f"{cer:.4f}")

                if accuracy >= 80:
                    st.success("Result: Good recognition output")
                elif accuracy >= 50:
                    st.warning("Result: Partially correct recognition output")
                else:
                    st.error("Result: Low recognition accuracy")

                save_result(
                    uploaded_file.name,
                    expected_equation,
                    recognized_text,
                    accuracy,
                    cer
                )

                st.success("Result saved to equasense_results.csv")

else:
    st.info("Please upload an image to start equation recognition.")


# ---------------------------------------------------------
# Result History
# ---------------------------------------------------------

st.markdown("---")
st.markdown("## Recognition History")

if os.path.exists("equasense_results.csv"):
    results_df = pd.read_csv("equasense_results.csv")
    st.dataframe(results_df, use_container_width=True)
else:
    st.write("No saved results yet.")
