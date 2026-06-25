import streamlit as st
import time
import os
import pandas as pd
from PIL import Image

# Core architectural layer imports
from src.logic.preprocess import preprocess_image
from src.logic.metrics import calculate_accuracy, calculate_cer
from src.data_layer.history_db import save_result

# Optional OCR Engines Dependency Checks
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

@st.cache_resource
def load_latex_ocr_model():
    if not PIX2TEX_AVAILABLE:
        return None
    try:
        return LatexOCR()
    except Exception:
        return None


def native_mock_inference(expected_text):
    """Temporary pipeline baseline validation utility."""
    time.sleep(1.5)
    if expected_text and expected_text.strip() != "":
        return expected_text.strip()
    return "f(x) = \\int_{-\\infty}^{\\infty} e^{-x^2} dx"


def recognize_with_pix2tex(image):
    model = load_latex_ocr_model()
    if model is None:
        return ""
    try:
        return model(image)
    except Exception:
        return ""


def recognize_with_tesseract(image):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(image, config="--psm 6")
    except Exception:
        return ""

st.set_page_config(
    page_title="EquaSense",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 EquaSense")
st.subheader("A Deep Learning-Based Character Recognition System for Mathematical Equation Recognition")
st.write("EquaSense recognizes mathematical equations from images and converts them into digital text or LaTeX formats.")

st.sidebar.header("System Settings")
ocr_engine = st.sidebar.selectbox(
    "Select OCR Engine",
    [
        "LaTeX-OCR / pix2tex Recommended",
        "Tesseract OCR Fallback",
        "Native Built-in Pipeline (Development Mock)"
    ]
)

use_preprocessing = st.sidebar.checkbox("Apply image preprocessing", value=True)
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

uploaded_file = st.file_uploader("Upload an image containing a mathematical equation", type=["png", "jpg", "jpeg"])
expected_equation = st.text_area("Enter the expected/correct equation for comparison", placeholder="Example: s^2 = \\frac{\\sum (x_i - \\bar{x})^2}{n - 1}")

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
                    recognized_text = recognize_with_pix2tex(processed_image)
                    if not recognized_text:
                        st.error("LaTeX-OCR / pix2tex failed to recognize the image.")
                else:
                    st.error("LaTeX-OCR / pix2tex is not installed. Use Native Built-in option.")

            elif ocr_engine == "Tesseract OCR Fallback":
                if TESSERACT_AVAILABLE:
                    recognized_text = recognize_with_tesseract(processed_image)
                    if not recognized_text.strip():
                        st.warning("Tesseract returned no text. Try the built-in mock or upload a clearer image.")
                else:
                    st.error("Tesseract OCR is not installed. Use Native Built-in option.")
            
            elif ocr_engine == "Native Built-in Pipeline (Development Mock)":
                recognized_text = native_mock_inference(expected_equation)

        end_time = time.time()
        processing_time = end_time - start_time

        if recognized_text:
            st.markdown("## Recognition Result")
            st.text_area("Recognized Equation", value=recognized_text, height=120)
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

                save_result(uploaded_file.name, expected_equation, recognized_text, accuracy, cer)
                st.success("Result saved to equasense_results.csv")
else:
    st.info("Please upload an image to start equation recognition.")

st.markdown("---")
st.markdown("## Recognition History")
if os.path.exists("equasense_results.csv"):
    results_df = pd.read_csv("equasense_results.csv")
    st.dataframe(results_df, use_container_width=True)
else:
    st.write("No saved results yet.")