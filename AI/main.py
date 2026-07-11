import importlib
import importlib.util
import os
import sys
import time

import pandas as pd
import streamlit as st
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)

VENV_SITE_PACKAGES = os.path.join(APP_DIR, ".venv", "Lib", "site-packages")
if os.path.isdir(VENV_SITE_PACKAGES) and VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

# Core architectural layer imports
from src.logic.preprocess import preprocess_image
from src.logic.metrics import calculate_accuracy, calculate_cer
from src.data_layer.history_db import save_result


def _check_pix2tex():
    try:
        if importlib.util.find_spec("pix2tex") is None:
            if os.path.isdir(os.path.join(VENV_SITE_PACKAGES, "pix2tex")):
                return True, "available via local venv path"
            return False, "pix2tex package not found"
        pix2tex_cli = importlib.import_module("pix2tex.cli")
        if not hasattr(pix2tex_cli, "LatexOCR"):
            return False, "LatexOCR class not available"
        return True, "available"
    except Exception as exc:
        return False, f"import failed: {exc}"


def _check_tesseract():
    try:
        if importlib.util.find_spec("pytesseract") is None:
            return False, "pytesseract package not found"
        import pytesseract
        return True, "available"
    except Exception as exc:
        return False, f"import failed: {exc}"


# Optional OCR Engines Dependency Checks
PIX2TEX_AVAILABLE, PIX2TEX_STATUS = _check_pix2tex()
TESSERACT_AVAILABLE, TESSERACT_STATUS = _check_tesseract()

if PIX2TEX_AVAILABLE:
    from pix2tex.cli import LatexOCR

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
        # pix2tex works best with original, non-thresholded images
        # Convert to grayscale if needed, but preserve gradients
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return model(image)
    except Exception as e:
        st.error(f"pix2tex recognition error: {str(e)}")
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
st.sidebar.caption("Note: Preprocessing applies only to Tesseract. LaTeX-OCR uses the original image for best results.")
st.sidebar.markdown("---")
st.sidebar.write("**Engine Status:**")

if PIX2TEX_AVAILABLE:
    st.sidebar.success("LaTeX-OCR / pix2tex available")
else:
    st.sidebar.warning(f"LaTeX-OCR / pix2tex not installed ({PIX2TEX_STATUS})")

if TESSERACT_AVAILABLE:
    st.sidebar.success("Tesseract available")
else:
    st.sidebar.warning(f"Tesseract not installed ({TESSERACT_STATUS})")

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
                    # pix2tex works best with original, non-thresholded images
                    recognized_text = recognize_with_pix2tex(original_image)
                    if not recognized_text:
                        st.error("LaTeX-OCR / pix2tex failed to recognize the image.")
                else:
                    st.error("LaTeX-OCR / pix2tex is not installed. Use Native Built-in option.")

            elif ocr_engine == "Tesseract OCR Fallback":
                if TESSERACT_AVAILABLE:
                    # Tesseract benefits from preprocessing
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