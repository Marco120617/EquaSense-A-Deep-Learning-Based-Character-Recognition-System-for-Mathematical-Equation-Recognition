import cv2
import numpy as np
from PIL import Image

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def preprocess_image(pil_image):
    image = pil_to_cv2(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresholded = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 11
    )
    height, width = thresholded.shape
    scale_percent = 150
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized = cv2.resize(thresholded, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return cv2_to_pil(resized)