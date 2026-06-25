## EquaSense - A Deep Learning-Based Optical Character Recognition System for Mathematical Equation Recognition

### Overview

EquaSense is a deep learning–based Optical Character Recognition (OCR) system designed to recognize and convert handwritten or printed mathematical equations into structured digital formats such as LaTeX, TXT, and PDF. The project addresses the limitations of traditional OCR systems by analyzing both mathematical symbols and their spatial relationships to accurately reconstruct complex mathematical expressions.

This project is developed as a requirement for **CPE179P – Software Design** at **Mapúa University**.

---

## Features

### User Functions

* Upload or capture mathematical equation images
* Image preprocessing (grayscale conversion, noise reduction, binarization)
* Mathematical symbol recognition using OCR and Deep Learning
* Equation structure analysis
* Conversion to LaTeX format
* Export recognized equations to:

  * TXT
  * PDF
  * LaTeX
* Batch processing of multiple equation images

### Administrator Functions

* OCR model management
* Model validation and deployment
* System maintenance and updates

---

## System Architecture

The system follows a modular **Three-Tier Architecture**:

### Presentation Layer

* User Interface
* Image Upload Module
* Results Viewer

### Processing Layer

* Image Preprocessing
* Feature Extraction (DenseNet/CNN)
* Transformer Encoder-Decoder
* Equation Structure Analysis
* Recognition Engine

### Data Layer

* Recognition History
* Export Management
* Model Storage

---

## Technologies Used

### Programming Language

* Python 3.x

### Libraries and Frameworks

* PyTorch
* OpenCV
* NumPy
* Pillow
* Tkinter
* Pytesseract
* ReportLab

### Datasets

* CROHME Dataset
* Im2LaTeX-100K Dataset

---

## Project Objectives

1. Develop a mathematical equation image preprocessing module.
2. Recognize mathematical symbols and structures using deep learning.
3. Convert recognized equations into structured formats such as LaTeX.
4. Support batch processing of multiple equation images.
5. Provide export functionality for academic and professional use.
6. Implement model management capabilities for system maintenance.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Keyarru/Sample-Project-Group-4.git
cd Sample-Project-Group-4
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Tesseract OCR

Download and install Tesseract OCR:

https://github.com/UB-Mannheim/tesseract/wiki

After installation, ensure the Tesseract executable is added to your system PATH.

---

## Running the Application

```bash
python main.py
```

---

## Project Workflow

1. Upload or capture an equation image.
2. Perform image preprocessing.
3. Extract symbol features using CNN/DenseNet.
4. Analyze symbol relationships using Transformer-based decoding.
5. Reconstruct the mathematical expression.
6. Generate structured output (LaTeX, TXT, PDF).
7. Save or export results.

---

## Expected Performance

* High mathematical symbol recognition accuracy
* Accurate equation structure reconstruction
* Near real-time processing speed
* Support for handwritten and printed equations
* Reliable LaTeX generation

## How to Download and Use EquaSense

### Step 1: Download the Project

Clone the repository using Git:

```bash
git clone https://github.com/Keyarru/Sample-Project-Group-4.git
```

Or download the ZIP file:

1. Open the GitHub repository.
2. Click the **Code** button.
3. Select **Download ZIP**.
4. Extract the ZIP file to your preferred location.

---

### Step 2: Open the Project

Navigate to the project folder:

```bash
cd Sample-Project-Group-4
```

Open the project using your preferred IDE:

* Visual Studio Code
* PyCharm
* Spyder
* Any Python-supported IDE

For Visual Studio Code:

```bash
code .
```

---

### Step 3: Install Python

Ensure Python 3.10 or later is installed.

Verify installation:

```bash
python --version
```

---

### Step 4: Install Required Libraries

Install all project dependencies:

```bash
pip install -r requirements.txt
```

---

### Step 5: Install Tesseract OCR

1. Download Tesseract OCR from:
   https://github.com/UB-Mannheim/tesseract/wiki

2. Install the software.

3. Add the Tesseract installation folder to your system PATH.

4. Verify installation:

```bash
tesseract --version
```

---

### Step 6: Run the Application

Start EquaSense:

```bash
python main.py
```

The graphical user interface (GUI) will open.

---

## How to Use EquaSense

### Upload an Equation Image

1. Click **Upload Image**.
2. Select an image containing a handwritten or printed mathematical equation.
3. The selected image will be displayed in the preview area.

### Preprocess the Image

1. Click **Preprocess**.
2. The system applies:

   * Grayscale conversion
   * Noise reduction
   * Binarization
3. The cleaned image will be displayed.

### Recognize the Equation

1. Click **Recognize Equation**.
2. The OCR engine analyzes the image.
3. The recognized equation is displayed in the output area.

### Export Results

1. Click **Export**.
2. Choose the desired output format:

   * TXT
   * PDF
   * LaTeX (future version)
3. Select a save location.
4. The output file will be generated and saved.

### Batch Processing (Future Feature)

1. Select multiple equation images.
2. Start batch processing.
3. EquaSense processes each image sequentially.
4. Results are automatically generated and stored.

---

## Troubleshooting

### Tesseract Not Found

If you receive a Tesseract error:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

Update the path according to your installation directory.

### Missing Libraries

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Application Does Not Start

Verify that:

* Python is installed correctly.
* All required libraries are installed.
* Tesseract OCR is properly configured.


---

## Project Team

### Group 4

* Marco Iggi Heinz D. Banzon
* Carl Daniel Bea
* Donn Marco Ancheta
* Razec Meir Biocarles

School of Electrical, Electronics and Computer Engineering

Mapúa University

---

## Course Information

**Course:** CPE179P – Software Design

**Project Title:** EquaSense: A Deep Learning Based Optical Character Recognition System for Mathematical Equation Recognition

**Academic Year:** 2025–2026

---

## License

This project is developed for academic and educational purposes under Mapúa University requirements.
