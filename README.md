# # EquaSense

## A Deep Learning-Based Optical Character Recognition System for Mathematical Equation Recognition

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
