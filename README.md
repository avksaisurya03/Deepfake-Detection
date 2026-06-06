# Explainable Deepfake Detection Using Transfer Learning and Facial Motion Analysis

## 📌 Overview

This project presents an **Explainable Deepfake Detection System** capable of accurately identifying manipulated (fake) videos and images using **Deep Learning, Transfer Learning, and Facial Motion Analysis**.

The system combines **spatial analysis (visual inconsistencies)** and **temporal analysis (facial motion behavior)** to improve detection reliability. Unlike traditional black-box detectors, this project also generates **visual and text-based explanations** for predictions to improve transparency and trust.

---

## 🚀 Features

* ✅ Detects **Real vs Fake** videos and images
* ✅ Supports **both image and video input**
* ✅ Face extraction using **MTCNN**
* ✅ Deepfake classification using **EfficientNet-B4 (Transfer Learning)**
* ✅ Facial motion analysis using **Optical Flow**
* ✅ Explainable AI with **Grad-CAM Heatmaps**
* ✅ Text-based explanations using **LLaMA 3.2 (LLM)**
* ✅ Confidence score generation
* ✅ Streamlit-based web application
* ✅ Offline functionality (No internet required)

---

## 🧠 Problem Statement

Deepfake technology has evolved rapidly, making manipulated videos highly realistic and difficult to detect. These fake videos pose serious threats such as:

* Fake news
* Identity fraud
* Political misinformation
* Cybercrime

This project aims to build an **explainable deepfake detection system** capable of identifying manipulation while providing understandable explanations behind the predictions.

---

## 📂 Dataset Used

This project uses a combination of:

### 1. FaceForensics++

* Large-scale manipulated face dataset
* Includes multiple face manipulation methods

### 2. Celeb-DF (v2)

* High-quality celebrity deepfake videos
* More realistic fake content

### Dataset Statistics

* **Total Images:** 43,222
* **Real Images:** 22,669
* **Fake Images:** 20,553

### Dataset Split

* **Training:** 32,209 images
* **Validation:** 5,516 images
* **Testing:** 5,497 images

---

## 🏗️ Project Architecture

The system follows the following workflow:

Input Video/Image
→ Frame Extraction
→ Face Detection (MTCNN)
→ Feature Extraction (EfficientNet-B4)
→ Facial Motion Analysis (Optical Flow)
→ Deepfake Classification
→ Grad-CAM Visualization
→ LLaMA Explanation
→ Final Prediction (Real/Fake)

---

## 🛠️ Technologies Used

* Python 3
* PyTorch
* Timm Library
* EfficientNet-B4
* MTCNN
* OpenCV
* pytorch-grad-cam
* Ollama + LLaMA 3.2
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/deepfake-detection.git
```

### Move to Project Folder

```bash
cd deepfake-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
streamlit run app.py
```

---

## 📊 Results

The developed system successfully:

* Identifies manipulated content
* Detects facial inconsistencies
* Analyzes motion abnormalities
* Produces explainable outputs

Outputs include:

* **Real/Fake Prediction**
* **Confidence Score**
* **Grad-CAM Heatmaps**
* **LLM-based Explanation**

---

## 🔮 Future Improvements

* Real-time webcam deepfake detection
* Multi-modal detection (Audio + Video)
* Mobile deployment optimization
* Improved robustness against advanced deepfakes

---

This project is developed for educational and research purposes.
