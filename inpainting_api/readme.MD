# Lithology Image Inpainting API (CRA-GAN)

This subdirectory contains the **FastAPI** service for reconstructing damaged geological core sample images using **Contextual Residual Aggregation (CRA) GAN**. 

In this research, this module serves as a "semantic bridge," restoring the structural continuity of rock textures before the deep lithological classification stage.

[Image of CRA-GAN architecture for image inpainting]

## 📌 Overview
In geological core analysis, physical damage (such as plug holes or mechanical cracks) creates "voids" that act as adversarial noise for deep learning classifiers. This API utilizes a generative model to:
1. Accept a damaged core image and its corresponding binary mask.
2. Fill the masked regions using high-frequency residual aggregation.
3. Output a reconstructed image optimized for texture-based feature extraction.

## 🚀 Key Features
* **Model:** CRA-GAN (Contextual Residual Aggregation).
* **Architecture:** Generator with attention mechanisms and residual context aggregation.
* **Performance:** High-throughput asynchronous processing via FastAPI.
* **Scientific Focus:** Specifically tuned to address the "semantic gap" in specialized visual domains.

## 🛠 Installation & Setup

### Prerequisites
* Python 3.7+
* NVIDIA GPU + CUDA 10.0 (Recommended for CRA-GAN inference)

### Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/GalymzhanAbdimanap/Lithology_recognition.git](https://github.com/GalymzhanAbdimanap/Lithology_recognition.git)
   cd Lithology_recognition/inpainting_api
2. Install dependencies:
   ```bash
   pip install -r requirements.txt```
3.Running the Server
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000```

💻 API Usage
Endpoint: POST /predict
Upload a damaged image and a mask to receive the reconstructed result.
