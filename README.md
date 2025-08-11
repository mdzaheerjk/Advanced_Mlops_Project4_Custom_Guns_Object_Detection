# 🔫 Custom Guns Object Detection – Advanced MLOps Project

![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![License](https://img.shields.io/badge/License-GPL--3.0-orange)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![YOLO](https://img.shields.io/badge/YOLO-Detection-success)

A complete MLOps project for custom guns object detection using deep learning. This repository demonstrates the full pipeline: data versioning with DVC, modular source code, training and inference scripts, and best-practice MLOps workflow for reproducibility and scalability.

> 📁 **Repository**: `Advanced_Mlops_Project4_Custom_Guns_Object_Detection`

---

## 🚀 Project Highlights

- 🎯 **Object Detection**: Custom deep learning model for guns detection in images/videos
- 🏗️ **Modular MLOps Workflow**: Clear separation of data, config, code, and pipelines
- 🗃️ **DVC Integration**: Data versioning and pipeline tracking with `dvc.yaml`
- 🛠️ **Config-Driven**: Easily adjust experiment and pipeline settings
- 📈 **Reproducible Research**: Versioned experiments and easy environment setup

---

## 🧠 Technical Stack

### 🛠️ Core Technologies
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![YOLO](https://img.shields.io/badge/YOLO-Detection-success)

### 📦 Libraries & Utilities
- PyTorch or TensorFlow (deep learning backend)
- OpenCV, NumPy, Pandas (data & image processing)
- DVC (data & pipeline versioning)
- Jupyter (experimentation)

---

## 🏗️ Project Structure

```bash
Advanced_Mlops_Project4_Custom_Guns_Object_Detection/
├── Dataset/                        # Gun detection datasets (images, labels)
├── Code/
│   ├── config/
│   │   ├── __init__.py
│   │   └── data_ingestion_config.py
│   ├── notebook/
│   │   └── guns-object-detection.ipynb   # EDA, training, and experiment logs
│   ├── src/                       # Core source code (model, training, utilities)
│   ├── dvc.yaml                   # DVC pipeline definition
│   ├── main.py                    # Main training/inference script
│   ├── requirements.txt           # Python dependencies
│   ├── setup.py                   # Package info
│   ├── LICENSE
│   ├── PDF and NOTES.pdf          # Project documentation & workflow
│   └── README.md
├── .gitignore
```

---

## ⚡ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/mdzaheerjk/Advanced_Mlops_Project4_Custom_Guns_Object_Detection.git
cd Advanced_Mlops_Project4_Custom_Guns_Object_Detection/Code
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Versioning with DVC
```bash
# (Optional) Initialize and pull datasets/outputs if not present
dvc pull
```

### 5. Run training or inference
```bash
# Train or evaluate the model (update command as per your main.py)
python main.py --train  # or --eval, see main.py for options
```

### 6. Experiment interactively (optional)
```bash
jupyter notebook notebook/guns-object-detection.ipynb
```

---

## 📊 Example Use Cases

- **Security and Surveillance**: Automated detection of guns in live video or CCTV feeds
- **Research**: Experimentation with object detection architectures and datasets
- **MLOps Demo**: Data versioning, pipeline reproducibility, and model management

---

## ✍️ Author

**Mohammed Zaheeruddin**  
🎓 First-Year B.Tech Student | AI/ML & GenAI Enthusiast  
🏫 Shetty Institute of Technology, Gulbarga  
📧 info.zaheerjk@gmail.com

---

## 📜 License

This project is licensed under the **GPL-3.0 License** – see the LICENSE file for details.

---

#### Key Features:
1. End-to-end custom object detection pipeline for gun detection
2. DVC-powered data and experiment versioning for robust MLOps
3. Modular and configurable codebase
4. Ready for research, security, and educational adaptation
