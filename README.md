# YOLO11 Multi-Task Computer Vision Pipeline

A comprehensive Python-based pipeline for training and testing multiple computer vision tasks using YOLO11 (You Only Look Once v11). This project demonstrates end-to-end workflows for classification, object detection, instance segmentation, and pose estimation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

This project provides a unified framework for training and evaluating YOLO11 models across four distinct computer vision tasks:

1. **Classification** - Image classification using CIFAR-10 dataset
2. **Detection** - Object detection using COCO128 dataset
3. **Segmentation** - Instance segmentation using COCO8-seg dataset
4. **Pose Estimation** - Human pose estimation using COCO8-pose dataset

All tasks are orchestrated through a single `main.py` script, making it easy to train and test multiple models sequentially.

## ✨ Features

- **Multi-Task Support**: Train and test all four vision tasks from a single entry point
- **Pre-trained Models**: Uses YOLO11 nano (lightweight) models as base
- **CPU-Friendly**: Configured to run on CPU devices (easily switchable to GPU)
- **Error Handling**: Robust exception handling for each task
- **Modular Design**: Separate modules for each task (classification, detection, segmentation, pose)
- **Automatic Model Discovery**: Dynamically finds the latest trained models for testing
- **Progress Tracking**: Clear console output showing training and testing progress

## 📦 Requirements

- Python 3.8+
- PyTorch (CPU or GPU version)
- Ultralytics YOLOv8/v11
- NumPy, OpenCV

## 🔧 Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd yolo11-multitask-pipeline
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n yolo11 python=3.10
conda activate yolo11
```

### 3. Install Dependencies

```bash
pip install ultralytics torch torchvision opencv-python numpy
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
yolo11-multitask-pipeline/
├── main.py                 # Entry point - orchestrates all tasks
├── models/                 # All model modules
│   ├── __init__.py
│   ├── classification.py   # Classification module
│   ├── detection.py        # Detection module
│   ├── segmentation.py     # Segmentation module
│   └── pose.py             # Pose estimation module
├── media/                  # Test images directory
│   ├── test_img.jpg        # Classification test image
│   ├── bus.jpg             # Detection test image
│   ├── messy_table.jpg     # Segmentation test image
│   └── poses.jpg           # Pose estimation test image
├── runs/                   # Output directory (auto-created)
│   ├── classify/           # Classification results
│   ├── detect/             # Detection results
│   ├── segment/            # Segmentation results
│   └── pose/               # Pose estimation results
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## 🚀 Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Train all four models sequentially
2. Test each model with sample images
3. Display results and save them to the `runs/` directory

### Run Individual Tasks

You can also run each task independently:

```bash
# Classification only
python classification.py

# Detection only
python detection.py

# Segmentation only
python segmentation.py

# Pose estimation only
python pose.py
```

## ⚙️ Configuration

### Training Parameters

Each module contains configurable training parameters:

#### Classification (`classification.py`)
```python
results = model.train(
    data='cifar10',           # Dataset
    epochs=5,                 # Number of epochs
    imgsz=224,               # Image size
    device='cpu',            # Device (cpu or 0 for GPU)
    patience=3,              # Early stopping patience
    batch=32,                # Batch size
    name='classification_model'
)
```

#### Detection (`detection.py`)
```python
results = model.train(
    data='coco128.yaml',      # Dataset
    epochs=10,               # Number of epochs
    imgsz=640,              # Image size
    device='cpu',           # Device
    patience=3,             # Early stopping patience
    batch=16,               # Batch size
    name='detection_model'
)
```

#### Segmentation (`segmentation.py`)
```python
results = model.train(
    data='coco8-seg.yaml',   # Dataset
    epochs=3,               # Number of epochs
    imgsz=640,             # Image size
    device='cpu',          # Device
    patience=3,            # Early stopping patience
    batch=16,              # Batch size
    name='segmentation_model'
)
```

#### Pose Estimation (`pose.py`)
```python
results = model.train(
    data='coco8-pose.yaml',  # Dataset
    epochs=3,               # Number of epochs
    imgsz=640,             # Image size
    device='cpu',          # Device
    patience=3,            # Early stopping patience
    batch=16,              # Batch size
    name='pose_model'
)
```

### Using GPU

To use GPU instead of CPU, change the `device` parameter:

```python
device='0'  # For single GPU
device='0,1'  # For multiple GPUs
```

## 📊 Output

After running the pipeline, results are saved in the `runs/` directory:

```
runs/
├── classify/
│   ├── classification_model/
│   │    ├── weights/
│   │    │   ├── best.pt
│   │    │   └── last.pt
│   │    ├── results.csv
│   │    └── confusion_matrix.png
│   └── predict/
│        └── img.jpg/
├── detect/
│   ├── detection_model/
│   │    ├── weights/
│   │    │   ├── best.pt
│   │    │   └── last.pt
│   │    ├── results.csv
│   │    └── confusion_matrix.png
│   └── predict/
│        └── img.jpg/
├── segment/
│   ├── segmentation_model/
│   │    ├── weights/
│   │    │   ├── best.pt
│   │    │   └── last.pt
│   │    └── results.csv
│   └── predict/
│        └── img.jpg/
└── pose/
    ├── pose_model/
    │    ├── weights/
    │    │   ├── best.pt
    │    │   └── last.pt
    │    └── results.csv
    └── predict/
         └── img.jpg/
```

Each task's predictions are also saved with visualization overlays.

## 🔍 Detailed Module Information

### Classification Module
- **Model**: YOLO11 Nano Classification (yolo11n-cls.pt)
- **Dataset**: CIFAR-10 (auto-downloaded)
- **Input**: Single images or batch of images
- **Output**: Class labels with confidence scores

### Detection Module
- **Model**: YOLO11 Nano Object Detection (yolo11n.pt)
- **Dataset**: COCO128 (subset of COCO)
- **Input**: Images
- **Output**: Bounding boxes with class labels and confidence scores

### Segmentation Module
- **Model**: YOLO11 Nano Segmentation (yolo11n-seg.pt)
- **Dataset**: COCO8-seg (small segmentation dataset)
- **Input**: Images
- **Output**: Segmentation masks with class labels

### Pose Estimation Module
- **Model**: YOLO11 Nano Pose (yolo11n-pose.pt)
- **Dataset**: COCO8-pose (small pose dataset)
- **Input**: Images
- **Output**: Keypoint coordinates with confidence scores

## 🐛 Troubleshooting

### Issue: Model files not found

**Solution**: Ultralytics automatically downloads pre-trained models on first use. Ensure you have internet connectivity during the first run.

### Issue: Out of memory error

**Solution**: Reduce batch size or image size in the training parameters:
```python
batch=8  # Instead of 16
imgsz=320  # Instead of 640
```

### Issue: Test images not found

**Solution**: Create a `media/` directory in the project root and add test images:
```bash
mkdir media
# Add test_img.jpg, bus.jpg, messy_table.jpg, poses.jpg
```

### Issue: CUDA/GPU not detected

**Solution**: Make sure PyTorch is installed with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Or stick with CPU mode by keeping `device='cpu'`.

### Issue: Dataset download fails

**Solution**: Ensure stable internet connection. If it persists, manually download datasets and specify local paths:
```python
data='path/to/local/dataset.yaml'
```

## 📈 Performance Monitoring

To monitor training in real-time:

1. Training logs are saved in each `runs/` subdirectory
2. Check `results.csv` for metrics over epochs
3. Tensorboard visualization (optional):
```bash
tensorboard --logdir runs/
```

## 🔄 Customization

### Add Custom Datasets

Modify the `data` parameter to point to your own YAML file:
```python
results = model.train(
    data='path/to/your/dataset.yaml',
    # ... other parameters
)
```

### Adjust Training Duration

Modify the `epochs` parameter:
```python
epochs=50  # For longer training
epochs=1   # For quick testing
```

### Change Batch Processing

Adjust batch size based on your hardware:
```python
batch=64  # Larger batches (requires more memory)
batch=4   # Smaller batches (slower training)
```

## 📝 License

This project uses YOLO11 from Ultralytics. Please refer to the [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for usage terms.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests

## 📚 References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Dataset](https://cocodataset.org/)

## ⚡ Quick Start Summary

```bash
# 1. Install dependencies
pip install ultralytics torch torchvision opencv-python

# 2. Create media directory with test images
mkdir media

# 3. Run the complete pipeline
python main.py

# 4. Check results
ls runs/
```

---

**Happy Training! 🚀**

For questions or issues, please refer to the troubleshooting section or consult the Ultralytics documentation.

Thank You!
