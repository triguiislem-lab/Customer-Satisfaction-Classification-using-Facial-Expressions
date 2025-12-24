# Customer Satisfaction Detector

A deep learning-based real-time customer satisfaction detection system using facial expression analysis with TensorFlow and OpenCV.

## 🎯 Overview

This project implements an AI-powered customer satisfaction detector that analyzes facial expressions in real-time using a trained CNN model. The system can classify customer emotions into three categories: Satisfied, Neutral, and Unsatisfied.

## ✨ Features

- **Real-time Detection**: Live facial expression analysis using webcam
- **High Accuracy**: Deep learning model trained on customer satisfaction data
- **Performance Optimized**: 
  - Real-time FPS monitoring
  - Frame skipping for better performance
  - Smart face detection focusing on the largest face
- **Advanced Analytics**:
  - Live prediction history (tracks last 30 predictions)
  - Session summaries with detailed statistics
  - Confidence score tracking
- **Interactive Controls**:
  - `q`: Quit application
  - `s`: Save current frame with timestamp
  - `r`: Reset statistics
  - `h`: Toggle help overlay
- **API Integration**: Optional backend API for data logging

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam/Camera
- TensorFlow 2.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/triguiislem-lab/Customer-Satisfaction-Classification-using-Facial-Expressions.git
cd Customer-Satisfaction-Classification-using-Facial-Expressions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the trained model:
   - Place `satisfaction_model_best.h5` in the project root directory

### Usage

#### Test Camera Availability
```bash
python realtime_satisfaction_detector.py --test-cameras
```

#### Run Real-time Detection
```bash
python realtime_satisfaction_detector.py
```

#### Use Different Camera
```bash
python realtime_satisfaction_detector.py --camera 1
```

#### Advanced Options
```bash
python realtime_satisfaction_detector.py --model path/to/model.h5 --skip-frames 2 --min-face-size 50
```

## 📁 Project Structure

```
customer-satisfaction-detector/
├── realtime_satisfaction_detector.py   # Main application
├── customer_satisfaction_classifier.ipynb  # Model training notebook
├── requirements.txt                    # Dependencies
├── satisfaction_model_best.h5          # Trained model (add separately)
├── docs/
│   ├── API_INTEGRATION.md             # API integration guide
│   └── REALTIME_USAGE.md              # Detailed usage guide
└── README.md                           # This file
```

## 🛠️ Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--camera` | Camera ID to use | 0 |
| `--model` | Path to model file | satisfaction_model_best.h5 |
| `--test-cameras` | Test camera availability | False |
| `--min-face-size` | Minimum face size for detection | 30 |
| `--skip-frames` | Process every Nth frame | 1 |
| `--api-url` | Backend API URL | None |
| `--session-id` | Session ID for API logging | None |

## 📊 Model Architecture

The system uses a Convolutional Neural Network (CNN) trained on facial expression data:
- Input: 48x48 grayscale images
- Multiple convolutional and pooling layers
- Dropout layers for regularization
- Output: 3 classes (Satisfied, Neutral, Unsatisfied)

## 🎓 Training Your Own Model

Use the included Jupyter notebook to train a custom model:
```bash
jupyter notebook customer_satisfaction_classifier.ipynb
```

## 🔧 API Integration (Optional)

The system supports optional backend API integration for logging predictions. See [docs/API_INTEGRATION.md](docs/API_INTEGRATION.md) for details.

## 📈 Performance

- Real-time processing at 15-30 FPS (depending on hardware)
- Average confidence score: 85%+
- Low latency face detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

triguiislem-lab
- GitHub: [@triguiislem-lab](https://github.com/triguiislem-lab)
- Project: [Customer Satisfaction Detector](https://github.com/triguiislem-lab/Customer-Satisfaction-Classification-using-Facial-Expressions)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV for computer vision capabilities
- The open-source community

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ for better customer experience analysis
