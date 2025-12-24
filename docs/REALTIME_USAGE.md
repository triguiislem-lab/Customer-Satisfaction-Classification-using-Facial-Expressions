# Real-Time Customer Satisfaction Detector - Usage Guide

## 🎯 Overview

This enhanced real-time satisfaction detector uses your trained `.h5` model with OpenCV to analyze customer satisfaction through facial expressions using your PC's camera.

## ✨ Features

### 🚀 **Enhanced Performance**
- Real-time FPS counter and performance monitoring
- Frame skipping for better performance on slower computers
- Optimized face detection with configurable minimum face size
- Smart processing - focuses on the largest detected face

### 📊 **Advanced Statistics**
- Live prediction history (tracks last 30 predictions)
- Session summaries with detailed statistics
- Real-time confidence tracking
- Distribution analysis of predictions

### 🎮 **Interactive Controls**
- **'q'**: Quit the application
- **'s'**: Save current frame with timestamp
- **'r'**: Reset all statistics
- **'h'**: Toggle help overlay on/off

### 🔧 **Command Line Options**
- `--camera ID`: Choose camera (default: 0)
- `--model PATH`: Specify model file path
- `--test-cameras`: Test camera availability
- `--min-face-size SIZE`: Minimum face size for detection
- `--skip-frames N`: Process every Nth frame for performance

## 🚀 Quick Start

### 1. Test Your Cameras
```bash
python realtime_satisfaction_detector.py --test-cameras
```

### 2. Start Real-Time Detection
```bash
python realtime_satisfaction_detector.py
```

### 3. Use Different Camera
```bash
python realtime_satisfaction_detector.py --camera 1
```

### 4. Performance Mode (for slower computers)
```bash
python realtime_satisfaction_detector.py --skip-frames 3 --min-face-size 100
```

## 📋 Usage Examples

### Basic Usage
```bash
python realtime_satisfaction_detector.py
```

### Advanced Usage
```bash
python realtime_satisfaction_detector.py --camera 1 --skip-frames 2 --min-face-size 80
```

### Custom Model
```bash
python realtime_satisfaction_detector.py --model my_custom_model.h5
```

## 🎯 What You'll See

### Live Video Display
Colored rectangles around detected faces:
- 🟢 **Green**: Satisfied
- 🟡 **Yellow**: Neutral  
- 🔴 **Red**: Unsatisfied

### Real-Time Information Overlay
- FPS counter - Current frames per second
- Total predictions - Number of predictions made
- Recent trend - Most common recent prediction with average confidence
- Control instructions - Keyboard shortcuts

### Prediction Labels
- **Satisfied**: Happy, surprised expressions
- **Neutral**: Calm, relaxed expressions
- **Unsatisfied**: Angry, sad, fearful expressions

## 📊 Session Statistics

When you quit (press 'q'), you'll see a detailed summary:

```
📊 SESSION SUMMARY
==================================================
⏱️  Duration: 45.2 seconds
📈 Total predictions: 127

🏷️ Prediction distribution:
   Satisfied: 45 (35.4%)
   Neutral: 62 (48.8%)
   Unsatisfied: 20 (15.7%)

📊 Average confidence: 78.3%
🎯 Most common: Neutral
==================================================
```

## 🔧 Troubleshooting

### Camera Issues
- **No camera found**: Run `--test-cameras` to see available cameras
- **Permission denied**: Check camera permissions in your OS
- **Camera in use**: Close other applications using the camera

### Performance Issues
- **Low FPS**: Use `--skip-frames 3` to process fewer frames
- **Lag**: Increase `--min-face-size` to ignore small faces
- **High CPU**: Close other applications

### Model Issues
- **Model not found**: Ensure `satisfaction_model_best.h5` is in the current directory
- **Loading errors**: Check the model file integrity

## 💡 Tips for Best Results

### For Accurate Predictions
1. Good lighting - Ensure your face is well-lit
2. Clear view - Face the camera directly
3. Stable position - Keep your head relatively still
4. Appropriate distance - Stay 2-4 feet from camera
5. Clean background - Avoid cluttered backgrounds

### For Better Performance
1. Close other apps - Free up system resources
2. Use frame skipping - `--skip-frames 2` or `--skip-frames 3`
3. Increase min face size - `--min-face-size 100`
4. Lower camera resolution - The script automatically sets 640x480

## 🎮 Keyboard Controls Summary

| Key | Action |
|-----|--------|
| **q** | Quit application |
| **s** | Save current frame |
| **r** | Reset statistics |
| **h** | Toggle help overlay |

## 🔄 Next Steps

1. Test the enhanced version with your camera
2. Experiment with different settings for optimal performance
3. Save interesting frames using the 's' key
4. Analyze the session statistics to understand model behavior
5. Try different cameras if you have multiple available
