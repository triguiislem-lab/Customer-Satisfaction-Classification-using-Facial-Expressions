import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import time
from collections import deque
from datetime import datetime
import argparse
import requests
import json

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

class SatisfactionDetector:
    def __init__(self, model_path='satisfaction_model_best.h5', api_url=None, session_id=None):
        self.model_path = model_path
        self.model = None
        self.satisfaction_labels = {
            0: 'Satisfied',
            1: 'Neutral',
            2: 'Unsatisfied'
        }
        self.colors = {
            'Satisfied': (0, 255, 0),      # Green
            'Neutral': (255, 255, 0),      # Yellow
            'Unsatisfied': (0, 0, 255)     # Red
        }

        # API configuration
        self.api_url = api_url
        self.session_id = session_id

        # Performance tracking
        self.prediction_history = deque(maxlen=30)  # Keep last 30 predictions
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Statistics
        self.total_predictions = 0
        self.session_start_time = time.time()

        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            print(f"Loading model from {self.model_path}...")

            # Try different loading methods
            try:
                # Method 1: Standard loading
                self.model = load_model(self.model_path, compile=False)
            except Exception as e1:
                print(f"First loading attempt failed: {str(e1)}")
                try:
                    # Method 2: Custom objects
                    self.model = tf.keras.models.load_model(
                        self.model_path,
                        compile=False,
                        custom_objects={'InputLayer': tf.keras.layers.InputLayer}
                    )
                except Exception as e2:
                    print(f"Second loading attempt failed: {str(e2)}")
                    try:
                        # Method 3: Load weights only
                        # Create a new model with the same architecture
                        inputs = tf.keras.Input(shape=(48, 48, 1))
                        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
                        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
                        x = tf.keras.layers.Flatten()(x)
                        x = tf.keras.layers.Dense(64, activation='relu')(x)
                        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
                        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

                        # Load weights
                        self.model.load_weights(self.model_path)
                    except Exception as e3:
                        print(f"All loading attempts failed. Last error: {str(e3)}")
                        raise Exception("Could not load model with any method")

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def predict_satisfaction(self, face_img):
        """
        Predicts customer satisfaction from a face image.

        Args:
            face_img (np.ndarray): The face image (should be 48x48 grayscale)

        Returns:
            tuple: (predicted_label, confidence, probabilities)
        """
        if face_img is None or face_img.shape != (48, 48):
            return None, None, None

        # Preprocess the image
        img = face_img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1) # Add channel dimension

        # Make prediction
        pred_probs = self.model.predict(img, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        confidence = np.max(pred_probs) * 100
        pred_label = self.satisfaction_labels.get(pred_class, 'Unknown')

        return pred_label, confidence, pred_probs

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time

    def add_prediction(self, prediction_data):
        """Add prediction to history"""
        if prediction_data:
            prediction_data['timestamp'] = time.time()
            self.prediction_history.append(prediction_data)
            self.total_predictions += 1

    def get_recent_stats(self, last_n=10):
        """Get statistics from recent predictions"""
        if not self.prediction_history:
            return None

        recent = list(self.prediction_history)[-last_n:]
        if not recent:
            return None

        # Count predictions by class
        class_counts = {}
        for label in self.satisfaction_labels.values():
            count = sum(1 for p in recent if p.get('label') == label)
            class_counts[label] = count

        # Calculate average confidence
        avg_confidence = np.mean([p.get('confidence', 0) for p in recent])

        # Most common prediction
        most_common = max(class_counts, key=class_counts.get) if class_counts else None

        return {
            'class_counts': class_counts,
            'avg_confidence': avg_confidence,
            'most_common': most_common,
            'total_recent': len(recent)
        }

    def draw_info_overlay(self, frame):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]

        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Total predictions
        cv2.putText(frame, f"Predictions: {self.total_predictions}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Recent stats
        stats = self.get_recent_stats()
        if stats and stats['most_common']:
            text = f"Recent: {stats['most_common']} ({stats['avg_confidence']:.1f}%)"
            cv2.putText(frame, text, (10, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 'r' to reset",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"satisfaction_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Frame saved: {filename}")
        return filename

    def reset_stats(self):
        """Reset all statistics"""
        self.prediction_history.clear()
        self.total_predictions = 0
        self.session_start_time = time.time()
        print("🔄 Statistics reset")

    def send_session_to_api(self):
        """Send session summary to the API"""
        if not self.api_url:
            return False

        try:
            session_duration = time.time() - self.session_start_time

            # Prepare session data
            session_data = {
                "session_duration": round(session_duration, 2),
                "total_predictions": self.total_predictions,
                "satisfied_count": 0,
                "neutral_count": 0,
                "unsatisfied_count": 0,
                "average_confidence": 0.0,
                "most_common_prediction": "unknown"
            }

            # Add session_id if provided
            if self.session_id:
                session_data["session_id"] = self.session_id

            if self.prediction_history:
                predictions = list(self.prediction_history)

                # Count by class
                for p in predictions:
                    label = p.get('label', '')
                    if label == 'Satisfied':
                        session_data["satisfied_count"] += 1
                    elif label == 'Neutral':
                        session_data["neutral_count"] += 1
                    elif label == 'Unsatisfied':
                        session_data["unsatisfied_count"] += 1

                # Calculate average confidence
                confidences = [p.get('confidence', 0) for p in predictions]
                session_data["average_confidence"] = round(np.mean(confidences) / 100, 4)

                # Most common prediction
                class_counts = {
                    'satisfied': session_data["satisfied_count"],
                    'neutral': session_data["neutral_count"],
                    'unsatisfied': session_data["unsatisfied_count"]
                }
                if any(class_counts.values()):
                    session_data["most_common_prediction"] = max(class_counts, key=class_counts.get)

            # Send to API
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{self.api_url}/api/session-summaries",
                headers=headers,
                json=session_data,
                timeout=20
            )

            if response.status_code == 201:
                print("✅ Session data sent to API successfully!")
                return True
            else:
                print(f"⚠️ API returned status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending data to API: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return False

    def test_api_connection(self):
        """Test if the API is reachable"""
        if not self.api_url:
            return False

        try:
            # Try to get statistics endpoint to test connection
            response = requests.get(
                f"{self.api_url}/api/session-summaries/statistics",
                timeout=20
            )
            if response.status_code == 200:
                print("✅ API connection test successful!")
                return True
            else:
                print(f"⚠️ API test returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ API connection test failed: {str(e)}")
            return False

    def print_session_summary(self):
        """Print summary of the session"""
        session_duration = time.time() - self.session_start_time

        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        print(f"⏱️  Duration: {session_duration:.1f} seconds")
        print(f"📈 Total predictions: {self.total_predictions}")

        if self.prediction_history:
            predictions = list(self.prediction_history)

            # Count by class
            class_counts = {}
            for label in self.satisfaction_labels.values():
                count = sum(1 for p in predictions if p.get('label') == label)
                class_counts[label] = count

            print("\n🏷️ Prediction distribution:")
            for label, count in class_counts.items():
                percentage = (count / len(predictions)) * 100 if predictions else 0
                print(f"   {label}: {count} ({percentage:.1f}%)")

            # Average confidence
            avg_confidence = np.mean([p.get('confidence', 0) for p in predictions])
            print(f"\n📊 Average confidence: {avg_confidence:.1f}%")

            # Most common
            if class_counts:
                most_common = max(class_counts, key=class_counts.get)
                print(f"🎯 Most common: {most_common}")
        else:
            print("❌ No predictions recorded")

        print("="*50)

        # Send to API if configured
        if self.api_url:
            print("\n📡 Sending session data to API...")
            self.send_session_to_api()

def test_camera_availability():
    """Test which cameras are available"""
    print("🔍 Testing camera availability...")

    available_cameras = []
    for i in range(5):  # Test first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                available_cameras.append({
                    'id': i,
                    'resolution': f"{width}x{height}"
                })
                print(f"✅ Camera {i}: Available ({width}x{height})")
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")

    return available_cameras

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Customer Satisfaction Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--model', default='satisfaction_model_best.h5', help='Model path')
    parser.add_argument('--test-cameras', action='store_true', help='Test camera availability')
    parser.add_argument('--min-face-size', type=int, default=80, help='Minimum face size for detection')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every Nth frame (for performance)')
    parser.add_argument('--api-url', type=str, default='https://laravel-api.fly.dev', help='API base URL for session summaries (default: https://laravel-api.fly.dev)')
    parser.add_argument('--session-id', type=str, help='Optional session ID for tracking')

    args = parser.parse_args()

    # Test cameras if requested
    if args.test_cameras:
        available_cameras = test_camera_availability()
        if not available_cameras:
            print("❌ No cameras found!")
            return
        else:
            print(f"\n📹 Found {len(available_cameras)} available camera(s)")
            return

    # Initialize detector
    print(f"🚀 Initializing satisfaction detector...")
    detector = SatisfactionDetector(args.model, args.api_url, args.session_id)

    # Print API configuration
    if args.api_url:
        print(f"📡 API URL configured: {args.api_url}")
        if args.session_id:
            print(f"🆔 Session ID: {args.session_id}")
        else:
            print("🆔 No session ID provided (will be auto-generated by API)")

        # Test API connection
        print("🔍 Testing API connection...")
        api_available = detector.test_api_connection()
        if not api_available:
            print("⚠️ API not available - session data will only be displayed locally")
    else:
        print("📡 API integration disabled - session data will only be displayed locally")

    # Initialize webcam
    print(f"📹 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}.")
        print("💡 Try running with --test-cameras to see available cameras")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Error: Could not load face cascade classifier.")
        return

    print("✅ Camera opened successfully!")
    print("\n📋 Controls:")
    print("   • Press 'q' to quit")
    print("   • Press 's' to save current frame")
    print("   • Press 'r' to reset statistics")
    print("   • Press 'h' to toggle help overlay")
    if args.api_url:
        print("   • Session data will be automatically sent to API when quitting")
    print("\n🎯 Starting real-time detection...")

    frame_counter = 0
    saved_frames = 0
    show_help = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Can't receive frame. Exiting...")
                break

            # Update FPS
            detector.update_fps()
            frame_counter += 1

            # Process every Nth frame for better performance
            if frame_counter % args.skip_frames == 0:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces with improved parameters
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(args.min_face_size, args.min_face_size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # Process largest face only for better performance
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face

                    try:
                        # Extract face ROI
                        face_roi = gray[y:y+h, x:x+w]

                        # Resize to 48x48
                        resized_face = cv2.resize(face_roi, (48, 48))

                        # Get prediction
                        pred_label, confidence, probabilities = detector.predict_satisfaction(resized_face)

                        if pred_label:
                            # Add to history
                            prediction_data = {
                                'label': pred_label,
                                'confidence': confidence,
                                'probabilities': probabilities.tolist()
                            }
                            detector.add_prediction(prediction_data)

                            # Get color based on prediction
                            color = detector.colors[pred_label]

                            # Draw rectangle around face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                            # Display prediction and confidence
                            text = f"{pred_label}: {confidence:.1f}%"

                            # Background for text
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(frame, (x, y-text_size[1]-10),
                                        (x + text_size[0], y), color, -1)

                            # Text
                            cv2.putText(frame, text, (x, y-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"⚠️ Error processing face: {str(e)}")
                        continue

            # Draw info overlay
            if show_help:
                frame = detector.draw_info_overlay(frame)

            # Display frame
            cv2.imshow('Customer Satisfaction Analysis - Real-Time', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = detector.save_frame(frame)
                saved_frames += 1
            elif key == ord('r'):
                detector.reset_stats()
            elif key == ord('h'):
                show_help = not show_help
                print(f"📋 Help overlay: {'ON' if show_help else 'OFF'}")

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print session summary
        detector.print_session_summary()

        if saved_frames > 0:
            print(f"💾 Total frames saved: {saved_frames}")

        print("👋 Thank you for using Customer Satisfaction Detector!")

if __name__ == "__main__":
    main()