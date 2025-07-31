import os
import cv2
import numpy as np
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2

# Configuration
DATA_DIR = 'hand_data'
MODEL_PATH = 'finger_gesture_model.h5'
IMG_SIZE = (128, 128)  # Increased resolution for better finger details
BATCH_SIZE = 32
EPOCHS = 30

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_webcam_connection():
    """Initialize webcam connection"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return None
    
    # Set higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap

def create_advanced_model(input_shape, num_classes):
    """Create a more sophisticated CNN model for finger counting"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Conv2D(256, (3,3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def collect_finger_data(gesture_name, num_samples=500):
    """Collect data using MediaPipe for hand cropping"""
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    
    cap = get_webcam_connection()
    if cap is None:
        return
    
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)
    
    print(f"\nCollecting {num_samples} samples for '{gesture_name}'")
    print("Show your hand gesture to the camera")
    print("Press 's' to save, 'q' to quit")
    
    count = len(os.listdir(gesture_dir))
    try:
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            display = frame.copy()
            hand_landmarks = None
            
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        display, landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get hand bounding box
                    x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
                    y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]
                    min_x, max_x = int(min(x_coords)), int(max(x_coords))
                    min_y, max_y = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding to bounding box
                    padding = 30
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(frame.shape[1], max_x + padding)
                    max_y = min(frame.shape[0], max_y + padding)
                    
                    # Draw bounding box
                    cv2.rectangle(display, (min_x, min_y), (max_x, max_y), (0,255,0), 2)
                    hand_landmarks = landmarks
            
            # Display instructions
            cv2.putText(display, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display, f"Saved: {count}/{num_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display, "Press 's' to save, 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow('Finger Data Collection', display)
            
            key = cv2.waitKey(1)
            if key == ord('s') and hand_landmarks is not None:
                # Crop and save hand region
                hand_roi = frame[min_y:max_y, min_x:max_x]
                if hand_roi.size > 0:
                    # Convert to grayscale and resize
                    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, IMG_SIZE)
                    
                    # Apply adaptive thresholding to enhance fingers
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
                    
                    save_path = os.path.join(gesture_dir, f"{gesture_name}_{count}.jpg")
                    cv2.imwrite(save_path, gray)
                    count += 1
                    print(f"Saved {count}/{num_samples}")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

def train_finger_model():
    """Train the finger counting model with advanced augmentation"""
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) < 3:
        print("\n❌ Need at least 3 different gestures to train")
        return None, None
    
    # Advanced data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20
    )
    
    gesture_classes = sorted(os.listdir(DATA_DIR))
    print("\nTraining with gestures:", ", ".join(gesture_classes))
    
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='grayscale'
    )
    
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='grayscale'
    )
    
    model = create_advanced_model((*IMG_SIZE, 1), len(gesture_classes))
    
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, save_best_only=True),
            EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    print(f"\n✅ Training complete! Model saved to {MODEL_PATH}")
    return model, gesture_classes

def count_fingers(landmarks):
    """Count extended fingers using hand landmarks"""
    finger_tips = [4, 8, 12, 16, 20]  # Landmark indices for finger tips
    finger_dips = [3, 7, 11, 15, 19]  # Landmark indices for finger dips
    
    extended_fingers = 0
    
    # Thumb (special case)
    if landmarks.landmark[finger_tips[0]].x < landmarks.landmark[finger_dips[0]].x:
        extended_fingers += 1
    
    # Other fingers
    for tip, dip in zip(finger_tips[1:], finger_dips[1:]):
        if landmarks.landmark[tip].y < landmarks.landmark[dip].y:
            extended_fingers += 1
    
    return extended_fingers

def recognize_finger_gesture(model, gesture_classes):
    """Real-time finger counting and gesture recognition"""
    cap = get_webcam_connection()
    if cap is None:
        return
    
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)
    
    print("\nStarting recognition... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            display = frame.copy()
            finger_count = 0
            gesture_pred = "No hand"
            confidence = 0
            
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        display, landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Count fingers
                    finger_count = count_fingers(landmarks)
                    
                    # Get hand ROI for gesture recognition
                    x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
                    y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]
                    min_x, max_x = int(min(x_coords)), int(max(x_coords))
                    min_y, max_y = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding and ensure within frame bounds
                    padding = 50
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(frame.shape[1], max_x + padding)
                    max_y = min(frame.shape[0], max_y + padding)
                    
                    # Crop and process hand region
                    hand_roi = frame[min_y:max_y, min_x:max_x]
                    if hand_roi.size > 0:
                        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, IMG_SIZE)
                        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)
                        
                        # Predict gesture
                        input_img = np.expand_dims(gray/255.0, axis=(0, -1))
                        preds = model.predict(input_img, verbose=0)[0]
                        gesture_pred = gesture_classes[np.argmax(preds)]
                        confidence = np.max(preds)
            
            # Display results
            cv2.putText(display, f"Fingers: {finger_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(display, f"Gesture: {gesture_pred}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(display, f"Confidence: {confidence:.1%}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(display, "Press 'q' to quit", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow('Finger Counting & Gesture Recognition', display)
            
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

def main():
    print("\n" + "="*50)
    print("Advanced Finger Counting & Gesture Recognition System")
    print("="*50)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    while True:
        print("\nMenu:")
        print("1. Collect new finger gesture data")
        print("2. Train model")
        print("3. Recognize fingers and gestures")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter gesture name (e.g., 'fist', 'five', 'peace'): ").strip()
            if name:
                collect_finger_data(name)
            else:
                print("❌ Please enter a valid name")
        elif choice == '2':
            train_finger_model()
        elif choice == '3':
            if not os.path.exists(MODEL_PATH):
                print("❌ No trained model found. Train first!")
                continue
            model = load_model(MODEL_PATH)
            gesture_classes = sorted(os.listdir(DATA_DIR))
            recognize_finger_gesture(model, gesture_classes)
        elif choice == '4':
            print("\nExiting...")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4")

if __name__ == "__main__":
    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    main()
