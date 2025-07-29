import os
import cv2
import numpy as np
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration
DATA_DIR = 'hand_sign_data'
MODEL_PATH = 'hand_sign_model.h5'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

def get_droidcam_connection():
    """Try to find DroidCam (USB) webcam index between 0–4"""
    print("🔍 Scanning for DroidCam virtual camera (USB)...")

    for index in range(5):
        cap = cv2.VideoCapture(index)
        time.sleep(1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Successfully connected to DroidCam at index {index}")
                cv2.imshow('DroidCam Preview (Press any key)', frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                return cap
            else:
                print(f"⚠ Opened index {index} but no frame")
        else:
            print(f"❌ Cannot open camera index {index}")
        cap.release()

    print("❌ Failed to connect to DroidCam via USB.")
    print("Make sure DroidCam is running and connected via USB.")
    return None

def create_model(input_shape, num_classes):
    """Create CNN model for hand sign recognition"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def collect_data(class_name, num_samples=200):
    """Data collection from DroidCam USB virtual webcam"""
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    cap = get_droidcam_connection()
    if cap is None:
        return
    
    print(f"\nCollecting {num_samples} samples for '{class_name}'")
    print("Press 's' to save, 'q' to quit")
    
    count = len(os.listdir(class_dir))
    try:
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Frame capture failed, retrying...")
                time.sleep(1)
                continue
                
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)
            
            # Display instructions
            display = frame.copy()
            cv2.putText(display, f"Saved: {count}/{num_samples}", (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display, f"Class: {class_name}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display, "Press 's' to save, 'q' to quit", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Data Collection', display)
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(os.path.join(class_dir, f"{class_name}_{count}.jpg"), gray)
                count += 1
                print(f"Saved {count}/{num_samples}")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinished collecting {count} samples")

def train_model():
    """Train the model with data augmentation"""
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("\n❌ No training data found. Collect data first!")
        return None, None
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    classes = sorted(os.listdir(DATA_DIR))
    print("\nTraining with classes:", ", ".join(classes))
    
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
    
    model = create_model((*IMG_SIZE, 1), len(classes))
    
    print("\nStarting training... (This may take several minutes)")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, save_best_only=True),
            EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    print(f"\n✅ Training complete! Model saved to {MODEL_PATH}")
    return model, classes

def recognize_hand_sign(model, classes):
    """Real-time recognition using DroidCam USB"""
    cap = get_droidcam_connection()
    if cap is None:
        return
    
    print("\nStarting recognition... Press 'q' to quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Frame capture failed, retrying...")
                time.sleep(1)
                continue
                
            # Process and predict
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, IMG_SIZE)
            input_img = np.expand_dims(resized/255.0, axis=(0,-1))
            
            preds = model.predict(input_img, verbose=0)[0]
            pred_class = classes[np.argmax(preds)]
            confidence = np.max(preds)
            
            # Display results
            display = frame.copy()
            cv2.putText(display, f"Sign: {pred_class}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display, f"Confidence: {confidence:.1%}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display, "Press 'q' to quit", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Hand Sign Recognition', display)
            
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("\n" + "="*50)
    print("Hand Sign Recognition with DroidCam (USB Mode)")
    print("="*50)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    while True:
        print("\nMenu:")
        print("1. Collect new hand sign data")
        print("2. Train model")
        print("3. Recognize hand signs")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter hand sign name (e.g., 'peace', 'thumbs_up'): ").strip()
            if name:
                collect_data(name)
            else:
                print("❌ Invalid name")
        elif choice == '2':
            train_model()
        elif choice == '3':
            if not os.path.exists(MODEL_PATH):
                print("❌ No trained model found. Train first!")
                continue
            model = load_model(MODEL_PATH)
            classes = sorted(os.listdir(DATA_DIR))
            recognize_hand_sign(model, classes)
        elif choice == '4':
            print("\nExiting...")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    main()
