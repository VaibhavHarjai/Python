import cv2
import numpy as np
import os
import pickle
import imutils
import time
from imutils.video import VideoStream
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FaceMaskDetector:
    def __init__(self):
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Model paths
        self.model_path = "face_mask_detector.pkl"
        self.le_path = "label_encoder.pkl"

        # Load model if exists, else set to None
        if os.path.exists(self.model_path) and os.path.exists(self.le_path):
            self.model = pickle.load(open(self.model_path, "rb"))
            self.le = pickle.load(open(self.le_path, "rb"))
            print(" Model loaded successfully!")
        else:
            print(" Model files not found. Please train the model first.")
            self.model = None
            self.le = None
    
    def extract_features(self, image):
        """Extracts HOG features from an image."""
        if image is None or image.size == 0:
            return np.zeros((1764,), dtype=np.float32)  # Return zero features if image is invalid

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            resized = cv2.resize(gray, (64, 64))
            
            # HOG descriptor parameters
            hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
            features = hog.compute(resized)
            return features.flatten()
        except Exception as e:
            print(f" Feature extraction error: {e}")
            return np.zeros((1764,), dtype=np.float32)
    
    def train(self, dataset_path):
        """Trains the face mask detection model."""
        print(" Training the face mask detector model...")
        data, labels = [], []

        if not os.path.exists(dataset_path):
            print(f" Dataset path '{dataset_path}' does not exist!")
            return
        
        for category in os.listdir(dataset_path):
            path = os.path.join(dataset_path, category)
            if not os.path.isdir(path):
                continue

            print(f" Processing {category} images...")
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    print(f" Skipped unreadable image: {img_path}")
                    continue
                
                features = self.extract_features(image)
                if np.all(features == 0):
                    continue
                
                data.append(features)
                labels.append(category)
        
        if len(data) == 0:
            print(" No valid images found in dataset. Training aborted.")
            return

        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        # 🔹 FIXED: Correct label encoding
        self.le = LabelEncoder()
        self.le.fit(["With_Mask", "Without_Mask"])  # Ensure correct label order
        labels_encoded = self.le.transform(labels)  # Convert labels to numbers

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.20, random_state=42)

        # Train the model
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
        self.model = SVC(kernel="linear", probability=True)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

        # Save the model and label encoder
        pickle.dump(self.model, open(self.model_path, "wb"))
        pickle.dump(self.le, open(self.le_path, "wb"))
        print(" Model and label encoder saved successfully.")

    def detect_and_predict_mask(self, frame):
        """Detects faces and predicts mask status."""
        orig_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0 or self.model is None:
            return orig_frame

        for (x, y, w, h) in faces:
            face = orig_frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # Extract features and make prediction
            features = self.extract_features(face).reshape(1, -1)
            preds = self.model.predict(features)[0]  # Get class index
            label = self.le.inverse_transform([preds])[0]  # Convert back to label

            # Set color based on label
            if label == "With_Mask":
                color = (0, 255, 0)  # Green for Mask
            else:
                color = (0, 0, 255)  # Red for No Mask

            # Draw rectangle and text
            label_text = f"{label}"
            cv2.rectangle(orig_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(orig_frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return orig_frame

    def start_video_detection(self):
        """Starts real-time face mask detection."""
        if self.model is None:
            print(" Error: Model not loaded. Please train the model first.")
            return

        print(" Starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=600)

            output_frame = self.detect_and_predict_mask(frame)

            cv2.imshow("Face Mask Detection", output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()
        print("Video stream stopped.")

def main():
    detector = FaceMaskDetector()

    # 🔹 Train the model if needed
    detector.train("dataset")  # Ensure "dataset" folder exists

    # 🔹 Start real-time mask detection
    detector.start_video_detection()

if __name__ == "__main__":
    main()
