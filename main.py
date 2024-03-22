import cv2
import os

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict gender and age for a given face
def predict_gender_age(face):
    # This is a placeholder function. You need to replace it with your actual gender and age prediction model.
    # Preprocess the face if needed
    # gender, age = gender_age_model.predict(face)
    gender = "Male"  # Replace this with the actual gender prediction
    age = 25  # Replace this with the actual age prediction
    return gender, age

def detect_faces_and_predict_gender_age(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict gender and age for the face
        gender, age = predict_gender_age(face_roi)
        
        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display gender and age predictions
        cv2.putText(image, f'{gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate the appropriate window size based on the image dimensions
    window_width = image.shape[1]  # Width of the image
    window_height = image.shape[0]  # Height of the image

    # Set the window size
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', window_width, window_height)

    # Display the final image with detected faces and predictions
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Get the list of image files in the 'faces' folder
folder_path = 'faces'
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

# Detect faces and predict gender and age for each image
for image_file in image_files:
    detect_faces_and_predict_gender_age(image_file)
