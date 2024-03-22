import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def predict_gender_age(face):
   
    gender = "Male"  
    age = 25  
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
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict gender and age for the face
        gender, age = predict_gender_age(face_roi)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display gender and age predictions
        cv2.putText(image, f'{gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    window_width = image.shape[1]  
    window_height = image.shape[0]  
    
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', window_width, window_height)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
folder_path = 'faces'
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
# Detect faces and predict gender and age for each image
for image_file in image_files:
    detect_faces_and_predict_gender_age(image_file)
