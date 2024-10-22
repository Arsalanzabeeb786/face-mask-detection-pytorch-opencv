import os
import torch
from PIL import Image
import numpy as np
import cv2  # OpenCV for webcam
from torchvision import transforms
import time  # To introduce delay

# Load the saved model and set it to evaluation mode
trained_model = torch.load('models/val_accuracy_0.9935_model.pth')
trained_model.eval()

# Define image preprocessing transformations (ensure it matches training-time transforms)
preprocess = transforms.Compose([
    transforms.Resize((342, 342)),  # Assuming 224x224 input size
    transforms.CenterCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names for the two classes
class_names = ['WithMask', 'WithoutMask']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = trained_model.to(device)

# Function to run inference and return predicted label
def predict_frame(frame, model):
    # Convert the frame from OpenCV (BGR) to PIL format (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Preprocess the image
    image_tensor = preprocess(pil_image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, torch.Tensor):
            logits = output
        else:
            logits = output.logits

    # Get predicted class (Binary Classification)
    predicted = (logits > 0.0).float()  # The class with the highest score
    prediction = int(predicted.item())

    # Get the predicted label
    predicted_label = class_names[prediction]

    return predicted_label

# Open webcam video feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform prediction on the frame
    predicted_label = predict_frame(frame, trained_model)

    # Put the prediction label on the top-left corner of the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with the label
    cv2.imshow('Mask Detection', frame)

    # Wait for 1 second before processing the next frame
    #time.sleep(1)

    # Press 'q' to exit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
