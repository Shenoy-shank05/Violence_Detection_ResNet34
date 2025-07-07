import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import os

# ==== Config ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path = "/content/drive/MyDrive/violence_detection_resnet/personel.mp4"
output_path = "/content/drive/MyDrive/violence_detection_resnet/processed_video.mp4"
frame_rate_skip = 1 
threshold = 0.5  

# ==== Load Model ====
fc_units = 448
dropout_rate = 0.2345

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, fc_units),
    nn.ReLU(),
    nn.Dropout(dropout_rate),
    nn.Linear(fc_units, 1)
)

model.load_state_dict(torch.load("/content/drive/MyDrive/violence_detection_resnet/full_checkpoint_final.pth", map_location=device)['model_state_dict'])
model.to(device)
model.eval()

# ==== Preprocessing ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Video Setup ====
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Every nth frame (for speed), otherwise copy previous prediction
        if frame_count % frame_rate_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

            label = "Violence" if prob > threshold else "Non-Violence"
            color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
            label_to_draw = f"{label} ({prob:.2f})"
        # Draw label on all frames
        cv2.putText(frame, label_to_draw, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processed and saved at:", output_path)
