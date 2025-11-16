import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from torch.nn.functional import softmax

# --- INITIALIZE THE FLASK APP ---
app = Flask(__name__)
CORS(app)

# --- SETUP MODELS AND CONSTANTS ---
print("="*30)
print("Loading models and setting up environment...")

# Constants
FRAMES_PER_VIDEO = 32
IMAGE_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load BlazeFace
from blazeface import BlazeFace
facedet = BlazeFace().to(device); facedet.load_weights("blazeface.pth"); facedet.load_anchors("anchors.npy"); facedet.eval()
print(f"Running on device: {device}"); print("BlazeFace detector loaded.")

# --- START OF FIX ---
# 1. Define the model architectures EXACTLY as they were in the training notebook
model_a = timm.create_model('xception', pretrained=False, num_classes=2)
model_b = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)

# 2. Load the state_dict (weights only) into the architectures
model_a.load_state_dict(torch.load('./models/xception_weights.pth', map_location=device))
model_b.load_state_dict(torch.load('./models/efficientnet_b4_weights.pth', map_location=device))

# 3. Move models to device and set to evaluation mode
model_a = model_a.to(device); model_a.eval()
model_b = model_b.to(device); model_b.eval()
# --- END OF FIX ---

print("XceptionNet and EfficientNet models loaded successfully.")

# Define transforms
data_transforms = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
class_names = ['FAKE', 'REAL']

print("Setup complete. Server is ready.")
print("="*30)


# --- THE PREDICTION FUNCTION (No changes needed here) ---
def predict_video_ensemble(video_path):
    faces = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return "Invalid video file", None
        frame_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            detections = facedet.predict_on_image(cv2.resize(frame, (128, 128)))
            if len(detections) > 0:
                best_detection = detections[0]
                h, w, _ = frame.shape
                coords = (best_detection[:4] * torch.tensor([h, w, h, w]).to(device)).cpu().numpy().astype(int)
                ymin, xmin, ymax, xmax = coords
                face = frame[ymin:ymax, xmin:xmax]
                if face.size > 0: faces.append(face)
    finally:
        if 'cap' in locals(): cap.release()

    if not faces: return "No faces detected", None
    
    face_tensors = [data_transforms(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in faces]
    batch = torch.stack(face_tensors).to(device)
    with torch.no_grad():
        logits_a = model_a(batch); probs_a = softmax(logits_a, dim=1).mean(dim=0)
        logits_b = model_b(batch); probs_b = softmax(logits_b, dim=1).mean(dim=0)
    ensemble_probs = (probs_a + probs_b) / 2
    prediction_idx = torch.argmax(ensemble_probs).item()
    confidence = ensemble_probs[prediction_idx].item()
    predicted_class = class_names[prediction_idx]
    return predicted_class, confidence

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files: return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    uploads_dir = 'uploads'; os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, video_file.filename)
    video_file.save(video_path)
    prediction, confidence = predict_video_ensemble(video_path)
    os.remove(video_path)
    if confidence is None: return jsonify({'prediction': prediction, 'confidence': 0})
    else: return jsonify({'prediction': prediction, 'confidence': float(confidence)})

# --- START THE SERVER ---
if __name__ == '__main__':
    app.run(debug=True)