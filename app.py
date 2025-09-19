
import os, requests

MODEL_PATH = "weather_model.pth"
MODEL_URL = "https://drive.google.com/file/d/1obrYT-31tbPUEnmVuPiA_F1GLr5ZEZHO/view?usp=sharing"

if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

MODEL_PATH = "weather_model.pth"

st.title("Sky Photo -> Weather Recommendation")
st.write("Upload a photo of the sky and get a weather prediction with advice.")

@st.cache_resource
def load_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes

model, classes = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

recommendation = {
    "Cloudy": "Cloudy: mild weather; a light jacket may be useful.",
    "Rain": "Rain expected: carry an umbrella or raincoat.",
    "Shine": "Bright and sunny: wear sunscreen and sunglasses.",
    "Sunrise": "Sunrise: great time for a walk and take photos."
}

uploaded = st.file_uploader("Upload sky photo", type=["jpg","jpeg","png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded photo", use_column_width=True)
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        idx = int(outputs.argmax(dim=1).item())
        pred_class = classes[idx]
    st.subheader(f"Predicted: {pred_class}")
    st.success(recommendation.get(pred_class, "No specific advice for this class."))

