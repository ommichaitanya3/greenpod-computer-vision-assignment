import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Orange Image Classifier")
st.warning("Dataset contains orange classes, not avocado ripeness classes.")

classes = ["blackspot", "canker", "fresh", "grenning"]
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    model.load_state_dict(
        torch.load("avocado_mobilenetv2.pth", map_location=device)
    )
    model.eval()
    return model

uploaded_file = st.file_uploader(
    "Upload an orange image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model = load_model()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            index = probabilities.argmax().item()

        st.success(f"Prediction: {classes[index]}")
        st.write(f"Confidence: {probabilities[index].item():.2%}")

    except Exception as e:
        st.error(f"Could not load model or predict image: {e}")
