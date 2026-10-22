
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Orange Image Classifier",
    page_icon="🍊",
    layout="centered"
)

st.title("🍊 Orange Image Classifier")
st.write("Upload an orange image to predict its class.")

st.warning(
    "Note: The available dataset contains orange conditions, "
    "not avocado ripening stages."
)

CLASSES = ["blackspot", "canker", "fresh", "grenning"]
MODEL_PATH = "avocado_mobilenetv2.pth"
DEVICE = torch.device("cpu")


@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Load the trained model weights from the checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Choose an orange image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        with st.spinner("Classifying image..."):
            model = load_model()
            image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_index = probabilities.argmax().item()

        st.success(f"Predicted class: {CLASSES[predicted_index]}")
        st.metric(
            "Prediction confidence",
            f"{probabilities[predicted_index].item():.2%}"
        )

        st.subheader("Class probabilities")
        for index, class_name in enumerate(CLASSES):
            st.write(f"**{class_name}:** {probabilities[index].item():.2%}")
            st.progress(float(probabilities[index].item()))

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the model file exists in the GitHub repository "
            "and its weights match the MobileNetV2 architecture."
        )
