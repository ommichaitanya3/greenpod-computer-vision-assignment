import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Computer Vision Assignment",
    page_icon="🍊",
    layout="wide"
)

st.title("Computer Vision Classification")
st.write("Greenpod Labs Assignment — Image Classification")

st.warning(
    "Dataset note: The available dataset contains orange classes "
    "(blackspot, canker, fresh, grenning), not avocado ripening stages. "
    "Results therefore apply to the provided orange dataset only."
)

st.header("Approach 1: Classical Machine Learning")
st.write(
    "Feature extraction: CIELAB color histograms and Local Binary Patterns (LBP). "
    "Models: Support Vector Machine (SVM) and Random Forest."
)

st.header("Approach 2: Lightweight Deep Learning")
st.write(
    "MobileNetV2 transfer learning was evaluated on the same provided dataset."
)

st.header("Model Results")
st.write("Random Forest test accuracy: approximately 96.97% (99 test images).")
st.write("MobileNetV2 test accuracy: approximately 94.95% (99 test images).")

st.info(
    "These results are from the existing evaluation. Performance on new images "
    "or real-world avocado ripeness has not been established."
)

st.header("Project Files")
for filename in [
    "avocado_ripening_classification.py",
    "Avocado_Ripening_Approach2.ipynb",
    "Project_Report.md",
    "README.md",
]:
    st.write(f"- {filename}")

st.caption("Educational demonstration — not validated for production use.")
