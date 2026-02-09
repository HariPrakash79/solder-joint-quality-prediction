import yaml
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import urllib.request

st.set_page_config(page_title="Solder Joint QC", layout="centered")


def load_config(path="config.yaml"):
    return yaml.safe_load(Path(path).read_text())


def ensure_model(model_path, model_url=None):
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    if not model_url:
        return None
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(model_url) as r, open(model_path, "wb") as f:
        f.write(r.read())
    return model_path


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(img: Image.Image, img_size: int):
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, 0)


def main():
    st.title("Solder Joint Quality Demo")
    st.write("Upload an X-ray image and get a defect prediction.")

    cfg = load_config()
    model_path = cfg["model_path"]
    model_url = cfg.get("model_url")
    img_size = int(cfg["img_size"])
    thr = float(cfg["decision_threshold"])
    class_names = cfg["class_names"]

    resolved = ensure_model(model_path, model_url)
    if resolved is None or not Path(resolved).exists():
        st.error("Model file not found. Set model_url in config.yaml or place the model at the path.")
        st.stop()

    model = load_model(str(resolved))

    uploaded = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Input image", use_container_width=True)
        x = preprocess_image(img, img_size)
        prob = float(model.predict(x, verbose=0).ravel()[0])
        label = class_names[1] if prob >= thr else class_names[0]
        st.metric("Defect probability", f"{prob:.3f}")
        st.metric("Prediction", label)
        st.caption(f"Decision threshold = {thr}")


if __name__ == "__main__":
    main()