import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

# Must be first Streamlit command
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")

# ------------------------------
# Safe model path & load model
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "ham10000_efficientnet.h5")

@st.cache_resource
def load_everything():
    model = tf.keras.models.load_model(MODEL_PATH)

    effnet = model.get_layer("efficientnetb0")
    last_conv_layer = effnet.get_layer("top_conv")  # last conv

    # Grad model: EfficientNet input -> last conv + entire effnet output
    grad_model = tf.keras.models.Model(
        [effnet.input],
        [last_conv_layer.output, effnet.output]
    )

    preprocess = model.get_layer("sequential")
    gap = model.get_layer("global_average_pooling2d")
    drop = model.get_layer("dropout")
    dense = model.get_layer("dense")

    return {
        "model": model,
        "preprocess": preprocess,
        "grad_model": grad_model,
        "gap": gap,
        "drop": drop,
        "dense": dense,
    }

ctx = load_everything()
model = ctx["model"]
preprocess = ctx["preprocess"]
grad_model = ctx["grad_model"]
gap_layer = ctx["gap"]
drop_layer = ctx["drop"]
dense_layer = ctx["dense"]

# ------------------------------
# Full HAM10000 labels
# ------------------------------
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

LABELS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "vasc":  "Vascular Lesion"
}

# ------------------------------
# Grad-CAM function
# ------------------------------
def compute_gradcam(img_array, class_index):

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    x = preprocess(img_tensor)

    with tf.GradientTape() as tape:
        conv_output, eff_out = grad_model(x)
        tape.watch(conv_output)

        pooled = gap_layer(eff_out)
        pooled = drop_layer(pooled, training=False)
        preds = dense_layer(pooled)

        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)
    conv_output = conv_output[0].numpy()
    grads = grads[0].numpy()

    weights = np.mean(grads, axis=(0, 1))

    cam = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)

    return cam


# ------------------------------
# UI
# ------------------------------
st.title("🔬 Skin Cancer Detection App")
st.write("Upload a dermoscopy image to classify it and visualize Grad-CAM.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=360)

    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_np, axis=0)

    preds = model.predict(img_batch)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])

    # FULL NAME OUTPUT
    pred_key = CLASS_ORDER[idx]
    full_name = LABELS[pred_key]

    st.subheader("Prediction")
    st.success(f"{full_name} ({confidence*100:.2f}% confidence)")

    if st.button("Show Grad-CAM Heatmap"):
        try:
            cam = compute_gradcam(img_batch, idx)

            cam_resized = cv2.resize(cam, (img.width, img.height))
            cam_uint8 = np.uint8(255 * cam_resized)

            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # ONLY SHOW HEATMAP — original removed
            st.subheader("🔥 Grad-CAM Heatmap (JET)")
            st.image(heatmap_rgb, width=360)

        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")
