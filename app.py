import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from io import BytesIO
from PIL import Image
import base64

st.set_page_config(page_title="Uniformity Index Analyzer", layout="centered")
st.title("📷 Uniformity Index Analyzer")

if 'history' not in st.session_state:
    st.session_state.history = []

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def calculate_ui(image, max_division=6):
    h, w = image.shape
    chi_values = []
    for i in range(2, max_division + 1):
        n = i * i
        block_h, block_w = h // i, w // i
        if block_h == 0 or block_w == 0:
            continue
        means = []
        for y in range(i):
            for x in range(i):
                block = image[y*block_h:(y+1)*block_h, x*block_w:(x+1)*block_w]
                if block.size > 0:
                    means.append(np.mean(block))
        means = np.array(means)
        mean_val = np.mean(means)
        var_val = np.var(means)
        I = var_val / mean_val if mean_val != 0 else 0
        chi = I * (n - 1)
        chi_values.append(chi)

    if not chi_values:
        return 0.0, []

    A = np.trapz(chi_values)
    a0 = 10
    UI = (1 / (1 + (A / a0))) * 100
    return UI, chi_values

def plot_chi_curve(chi_curve):
    fig, ax = plt.subplots()
    ax.plot(range(2, 2 + len(chi_curve)), chi_curve, marker='o')
    ax.set_title("Chi²-Kurve über Quadrantenanzahl")
    ax.set_xlabel("Teilung (z. B. 2x2, 3x3 ...)")
    ax.set_ylabel("Chi²-Wert")
    ax.grid(True)
    st.pyplot(fig)

def display_image(img_array):
    st.image(img_array, caption="Hochgeladenes Bild", use_column_width=True)

st.subheader("Bild hochladen und analysieren")

uploaded_file = st.file_uploader("Bild auswählen", type=["png", "jpg", "jpeg", "tif"])
apply_hist_eq = st.checkbox("Histogramm glätten (empfohlen)", value=True)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    if apply_hist_eq:
        image_gray = histogram_equalization(image_gray)

    UI, chi_curve = calculate_ui(image_gray)

    st.markdown(f"### 📊 Uniformity Index: **{UI:.2f} %**")
    display_image(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plot_chi_curve(chi_curve)

    # Speichere Verlaufseintrag
    buffered = BytesIO()
    img_pil = Image.fromarray(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    st.session_state.history.append({
        "ui": UI,
        "img": img_str,
        "chi": chi_curve
    })

# Historie anzeigen
if st.session_state.history:
    st.subheader("📚 Analyse-Historie")
    for i, entry in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**{i+1}. Ergebnis:** Uniformity Index = **{entry['ui']:.2f} %**")
        st.image(Image.open(BytesIO(base64.b64decode(entry['img']))), width=200)
        with st.expander("Details anzeigen"):
            st.line_chart(entry['chi'])
