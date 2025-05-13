import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from io import BytesIO
from PIL import Image
import base64
from streamlit_drawable_canvas import st_canvas
from scipy.stats import chi2

st.set_page_config(page_title="Uniformity Index Analyzer", layout="centered")
st.title("📷 Uniformity Index Analyzer")

if 'history' not in st.session_state:
    st.session_state.history = []

# Interaktive Quadrantenaufteilung wählbar
vis_division = st.sidebar.slider("Visualisierte Teilung (z. B. 4 = 4x4)", min_value=2, max_value=100, value=4, step=1)


def calculate_ui(image, max_division=6, vis_div=4):
    h, w = image.shape
    chi_values = []
    quadrant_overlay = image.copy()
    quadrant_overlay = cv2.cvtColor(quadrant_overlay, cv2.COLOR_GRAY2RGB)
    heatmap_data = np.zeros((vis_div, vis_div))

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

    # Visualisierung basierend auf gewählter Teilung
    i = vis_div
    block_h, block_w = h // i, w // i
    for y in range(i):
        for x in range(i):
            top_left = (x*block_w, y*block_h)
            bottom_right = ((x+1)*block_w - 1, (y+1)*block_h - 1)
            cv2.rectangle(quadrant_overlay, top_left, bottom_right, (0, 255, 0), 1)
            block = image[y*block_h:(y+1)*block_h, x*block_w:(x+1)*block_w]
            heatmap_data[y, x] = np.mean(block)

    if not chi_values:
        return 0.0, [], None, None

        n_values = np.arange(2, 2 + len(chi_values))
    df_values = n_values - 1
    chi_crit_vals = chi2.ppf(0.0275, df_values)
    a0 = np.trapz(chi_crit_vals, x=n_values)
    A = np.trapz(chi_values, x=n_values)
    UI = (1 / (1 + (A / a0))) * 100
    return UI, chi_values, quadrant_overlay, heatmap_data, a0


def plot_chi_curve(chi_curve):
    fig, ax = plt.subplots()
    ax.plot(range(2, 2 + len(chi_curve)), chi_curve, marker='o')
    ax.set_title("Chi²-Kurve über Quadrantenanzahl")
    ax.set_xlabel("Teilung (z. B. 2x2, 3x3 ...)")
    ax.set_ylabel("Chi²-Wert")
    ax.grid(True)
    st.pyplot(fig)


def display_image(img_array):
    st.image(img_array, caption="Hochgeladenes Bild", use_container_width=True)


st.subheader("Bild hochladen und analysieren")

uploaded_file = st.file_uploader("Bild auswählen", type=["png", "jpg", "jpeg", "tif"])

# Wenn neues Bild hochgeladen wird, cropping_done zurücksetzen
if uploaded_file:
    st.session_state.cropping_done = False

# Zuschneideoption anbieten
use_crop = st.checkbox("🖼️ Bild zuschneiden vor der Analyse?", value=False)

crop_confirmed = False

if uploaded_file and use_crop and not st.session_state.get("cropping_done"):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_gray = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)

    # Randerkennung und Vorschau
    st.markdown("### ✏️ Interaktiver Zuschnitt")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=Image.fromarray(cv2.cvtColor(original_color.copy(), cv2.COLOR_BGR2RGB)).convert("RGBA"),
        update_streamlit=True,
        height=original_color.shape[0],
        width=original_color.shape[1],
        drawing_mode="rect",
        key="canvas_crop"
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        rect = canvas_result.json_data["objects"][-1]
        left = int(rect["left"])
        top = int(rect["top"])
        width = int(rect["width"])
        height = int(rect["height"])
        x1, x2 = left, left + width
        y1, y2 = top, top + height

        st.markdown(f"🟦 Ausgewählter Bereich: X={x1}-{x2}, Y={y1}-{y2}")
        preview_image = original_color.copy()
        cv2.rectangle(preview_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        st.image(preview_image, caption="Vorschau des gewählten Bereichs", channels="BGR")

        if st.button("Ausschnitt verwenden", key="use_crop_main"):
            st.session_state.cropping_done = True
            st.session_state.image_color = original_color[y1:y2, x1:x2]
            st.session_state.image_gray = original_gray[y1:y2, x1:x2]

        # Manuelle Anpassung
        st.markdown("**🔧 Optional: Zuschneidebereich anpassen**")
        x1 = st.number_input("Start X", min_value=0, max_value=original_gray.shape[1]-1, value=x1)
        x2 = st.number_input("Ende X", min_value=x1+1, max_value=original_gray.shape[1], value=x2)
        y1 = st.number_input("Start Y", min_value=0, max_value=original_gray.shape[0]-1, value=y1)
        y2 = st.number_input("Ende Y", min_value=y1+1, max_value=original_gray.shape[0], value=y2)

        if st.button("Ausschnitt verwenden", key="use_crop_manual"):
            st.session_state.cropping_done = True
            st.session_state.image_color = original_color[y1:y2, x1:x2]
            st.session_state.image_gray = original_gray[y1:y2, x1:x2]

        if st.button("Erneut erkennen", key="reset_crop"):
            if 'cropping_done' in st.session_state:
                del st.session_state['cropping_done']

        if st.button("Ausschnitt verwenden", key="use_crop_final"):
            st.session_state.cropping_done = True
            st.session_state.image_color = original_color[y1:y2, x1:x2]
            st.session_state.image_gray = original_gray[y1:y2, x1:x2]

if uploaded_file and not use_crop:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    st.session_state.image_color = image_color
    st.session_state.image_gray = image_gray
    st.session_state.cropping_done = True

if st.session_state.get("cropping_done"):
    image_color = st.session_state.image_color
    image_gray = st.session_state.image_gray

    UI, chi_curve, quadrant_overlay, heatmap_data, a0 = calculate_ui(image_gray, max_division=vis_division, vis_div=vis_division)

    st.markdown(f"### 📊 Uniformity Index: **{UI:.2f} %**")
st.markdown(f"Referenzfläche a₀ (Chi²-Grenze bei p=0.0275): **{a0:.4f}**")

    # Histogramm anzeigen
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(image_gray.ravel(), bins=25, color='black', edgecolor='white', weights=np.ones_like(image_gray.ravel()) / len(image_gray.ravel()) * 100)
    ax_hist.set_title("Histogramm der Grauwerte")
    ax_hist.set_xlabel("Gray Level")
    ax_hist.set_ylabel("Frequency %")
    ax_hist.set_xlim(0, 255)
    st.pyplot(fig_hist)

    # Heatmap der Blockmittelwerte
    if heatmap_data is not None:
        fig_heat, ax_heat = plt.subplots()
        cax = ax_heat.imshow(heatmap_data, cmap="gray")
        fig_heat.colorbar(cax, ax=ax_heat)
        ax_heat.set_title("Heatmap der Blockmittelwerte")
        st.pyplot(fig_heat)

    display_image(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    if quadrant_overlay is not None:
        st.image(quadrant_overlay, caption=f"Visualisierung der Quadranten ({vis_division}x{vis_division})", channels="RGB")
    plot_chi_curve(chi_curve)

    # Speichere Verlaufseintrag
    buffered = BytesIO()
    img_pil = Image.fromarray(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    buffered_hist = BytesIO()
    fig_hist.savefig(buffered_hist, format="PNG")
    hist_str = base64.b64encode(buffered_hist.getvalue()).decode()
    buffered_heat = BytesIO()
    fig_heat.savefig(buffered_heat, format="PNG")
    heat_str = base64.b64encode(buffered_heat.getvalue()).decode()

    st.session_state.history.append({
        "ui": UI,
        "img": img_str,
        "chi": chi_curve,
        "hist": hist_str,
        "heat": heat_str
    })

# Historie anzeigen
if st.session_state.history:
    st.subheader("📚 Analyse-Historie")
    for i, entry in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**{i+1}. Ergebnis:** Uniformity Index = **{entry['ui']:.2f} %**")
        st.image(Image.open(BytesIO(base64.b64decode(entry['img']))), width=200)
        with st.expander("Details anzeigen"):
            st.markdown("**Chi²-Kurve:**")
            st.line_chart(entry['chi'])
            st.markdown("**Histogramm:**")
            st.image(Image.open(BytesIO(base64.b64decode(entry['hist']))), use_column_width=True)
            st.markdown("**Heatmap der Blockmittelwerte:**")
            st.image(Image.open(BytesIO(base64.b64decode(entry['heat']))), use_column_width=True)



