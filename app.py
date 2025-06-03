from io import BytesIO
import base64
from typing import Iterable

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# General Information
__author__ = "Bendix Brüggenjürgen"
__copyright__ = "Copyright 2025, Institut für Textiltechnik der RWTH Aachen University"
__credits__ = ["Bendix Brüggenjürgen"]
__version__ = "0.2"
__maintainer__ = "Bendix Brüggenjürgen"
__email__ = "bendix.brueggenjuergen@rwth-aachen.de"
__status__ = "In Development"

# Seiteneinrichtung
st.set_page_config(page_title="Uniformity Index Analyzer", layout="centered", initial_sidebar_state="expanded")
st.title("📷 Uniformity Index Analyzer")

if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.header("Einstellungen")
vis_division = st.sidebar.slider("Visualisierte Teilung (z. B. 4 = 4x4)", min_value=2, max_value=70, value=4, step=1)


############# math functions ##################################################
def _calculate_chi_values(sums_blocks: list[float]) -> float:
    """
    sums_blocks: 1D array of block‑sums.  Each entry must already be an integer “count”
                 (e.g. a Poisson‑distributed photon‑count).
    Returns the standard Poisson dispersion index:
       D = sum((x_i - x̄)^2) / x̄   which equals (n-1) * (sample_var / sample_mean),
       and under H0≈Poisson it ≃ χ²_{n-1}.
    """
    arr = np.array(sums_blocks, dtype=float)
    n = arr.size
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 0.0
    # use ddof=1 to get the unbiased sample variance s²:
    var_val = np.var(arr, ddof=1)   # s² = (1/(n-1)) Σ (x_i - x̄)²
    # I = s² / x̄  →  χ² = (n-1)*I = Σ(x_i - x̄)² / x̄
    I   = var_val / mean_val
    chi = I * (n - 1)
    return chi


def _block_sums(image: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Crop the grayscale image (output of cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY))
    so that its height/width is an integer multiple of n_blocks,
    then reshape into (n_blocks, block_h, n_blocks, block_w), and SUM over each block.
    The *only* reason to SUM is if 'image' is already a true Poisson‑count array
    (per‑pixel counts).  If 'image' is 8‑bit gray levels, this sum will be O(10^4…10^5)
    per block and break the Var=Mean assumption.
    """
    h, w = image.shape[:2]
    block_h = h // n_blocks
    block_w = w // n_blocks

    if block_h == 0 or block_w == 0:
        raise ValueError(f"Cannot split {h}×{w} image into {n_blocks}×{n_blocks} blocks.")

    cropped = image[: block_h * n_blocks, : block_w * n_blocks]

    if cropped.ndim == 2:
        # reshape into (n_blocks, block_h, n_blocks, block_w)
        blocks = cropped.reshape(n_blocks, block_h, n_blocks, block_w)
        # sum over rows_in_block (axis=1) and cols_in_block (axis=3)
        # → shape (n_blocks, n_blocks) of block‑sums
        return blocks.mean(axis=(1, 3))
    else:
        raise ValueError("Image must be a 2D grayscale array from cv2.cvtColor.")


def _calculate_uniformity_index_range(
    chi_values: list[float],
    max_division: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    chi_values: list of observed χ² statistics for subdivisions i=2..max_division.
                chi_values[k] corresponds to i = k+2  (i.e. 2×2, 3×3, …, max_division×max_division).
    max_division: largest i for which you computed a χ².

    Returns:
      UI                 : Uniformity Index (float, percent)
      dof                : array([i^2 - 1  for i=2..max_division])
      chi_crit_vals_a0   : χ²_{0.0275, dof}       for each df
      chi_crit_vals_025  : χ²_{0.025,  dof}       for each df
      chi_crit_vals_975  : χ²_{0.975,  dof}       for each df
      A                  : ∫ [observed χ²(i)  vs n=i²]  d(n)
      a0                 : ∫ [χ²_{0.0275, df(i)} vs n=i²] d(n)
    """
    # 1) i ranges from 2..max_division
    i_vals = np.arange(2, max_division + 1)

    # 2) total block‑count n_i = i^2
    n_vals = i_vals**2

    # 3) degrees of freedom at level i is df_i = n_i - 1 = i^2 - 1
    dof = n_vals - 1

    # 4) convert chi_values to array and check its length
    D = np.array(chi_values, dtype=float)
    if D.shape[0] != i_vals.shape[0]:
        raise ValueError(
            f"chi_values must have length {max_division - 1} (for i=2..{max_division}), "
            f"but got length {len(chi_values)}."
        )

    # 5) compute the lower‑tail 2.75% cutoff, plus 2.5%/97.5% if desired
    chi_crit_vals_a0  = chi2.ppf(0.0275, dof)
    chi_crit_vals_025 = chi2.ppf(0.025,  dof)
    chi_crit_vals_975 = chi2.ppf(0.975,  dof)

    # 6) now integrate *vs n = i²* using np.trapezoid
    #    A  = ∫ D(i)         d(n_i)
    #    a0 = ∫ χ²_{0.0275,dof} d(n_i)
    A  = np.trapezoid(D,            x=n_vals)
    a0 = np.trapezoid(chi_crit_vals_a0, x=n_vals)

    # 7) Uniformity Index = 100 * [1 / (1 + A/a0)]
    UI = 100.0 * (1.0 / (1.0 + (A / a0))) if a0 != 0 else 0.0

    return UI, dof, chi_crit_vals_a0, chi_crit_vals_025, chi_crit_vals_975, A, a0


def compute_chi_series(image: np.ndarray, max_division: int) -> list[float]:
    """
    Build the list [χ² for 2×2, 3×3, …, max_division×max_division].
    Note: image must be the 2D numpy array returned by cv2.cvtColor(..., cv2.COLOR_BGR2GRAY).
    If you pass 8‑bit gray levels, you will see huge χ² because Var≠Mean.
    """
    chi_values = [
        _calculate_chi_values(_block_sums(image, n).ravel())
        for n in range(2, max_division + 1)
    ]
    return chi_values

############# UI functions ##################################################
def calculate_overlay(image_overlay: Image, vis_div: int):
    h, w = image_overlay.shape[:2]
    block_height, block_width = h // vis_div, w // vis_div

    quadrant_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_GRAY2RGB)
    heatmap_data = _block_sums(image_overlay, vis_div)

    # --- draw the grid by drawing the horizontal and vertical lines once each ---
    # draw horizontal grid lines
    for i in range(1, vis_div):
        y = i * block_height
        cv2.line(quadrant_overlay, (0, y), (w, y), (0, 255, 0), 1)
    # draw vertical grid lines
    for i in range(1, vis_div):
        x = i * block_width
        cv2.line(quadrant_overlay, (x, 0), (x, h), (0, 255, 0), 1)
    return heatmap_data, quadrant_overlay


def calculate_ui(image: Image, max_division: int = 6):

    chi_values = compute_chi_series(image, max_division=max_division)

    # Visualisierung basierend auf gewählter Teilung
    heatmap_data, quadrant_overlay = calculate_overlay(image.copy(), max_division)

    UI, dof, chi_crit_vals_a0, chi_crit_vals_025, chi_crit_vals_975, A, a0 = _calculate_uniformity_index_range(
        chi_values, max_division
    )


    # Plot Chi-Square Curve with Reference Bounds
    fig, ax = plt.subplots()
    ax.plot(dof, chi_values, label='Sample χ²', marker='o')
    ax.plot(dof, chi_crit_vals_025, linestyle='--', color='red', label='χ²₀.₀₂₅')
    ax.plot(dof, chi_crit_vals_975, linestyle='--', color='blue', label='χ²₀.₉₇₅')
    ax.fill_between(dof, chi_crit_vals_025, chi_crit_vals_975, color='gray', alpha=0.3, label='Random Region')
    ax.set_title("Chi-square vs. Degrees of Freedom")
    ax.set_xlabel("Degrees of Freedom (n-1)")
    ax.set_ylabel("Chi-square Value")
    ax.legend()
    st.pyplot(fig)

    # Zeige Berechnungsformeln und Werte an
    st.markdown("### 📐 Formelübersicht")
    st.latex(r"I = \frac{\text{Var}(X)}{\text{E}(X)}")
    st.latex(r"\chi^2 = I \cdot (n - 1)")
    st.latex(r"A = \int_0^{n} \chi^2\,dn")
    st.latex(r"\text{UI} = \frac{1}{1 + \frac{A}{a_0}} \cdot 100")
    st.markdown(f"**A (Fläche unter Sample-χ²-Kurve):** {A:.2f}")
    st.markdown(f"**a₀ (Referenzfläche, χ²₀.₀₂₇₅):** {a0:.2f}")
    st.markdown(f"**Uniformity Index (UI):** {UI:.2f} %")

    return UI, chi_values, quadrant_overlay, heatmap_data, a0


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

        st.markdown(f"🕦 Ausgewählter Bereich: X={x1}-{x2}, Y={y1}-{y2}")
        preview_image = original_color.copy()
        cv2.rectangle(preview_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        st.image(preview_image[y1:y2, x1:x2], caption="Vorschau des gewählten Bereichs", channels="BGR")

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

    UI, chi_curve, quadrant_overlay, heatmap_data, a0 = calculate_ui(np.array(image_gray, dtype=np.uint8),
                                                                     max_division=vis_division)

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
            st.image(Image.open(BytesIO(base64.b64decode(entry['hist']))), use_container_width=True)
            st.markdown("**Heatmap der Blockmittelwerte:**")
            st.image(Image.open(BytesIO(base64.b64decode(entry['heat']))), use_container_width=True)
