import cv2
import numpy as np
from PIL import Image
import streamlit as st
import requests

# Filter Functions
def apply_pencil_sketch(img_bgr, ksize=21, sigma=0):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(255 - gray, (ksize, ksize), sigma)
    return cv2.divide(gray, 255 - blurred, scale=256.0)

def apply_canny_edges(img_bgr, threshold1=50, threshold2=150):
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), threshold1, threshold2)
    output = np.ones_like(img_bgr) * 255
    output[edges != 0] = [255, 0, 0]
    return output

def apply_emboss(img_bgr, offset=128):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.add(cv2.filter2D(img_bgr, -1, kernel), offset)

# Image Loading
def load_image(uploaded_file):
    try:
        img_pil = Image.open(uploaded_file)
        img_np = np.array(img_pil)

        # Convert to BGR
        if len(img_np.shape) == 2:  # Grayscale
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)
        else:  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_bgr, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None

# Main App
def run_app():
    st.title("🎨 Artistic Filter Lab")
    st.write("Upload an image and apply creative filters!")

    # Sidebar controls
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    if not uploaded_file:
        st.info("Upload an image to begin!")
        return

    img_bgr, img_rgb = load_image(uploaded_file)
    if img_bgr is None:
        return

    filter_type = st.sidebar.selectbox("Select Filter",
        ["Original", "Pencil Sketch", "Stylized Edges", "Emboss"])

    # Filter parameters
    params = {}
    if filter_type == "Pencil Sketch":
        params['ksize'] = st.sidebar.slider("Blur Kernel", 3, 41, 21, step=2)
        params['sigma'] = st.sidebar.slider("Blur Sigma", 0.0, 10.0, 0.0, step=0.1)
    elif filter_type == "Stylized Edges":
        params['t1'] = st.sidebar.slider("Threshold 1", 0, 300, 50)
        params['t2'] = st.sidebar.slider("Threshold 2", 0, 500, 150)
    elif filter_type == "Emboss":
        params['offset'] = st.sidebar.slider("Gray Offset", 0, 255, 128)

    # Process image
    try:
        if filter_type == "Pencil Sketch":
            processed = apply_pencil_sketch(img_bgr, params['ksize'], params['sigma'])
            caption = "Pencil Sketch"
        elif filter_type == "Stylized Edges":
            processed = apply_canny_edges(img_bgr, params['t1'], params['t2'])
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            caption = "Stylized Edges"
        elif filter_type == "Emboss":
            processed = cv2.cvtColor(apply_emboss(img_bgr, params['offset']), cv2.COLOR_BGR2RGB)
            caption = "Emboss Effect"
        else:
            processed = img_rgb
            caption = "Original Image"

        # Display
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(img_rgb, use_column_width=True)
        col2.header(caption)
        col2.image(processed, use_column_width=True)

    except Exception as e:
        st.error(f"Error applying filter: {e}")
        st.image(img_rgb, caption="Original (Filter Failed)")

# Test Function
def test_filters():
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Monasterio_Khor_Virap%2C_Armenia%2C_2016-10-01%2C_DD_25.jpg/640px-Monasterio_Khor_Virap%2C_Armenia%2C_2016-10-01%2C_DD_25.jpg"
    try:
        resp = requests.get(url, headers={'User-Agent': 'Test/1.0'})
        img_bgr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)

        for filter_func, name in [
            (apply_pencil_sketch, "Pencil Sketch"),
            (apply_canny_edges, "Edges"),
            (apply_emboss, "Emboss")
        ]:
            result = filter_func(img_bgr)
            print(f"{name} filter applied successfully")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    if 'streamlit' in st.__dict__:
        run_app()
    else:
        test_filters()
        
