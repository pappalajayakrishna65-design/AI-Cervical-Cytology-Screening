import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Set up the Web Page
st.set_page_config(page_title="AI Cervical Cytology", layout="wide")
st.title("🔬 AI-Powered Cervical Cytology Screening")
st.write("Upload a Pap smear image to automatically segment the cell and calculate the Nucleus-to-Cytoplasm (N/C) Ratio.")

# 2. Load the AI Model
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in the same folder as this script
    return YOLO('best.pt')

model = load_model()

# 3. Create the Image Uploader
uploaded_file = st.file_uploader("Upload a Cervical Cell Image (JPG/PNG/BMP)", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Uploaded Image", use_container_width=True)

    if st.button("Run AI Analysis"):
        with st.spinner('Analyzing cell structure...'):
            # 4. Run the YOLO Segmentation
            results = model.predict(image)
            result = results[0]

            with col2:
                # Plot the YOLO segmentation output (draws the polygons on the image)
                annotated_img = result.plot()
                st.image(annotated_img, caption="AI Segmentation Output", use_container_width=True)

            # 5. Calculate the N/C Ratio
            if result.masks is not None:
                nucleus_area = 0
                cytoplasm_area = 0

                # Look at every mask the AI drew
                for i, mask in enumerate(result.masks.data):
                    # Convert the mask to a math array
                    mask_np = mask.cpu().numpy()
                    
                    # Check which class it is (0 = Nucleus, 1 = Cytoplasm)
                    class_id = int(result.boxes.cls[i].item()) 
                    
                    # Count the pixels
                    pixel_area = np.sum(mask_np)

                    if class_id == 0:
                        nucleus_area += pixel_area
                    elif class_id == 1:
                        cytoplasm_area += pixel_area

                # 6. Display Clinical Results
                st.markdown("---")
                st.subheader("📊 Morphometric Analysis")
                
                if cytoplasm_area > 0 and nucleus_area > 0:
                    nc_ratio = nucleus_area / cytoplasm_area
                    
                    # Display the data clearly
                    st.write(f"**Nucleus Pixel Area:** {nucleus_area}")
                    st.write(f"**Cytoplasm Pixel Area:** {cytoplasm_area}")
                    st.success(f"### Calculated N/C Ratio: {nc_ratio:.3f}")

                    # Clinical Threshold Warning
                    if nc_ratio > 0.3: 
                        st.error("⚠️ **High N/C Ratio detected.** Morphological indicators suggest potential cellular atypia or dysplasia. Flag for cytologist review.")
                    else:
                        st.info("✅ **N/C Ratio within normal limits.**")
                else:
                    st.warning("Could not detect both a clear Nucleus and Cytoplasm to calculate the ratio.")
            else:
                st.error("No distinct cell structures detected in this image.")
