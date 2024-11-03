import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sclera_segmentation import segmentation
from unet.unet import get_vessels_from_image
from svm.svm import calculate_bgr, adjust_contrast, calculate_lab, extract_white
import pickle

# Load the pre-trained SVM model
filename = 'svm/svm_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Streamlit app
st.title("Anemia Detection from Eye Image")
st.write("Upload an eye image to predict if the patient has anemia.")

# Image uploader in Streamlit
uploaded_file = st.file_uploader("Choose an eye image...", type="jpg")

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resizing image for faster processing (OPTIONAL)
    height, width, _ = image.shape
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))

    # Display original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Segment sclera
    mask_sclera, _, _, _ = segmentation.segment(image)
    img_segmented = cv2.bitwise_and(image, image, mask=mask_sclera)

    # Display segmented sclera image
    st.image(img_segmented, caption="Segmented Sclera", use_column_width=True)

    # Extract vessels from segmented sclera image
    pred_image = get_vessels_from_image(img_segmented)
    pred_image = pred_image.astype(np.uint8)

    # Mask sclera vessels
    sclera_vessels_masked = cv2.bitwise_and(img_segmented, img_segmented, mask=pred_image)

    # Display masked sclera vessels image
    st.image(sclera_vessels_masked, caption="Sclera Vessels Masked", use_column_width=True)

    # Calculate features
    sclera_quantiles_bgr, vessels_quantiles_bgr, vessels_density, value_r_minous_g_img_sclera, value_r_minous_g_img_vessels, dev_std_sclera, dev_std_vessels = calculate_bgr(img_segmented, sclera_vessels_masked)
    sclera_lab, sclera_vessels_lab = adjust_contrast(img_segmented, sclera_vessels_masked)
    value_a_img_sclera, value_a_img_vessels, dev_std_sclera_cielab, dev_std_vessels_cielab, sclera_quantiles_lab, vessels_quantiles_lab = calculate_lab(img_segmented, sclera_lab, sclera_vessels_lab)
    vessels_colors_white_deviations_cielab, vessels_colors_white_quantiles_cielab = extract_white(pred_image, img_segmented)

    # Prepare features for SVM model
    X = []
    one = vessels_density[0]
    tree = value_r_minous_g_img_vessels[0]
    four = dev_std_sclera[0][0][0] - dev_std_vessels[0][0][0]
    six = sclera_quantiles_bgr[0][2][1]
    eight = value_a_img_sclera[0] - value_a_img_vessels[0]
    nine = value_a_img_vessels[0]
    ten = dev_std_sclera_cielab[0][0][0]
    eleven = vessels_quantiles_lab[0][0][1]
    twelve = sclera_quantiles_lab[0][2][2] - vessels_quantiles_lab[0][2][2]

    # Add features to X list
    X.append([one, tree, four, six, eight, nine, ten, eleven, twelve])

    # Prediction
    y_pred = loaded_model.predict(X)

    # Display result
    if y_pred[0] == 1:
        st.write("### The patient has anemia")
    else:
        st.write("### The patient does not have anemia")
