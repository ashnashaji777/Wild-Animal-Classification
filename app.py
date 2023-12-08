import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detect import ObjectDetection  # Import your ObjectDetection class

def main():
    st.title("Object Detection App")
    st.sidebar.title("Settings")

    image_file = st.sidebar.file_uploader("poocha", type=["jpg", "jpeg", "png"])
    if image_file:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Perform object detection when a file is uploaded
        detector = ObjectDetection()
        im_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = detector.predict(im_cv)
        annotated_image, _ = detector.plot_bboxes(results, im_cv)
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()
