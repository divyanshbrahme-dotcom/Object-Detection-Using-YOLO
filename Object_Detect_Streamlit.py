import streamlit as st
import numpy as np
import pandas as pd
from  ultralytics import YOLO
from PIL import Image

st.title("Object Detection using YOLO")

model = YOLO("yolo26n.pt")
image = Image.open("ele.png").convert("RGB")

Image_array = np.array(image)

st.subheader("Original Image")
st.image(Image_array)

if st.button("Detect Object"):
    results = model.predict(Image_array)
    result = results[0]

    Detected_image = result.plot()
    st.subheader("Detected Image")
    st.image(Detected_image, channels="BGR")

    final = []

    for box in result.boxes:
            class_id = int(box.cls[0])
            object_name = result.names[class_id]
            conf = float(box.conf[0]) * 100

            coordinates = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = coordinates


            final.append([object_name,
                             conf,
                             x1,
                             y1,
                             x2,
                             y2])
            
            if len(final) > 0:
                   df = pd.DataFrame(
                final,
                columns=[
                    "Object Name",
                    "Confidence (%)",
                    "X1", "Y1", "X2", "Y2"
                ]
            )

            st.subheader("Detection Results")
            st.dataframe(df, use_container_width=True)
            st.write("Total objects detected:", len(final))

    else:
            st.warning("No objects detected.")