
import streamlit as st
import torch
import numpy as np
import io

from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

st.set_page_config(
    page_title="Road Scene Segmentation",
    page_icon="🚗",
    layout="wide"
)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-768-768"

st.title("🚗 Road Scene Segmentation")
st.write(
    "AI-powered road scene segmentation using a pretrained "
    "SegFormer deep learning model."
)

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model

with st.spinner("Loading SegFormer model..."):
    try:
        processor, model = load_model()
        st.success("SegFormer Model Loaded Successfully!")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

device = torch.device("cpu")
model.to(device)

palette = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32)
]

st.divider()

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Road Image")

    st.image(
        image,
        use_container_width=True
    )

    st.caption(
        f"Image dimensions: {image.width} × {image.height}"
    )

    if st.button("Generate Segmentation", type="primary"):

        with st.spinner("Analyzing road scene..."):

            try:
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.inference_mode():
                    outputs = model(**inputs)

                logits = outputs.logits

                logits = F.interpolate(
                    logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False
                )

                predicted_mask = logits.argmax(
                    dim=1
                )[0].cpu().numpy()

                num_classes = model.config.num_labels

                color_mask = np.zeros(
                    (image.height, image.width, 3),
                    dtype=np.uint8
                )

                for class_id in range(num_classes):
                    color_mask[
                        predicted_mask == class_id
                    ] = palette[class_id % len(palette)]

                segmented_image = Image.fromarray(
                    color_mask
                )

                overlay = Image.blend(
                    image,
                    segmented_image,
                    alpha=0.5
                )

                st.success("Segmentation Completed!")

                st.divider()

                st.subheader("Segmentation Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        image,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        segmented_image,
                        caption="Predicted Segmentation",
                        use_container_width=True
                    )

                st.divider()

                st.subheader("Segmentation Overlay")

                st.image(
                    overlay,
                    use_container_width=True
                )

                st.divider()

                st.subheader("Detected Classes")

                unique_classes, counts = np.unique(
                    predicted_mask,
                    return_counts=True
                )

                id2label = model.config.id2label

                total_pixels = predicted_mask.size

                for class_id, count in zip(
                    unique_classes,
                    counts
                ):

                    label = id2label.get(
                        int(class_id),
                        f"Class {class_id}"
                    )

                    percentage = (
                        count / total_pixels
                    ) * 100

                    st.write(
                        f"**{label}**: {percentage:.2f}%"
                    )

                    st.progress(
                        float(percentage / 100)
                    )

                st.divider()

                st.subheader("Download Results")

                buffer = io.BytesIO()

                segmented_image.save(
                    buffer,
                    format="PNG"
                )

                st.download_button(
                    "Download Segmented Image",
                    data=buffer.getvalue(),
                    file_name="segmented_road.png",
                    mime="image/png"
                )

                overlay_buffer = io.BytesIO()

                overlay.save(
                    overlay_buffer,
                    format="PNG"
                )

                st.download_button(
                    "Download Overlay",
                    data=overlay_buffer.getvalue(),
                    file_name="road_overlay.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Segmentation Error: {e}")
                st.exception(e)
