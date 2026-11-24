
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CAMO-ViT | Object Segmentation",
    page_icon="🎯",
    layout="wide"
)

SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
IMAGE_SIZE = 384
ASPECT_RATIO_TOLERANCE = 0.01


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_STATE = {
    "rgb_bytes": None,
    "ir_bytes": None,
    "prediction": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource(show_spinner=False)
def load_model():

    from inference import CAMOViTInference

    # inference.py downloads best.pth from Hugging Face.
    return CAMOViTInference(device="cpu")


# ==================================================
# IMAGE UTILITIES
# ==================================================

def decode_image(image_bytes, mode):

    if not image_bytes:
        raise ValueError("No image data received.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:

            image = ImageOps.exif_transpose(source)
            image.load()

            if image.width < 1 or image.height < 1:
                raise ValueError("Image has invalid dimensions.")

            return image.convert(mode)

    except Exception as error:
        raise ValueError(
            f"Could not decode image: {error}"
        ) from error


def get_aspect_ratio(image):

    return image.width / image.height


def validate_image_pair(rgb_image, ir_image):

    rgb_ratio = get_aspect_ratio(rgb_image)
    ir_ratio = get_aspect_ratio(ir_image)

    difference = abs(rgb_ratio - ir_ratio) / rgb_ratio

    return difference <= ASPECT_RATIO_TOLERANCE


def create_overlay(rgb_image, mask):

    rgb_image = rgb_image.convert("RGB")

    mask_image = Image.fromarray(mask).resize(
        rgb_image.size,
        Image.Resampling.NEAREST
    )

    red_layer = Image.new(
        "RGB",
        rgb_image.size,
        (255, 0, 0)
    )

    overlay_layer = Image.blend(
        rgb_image,
        red_layer,
        0.4
    )

    return Image.composite(
        overlay_layer,
        rgb_image,
        mask_image
    )


def image_to_bytes(image):

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def create_probability_image(probability):

    probability_uint8 = (
        np.clip(probability, 0, 1) * 255
    ).astype(np.uint8)

    return Image.fromarray(probability_uint8)


# ==================================================
# HEADER
# ==================================================

st.title("🎯 CAMO-ViT")

st.subheader(
    "Vision-Transformer Framework for "
    "Camouflaged Object Segmentation"
)

st.markdown(
    """
    Identify camouflaged objects in low-visibility aerial
    imagery using visible RGB and infrared information.

    Provide corresponding RGB and infrared images to generate
    a binary segmentation mask, overlay and probability map.
    """
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("Model Information")

    st.markdown(
        """
        **Architecture:** Dual-stream SegFormer-B0

        **Fusion:** Concatenation

        **Input Resolution:** 384 × 384

        **Output:** Binary segmentation

        **Threshold:** 0.50

        **Inference Device:** CPU
        """
    )

    st.divider()

    st.subheader("Model Performance")

    st.metric("Test Dice", "0.9232")
    st.metric("Test IoU", "0.8645")

    st.caption(
        "Previously reported evaluation results "
        "on the VIAC test set."
    )

    st.divider()

    st.caption("CAMO-ViT Research Demonstration")


# ==================================================
# INPUT SECTION
# ==================================================

st.header("1. Provide Input Images")

st.info(
    "Paste images using Ctrl + V inside the input areas, "
    "or use the attachment button. Both corresponding "
    "RGB and infrared images are required."
)

rgb_col, ir_col = st.columns(2)


# ==================================================
# RGB INPUT
# ==================================================

with rgb_col:

    st.subheader("🖼️ RGB / Visible Image")

    rgb_submission = st.chat_input(
        "Paste or attach RGB image",
        accept_file=True,
        file_type=SUPPORTED_FORMATS,
        key="rgb_chat_input"
    )

    if rgb_submission is not None:

        if len(rgb_submission.files) == 1:

            st.session_state.rgb_bytes = (
                rgb_submission.files[0].getvalue()
            )

            st.session_state.prediction = None

        elif len(rgb_submission.files) > 1:

            st.error("Please provide only one RGB image.")

    if st.session_state.rgb_bytes is not None:

        try:

            rgb_preview = decode_image(
                st.session_state.rgb_bytes,
                "RGB"
            )

            st.image(
                rgb_preview,
                caption="RGB input preview",
                use_container_width=True
            )

            st.caption(
                f"Resolution: {rgb_preview.width} × "
                f"{rgb_preview.height}"
            )

            if st.button(
                "Remove RGB image",
                key="remove_rgb"
            ):

                st.session_state.rgb_bytes = None
                st.session_state.prediction = None
                st.rerun()

        except Exception as error:

            st.error(f"Invalid RGB image: {error}")

            st.session_state.rgb_bytes = None
            st.session_state.prediction = None


# ==================================================
# INFRARED INPUT
# ==================================================

with ir_col:

    st.subheader("🌡️ Infrared Image")

    ir_submission = st.chat_input(
        "Paste or attach infrared image",
        accept_file=True,
        file_type=SUPPORTED_FORMATS,
        key="ir_chat_input"
    )

    if ir_submission is not None:

        if len(ir_submission.files) == 1:

            st.session_state.ir_bytes = (
                ir_submission.files[0].getvalue()
            )

            st.session_state.prediction = None

        elif len(ir_submission.files) > 1:

            st.error("Please provide only one infrared image.")

    if st.session_state.ir_bytes is not None:

        try:

            ir_preview = decode_image(
                st.session_state.ir_bytes,
                "L"
            )

            st.image(
                ir_preview,
                caption="Infrared input preview",
                use_container_width=True
            )

            st.caption(
                f"Resolution: {ir_preview.width} × "
                f"{ir_preview.height}"
            )

            if st.button(
                "Remove infrared image",
                key="remove_ir"
            ):

                st.session_state.ir_bytes = None
                st.session_state.prediction = None
                st.rerun()

        except Exception as error:

            st.error(f"Invalid infrared image: {error}")

            st.session_state.ir_bytes = None
            st.session_state.prediction = None


# ==================================================
# DECODE INPUTS
# ==================================================

rgb_image = None
ir_image = None

if st.session_state.rgb_bytes is not None:

    try:

        rgb_image = decode_image(
            st.session_state.rgb_bytes,
            "RGB"
        )

    except Exception as error:

        st.error(f"RGB image error: {error}")


if st.session_state.ir_bytes is not None:

    try:

        ir_image = decode_image(
            st.session_state.ir_bytes,
            "L"
        )

    except Exception as error:

        st.error(f"Infrared image error: {error}")


# ==================================================
# INPUT VALIDATION AND PREDICTION
# ==================================================

if rgb_image is not None and ir_image is not None:

    st.divider()

    if not validate_image_pair(rgb_image, ir_image):

        st.error(
            "The RGB and infrared images have incompatible "
            "aspect ratios. Please provide corresponding "
            "images from the same aligned scene."
        )

        st.caption(
            "Automatic resizing cannot correct camera "
            "misalignment or different fields of view."
        )

    else:

        st.success("Both input images are ready.")

        st.caption(
            "Different source resolutions are accepted when "
            "the images have compatible aspect ratios. "
            "Both inputs must represent the same aligned scene. "
            "Model preprocessing resizes both to 384 × 384."
        )

        if st.button(
            "🚀 Run CAMO-ViT Segmentation",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Loading model and generating prediction..."
                ):

                    model = load_model()

                    result = model.predict(
                        rgb_image,
                        ir_image
                    )

                    st.session_state.prediction = {
                        "mask": result["mask"],
                        "probability": result["probability"],
                        "inference_time": result["inference_time"]
                    }

            except Exception as error:

                st.session_state.prediction = None

                st.error(
                    "Prediction could not be completed."
                )

                with st.expander("Technical error details"):
                    st.exception(error)


elif rgb_image is not None:

    st.info(
        "RGB image received. Please provide its corresponding "
        "infrared image to use the dual-stream model."
    )


elif ir_image is not None:

    st.info(
        "Infrared image received. Please provide its "
        "corresponding RGB image."
    )


else:

    st.info(
        "Paste or attach your RGB and infrared images to begin."
    )


# ==================================================
# RESULTS
# ==================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    mask = prediction["mask"]
    probability = prediction["probability"]
    inference_time = prediction["inference_time"]

    overlay = create_overlay(
        rgb_image,
        mask
    )

    probability_image = create_probability_image(
        probability
    )

    st.divider()

    st.success("Segmentation completed successfully!")

    st.metric(
        "Inference Time",
        f"{inference_time:.3f} seconds"
    )

    st.header("2. Segmentation Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.image(
            Image.fromarray(mask),
            caption="Predicted Binary Mask (384 × 384)",
            use_container_width=True
        )

    with result_col2:

        st.image(
            overlay,
            caption="Segmentation Overlay",
            use_container_width=True
        )

    st.subheader("Prediction Probability Map")

    st.image(
        probability_image,
        caption="Pixel-wise Foreground Probability",
        use_container_width=True
    )

    st.divider()

    st.header("3. Download Results")

    download_col1, download_col2, download_col3 = (
        st.columns(3)
    )

    with download_col1:

        st.download_button(
            "Download Mask",
            data=image_to_bytes(
                Image.fromarray(mask)
            ),
            file_name="camo_vit_mask.png",
            mime="image/png",
            use_container_width=True
        )

    with download_col2:

        st.download_button(
            "Download Overlay",
            data=image_to_bytes(overlay),
            file_name="camo_vit_overlay.png",
            mime="image/png",
            use_container_width=True
        )

    with download_col3:

        st.download_button(
            "Download Probability Map",
            data=image_to_bytes(probability_image),
            file_name="camo_vit_probability.png",
            mime="image/png",
            use_container_width=True
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "CAMO-ViT | Dual-stream Vision Transformer "
    "for Camouflaged Object Segmentation"
)
