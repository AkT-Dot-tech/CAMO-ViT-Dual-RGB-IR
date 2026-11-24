
import time
import os

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from huggingface_hub import hf_hub_download

from camo_rgbt.common import CamoNet


# Model configuration
IMAGE_SIZE = 384
THRESHOLD = 0.5

HF_REPO_ID = "AKT033/camo-vit-dual-concat-b0"
HF_FILENAME = "best.pth"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CAMOViTInference:

    def __init__(self, checkpoint_path=None, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Download from Hugging Face if local checkpoint is unavailable.
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):

            try:
                checkpoint_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_FILENAME
                )

            except Exception as error:
                raise RuntimeError(
                    f"Unable to download CAMO-ViT checkpoint "
                    f"from Hugging Face: {error}"
                ) from error

        # Construct the exact trained architecture.
        self.model = CamoNet(
            mode="dual",
            fusion="concat",
            backbone="b0"
        )

        # Load trained parameters.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False
            )

            if "model" not in checkpoint:
                raise KeyError(
                    "The checkpoint does not contain a 'model' state dictionary."
                )

            self.model.load_state_dict(
                checkpoint["model"],
                strict=True
            )

        except Exception as error:
            raise RuntimeError(
                f"Unable to load CAMO-ViT checkpoint: {error}"
            ) from error

        self.model.to(self.device)
        self.model.eval()

        # Preprocessing must remain consistent with training.
        self.transform_rgb = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        self.transform_ir = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(MEAN, STD)
        ])

    def predict(self, rgb_image, ir_image):

        if not isinstance(rgb_image, Image.Image):
            raise TypeError("RGB input must be a PIL image.")

        if not isinstance(ir_image, Image.Image):
            raise TypeError("Infrared input must be a PIL image.")

        rgb_image = rgb_image.convert("RGB")
        ir_image = ir_image.convert("L")

        if rgb_image.width == 0 or rgb_image.height == 0:
            raise ValueError("RGB image is empty.")

        if ir_image.width == 0 or ir_image.height == 0:
            raise ValueError("Infrared image is empty.")

        rgb_tensor = (
            self.transform_rgb(rgb_image)
            .unsqueeze(0)
            .to(self.device)
        )

        ir_tensor = (
            self.transform_ir(ir_image)
            .unsqueeze(0)
            .to(self.device)
        )

        start_time = time.perf_counter()

        with torch.inference_mode():

            logits = self.model(rgb_tensor, ir_tensor)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if logits.ndim != 4:
                raise RuntimeError(
                    f"Unexpected model output shape: {tuple(logits.shape)}"
                )

            probability = (
                torch.sigmoid(logits)[0, 0]
                .float()
                .cpu()
                .numpy()
            )

        inference_time = time.perf_counter() - start_time

        probability = np.nan_to_num(
            probability,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        probability = np.clip(probability, 0.0, 1.0)

        mask = (
            probability > THRESHOLD
        ).astype(np.uint8) * 255

        return {
            "mask": mask,
            "probability": probability,
            "inference_time": inference_time
        }
