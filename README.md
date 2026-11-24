# CAMO-ViT
### Vision-Transformer Framework for Camouflaged Object Segmentation in Low-Visibility Aerial Imagery

CAMO-ViT is a research-oriented deep-learning project focused on camouflaged object segmentation in aerial imagery.

The project has evolved from an initial RGB-only segmentation model into a multimodal RGB-Infrared (RGB+IR) framework using a dual-stream SegFormer-B0 architecture.

The newer version explores whether combining visible and infrared information through feature-level fusion can improve camouflaged object segmentation.

## Project Evolution

CAMO-ViT has been developed through two major iterations.

### Version 1 â€” RGB-only CAMO-ViT

The initial implementation used visible RGB imagery and a SegFormer-B0-based segmentation architecture.

It established the initial segmentation pipeline and provided a baseline for subsequent experiments.

**Reported test performance on VIAC:**

| Metric | Result |
|---|---:|
| Dice Score | 0.8980 |
| IoU | 0.8148 |
| Precision | 0.9307 |
| Recall | 0.8674 |
| MAE | 0.0301 |

**Identified limitation:**

Although the model performed well on the dataset's test images, its predictions were less reliable on arbitrary images outside the training dataset. This highlighted the need to investigate generalization, dataset diversity, and the limitations of relying on visible-spectrum information alone.

Original RGB-only project: (https://github.com/AkT-Dot-tech/CAMO-ViT)

### Version 2 â€” Multimodal RGB+IR CAMO-ViT

The newer implementation introduces a dual-stream architecture that processes visible RGB and infrared imagery separately before combining their extracted features.

**Major improvements:**

- Separate RGB and infrared feature extraction streams.
- SegFormer-B0 transformer-based encoders.
- Feature-level concatenation fusion.
- Binary segmentation output.
- Updated training and evaluation experiments.
- Interactive Streamlit inference application.

The objective is to leverage complementary information from visible and infrared modalities to improve segmentation performance.

## Architecture

The selected model uses a dual-stream SegFormer-B0 architecture with concatenation-based feature fusion.

Processing pipeline:

1. Receive a corresponding RGB and infrared image pair.
2. Preprocess both modalities and resize them to 384 Ã— 384.
3. Extract features independently using two SegFormer-B0 encoders.
4. Combine the extracted features through concatenation fusion.
5. Generate segmentation logits through the segmentation decoder.
6. Apply sigmoid activation and a threshold of 0.5 to produce the binary segmentation mask.

## Performance â€” Current Model

The selected Dual-Stream SegFormer-B0 with concatenation fusion achieved the following results on the VIAC test dataset:

| Metric | Result |
|---|---:|
| Dice Score | 0.9232 |
| IoU | 0.8645 |
| MAE | 0.0126 |
| Structure Measure (SÎ±) | 0.9252 |
| Mean Enhanced Alignment (EÏ†) | 0.9725 |
| Weighted F-measure | 0.9047 |
| Parameters | 7.23M |
| Inference speed | 56.43 FPS |

*Inference speed was measured on an NVIDIA Tesla T4 using FP32 and batch size 1. Actual performance depends on hardware and runtime conditions.*

### Comparison with the Earlier RGB-only Version

| Metric | RGB-only | RGB+IR |
|---|---:|---:|
| Dice | 0.8980 | **0.9232** |
| IoU | 0.8148 | **0.8645** |

The newer multimodal model shows improved benchmark performance on the reported VIAC test results.

These results demonstrate progress on the evaluated dataset; they do not independently establish improved generalization to unseen real-world environments.

## Dataset

The project uses the VIAC multimodal dataset.

It contains:
- Visible RGB images
- Infrared images
- Ground-truth segmentation masks

Current dataset split:

| Split | Samples |
|---|---:|
| Training | 1,020 |
| Validation | 180 |
| Testing | 300 |

The current public inference application requires a corresponding RGB and infrared image pair.

RGB-only inference is not currently supported by this deployment.

## Interactive Demo

The project includes a Streamlit application that allows users to:

- Upload RGB and infrared images.
- Generate binary segmentation predictions.
- View predicted segmentation masks.
- Visualize overlays on the original RGB image.
- Inspect probability maps.
- Download prediction outputs.

**Live Demo:** To be added after deployment.

## Model Checkpoint

The trained dual-stream model checkpoint is hosted separately from the source code.

**Hugging Face Model Repository:** https://huggingface.co/AKT033/camo-vit-dual-concat-b0

The application automatically downloads `best.pth` from the Hugging Face repository during its first inference run. The checkpoint is cached for subsequent runs.text
model/
â””â”€â”€ dual_concat_b0_ft_s1/
    â””â”€â”€ dual_concat_b0_ft_s1/
        â””â”€â”€ best.pth
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CAMO-ViT-Deployment
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal and provide a compatible RGB and infrared image pair.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- SegFormer
- NumPy
- Pillow
- Streamlit

## Current Limitations

- Requires paired RGB and infrared inputs.
- Input modalities should be spatially aligned.
- The model is designed for binary camouflaged-object segmentation.
- Input images are resized to 384 Ã— 384.
- Benchmark performance does not guarantee equivalent performance on arbitrary real-world imagery.
- Further independent testing is required to assess generalization.

## Future Work

- RGB-only inference support.
- Improved multimodal feature-fusion strategies.
- Natural-language-guided object discovery.
- Broader multimodal visual understanding.
- Evaluation on additional datasets and real-world imagery.

## Project Status

The dual-stream model has been trained and evaluated. The local inference application is prepared, and public deployment is in progress.

---

**CAMO-ViT â€” Exploring Multimodal Vision for Camouflaged Object Segmentation.**


