# Synthetic Scalpel Dataset Generator

Generates synthetic surgical scalpel images using ComfyUI with ControlNet (canny edge) + SDXL img2img. Takes real scalpel photos as input and produces varied synthetic versions across different surgical scenarios and lighting conditions.

The full generated dataset is available on Kaggle: [synthetic-surgical-scalpel-w-yolo-box](https://www.kaggle.com/datasets/maliklainsbury/synthetic-surgical-scalpel-w-yolo-box)

Source images used are from the [Labeled Surgical Tools dataset](https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools) on Kaggle.

---

## Requirements

- Python 3.x with [Pillow](https://pypi.org/project/Pillow/) (`pip install Pillow`)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and running

### ComfyUI models

Download these and place them in the correct ComfyUI folders:

| Model | Folder | Source |
|-------|--------|--------|
| `RealVisXL_V5.0_fp16.safetensors` | `models/checkpoints/` | CivitAI / HuggingFace |
| `sdxl-vae-fp16-fix.safetensors` | `models/vae/` | HuggingFace (`madebyollin/sdxl-vae-fp16-fix`) |
| `diffusion_pytorch_model_promax.safetensors` | `models/controlnet/SDXL/controlnet-union-sdxl-1.0/` | HuggingFace (`xinsir/controlnet-union-sdxl-1.0`) |
| `4x-UltraSharp.pth` | `models/upscale_models/` | CivitAI |

---

## Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/Lainsm/synthetic-scalpel-dataset.git
   cd synthetic-scalpel-dataset
   ```

2. Install Python dependency:
   ```bash
   pip install Pillow
   ```

3. Add your source scalpel images to the `input/` folder (`.jpg`, `.jpeg`, or `.png`).

4. Start ComfyUI in a separate terminal:
   ```bash
   python main.py
   ```

---

## Usage

```bash
python script/generate_studio_scalpel.py
```

Generated images are saved to `output/`. A `generation_log.txt` is also written there tracking which source image and scenario produced each output.

### Configuration

Open `script/generate_studio_scalpel.py` and edit the top section:

```python
SERVER = "127.0.0.1:8188"   # change if ComfyUI is on a different address/port
TOTAL_IMAGES = 1000          # total number of images to generate
```

---

## Project structure

```
synthetic-scalpel-dataset/
├── input/                  # put your source scalpel images here
├── output/                 # generated images saved here
├── json/
│   └── comfyUI_instructions.json   # ComfyUI workflow
└── script/
    └── generate_studio_scalpel.py  # main generation script
```
