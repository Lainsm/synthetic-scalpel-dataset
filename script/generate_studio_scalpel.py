import io
import json
import random
import urllib.request
import urllib.parse
import time
import os
from PIL import Image

############# configuration #############
SERVER = "127.0.0.1:8188"
TOTAL_IMAGES = 10  # change this to generate more images (e.g. 1000)

############## Models (must match filenames in your ComfyUI models folders) #############
CHECKPOINT  = "RealVisXL_V5.0_fp16.safetensors"
VAE         = "sdxl-vae-fp16-fix.safetensors"
CONTROLNET  = "SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
UPSCALE     = "4x-UltraSharp.pth"

# ############# Paths (no changes needed) #############
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR     = os.path.join(BASE_DIR, "input")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
WORKFLOW_PATH = os.path.join(BASE_DIR, "json", "comfyUI_instructions.json")

# Some scenarios to add variety to the generated images. You can customise or expand this list as needed. These are the one I used to generate. 
# Not all generated images will match these perfectly since the prompt is just one part of the workflow 
# and the source image also has a big influence, but it should add some nice variety to your dataset.
SCENARIOS = [
    "isolated on white background, product photography, clean sterile instrument",
    "on green surgical drape, operating theatre, sterile field",
    "on blue sterile surgical drape, OR environment",
    "on stainless steel instrument tray, hospital, sterile",
    "on grey clinical surface, medical photography",
    "covered in blood, post-operative, surgical use",
    "with blood contamination, used surgical instrument",
    "on red surgical drape, operating theatre",
    "on white sterile cloth, medical setting",
    "on dark background, dramatic lighting, medical photography",
    "on kidney dish chrome bowl, sterile field",
    "held by gloved hand, surgery, operating theatre",
    "on blue surgical drape with other instruments visible",
    "on metal surface with reflections, clinical",
    "under harsh overhead operating theatre lights, strong shadows",
    "with uneven lighting, partial shadow",
    "with glare and reflections from OR lights",
    "poorly lit, dim lighting, shadows",
    "backlit, silhouette, strong backlight",
    "with motion blur, handheld camera"
]

POSITIVE_BASE = "a surgical scalpel, stainless steel blade and handle, surgical instrument, photorealistic, high quality, no text"
NEGATIVE_PROMPT = "multiple instruments, extra objects, patterns, distorted, hallucination, text, writing, letters, words, watermark, logo, brand name, engraving, ugly, blurry, cropped"

with open(WORKFLOW_PATH, "r") as f:
    workflow_template = json.load(f)

# Patch model names from config so you never need to edit the JSON but can if you want to test different models easily
# I found that these models gave me the best results for scalpel generation. 
# Feel free to experiment with different ones from your ComfyUI setup. 
# Just make sure to update the names here to match the filenames in your models folders.

workflow_template["2"]["inputs"]["ckpt_name"]          = CHECKPOINT
workflow_template["3"]["inputs"]["vae_name"]            = VAE
workflow_template["6"]["inputs"]["control_net_name"]   = CONTROLNET
workflow_template["20"]["inputs"]["model_name"]         = UPSCALE


def check_server():
    try:
        urllib.request.urlopen(f"http://{SERVER}/system_stats", timeout=3)
    except Exception:
        print(f"Cannot connect to ComfyUI at {SERVER}")
        print("Make sure ComfyUI is running before starting this script.")
        exit(1)

def upload_image(img, filename):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_data = buf.getvalue()
    boundary = b"----FormBoundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="image"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
    ) + image_data + b"\r\n--" + boundary + b"--\r\n"
    req = urllib.request.Request(
        f"http://{SERVER}/upload/image",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=----FormBoundary"}
    )
    urllib.request.urlopen(req)

def queue_prompt(prompt):
    data = json.dumps({"prompt": prompt}).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    response = urllib.request.urlopen(req)
    return json.loads(response.read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_image(filename, subfolder, folder_type):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
    with urllib.request.urlopen(f"http://{SERVER}/view?{params}") as response:
        return response.read()

def wait_for_completion(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history and history[prompt_id].get("outputs"):
            return history[prompt_id]
        time.sleep(2)
    return None


def generate_image(source_image, scenario, output_path):
    log_path = os.path.join(OUTPUT_DIR, "generation_log.txt")
    workflow = json.loads(json.dumps(workflow_template))

    img = Image.open(source_image)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        img = img.rotate(angle, expand=True)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    image_filename = f"aug_{angle}_{os.path.basename(source_image)}"
    upload_image(img, image_filename)

    workflow["1"]["inputs"]["image"]  = image_filename
    workflow["4"]["inputs"]["text"]   = f"{POSITIVE_BASE}, {scenario}"
    workflow["5"]["inputs"]["text"]   = NEGATIVE_PROMPT
    workflow["12"]["inputs"]["seed"]  = random.randint(1, 999999999999999)
    workflow["99"] = {
        "inputs": {"filename_prefix": "synthetic_scalpel", "images": ["13", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
    }

    result = queue_prompt(workflow)
    prompt_id = result["prompt_id"]
    print(f"  Queued: {prompt_id} | angle={angle}")

    history = wait_for_completion(prompt_id)

    if history and "99" in history.get("outputs", {}):
        for image_data in history["outputs"]["99"]["images"]:
            img_data = get_image(image_data["filename"], image_data["subfolder"], image_data["type"])
            with open(output_path, "wb") as f:
                f.write(img_data)
            with open(log_path, "a") as log:
                log.write(f"{output_path} | {os.path.basename(source_image)} | angle={angle} | {scenario}\n")
            return True
    else:
        print(f"  ERROR: outputs={list(history.get('outputs', {}).keys()) if history else 'timeout'}")

    return False


def main():
    check_server()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_images = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not valid_images:
        print(f"No images found in {IMAGE_DIR}")
        return
    for img in valid_images:
        print(f"Found: {os.path.basename(img)}")

    print(f"\nGenerating {TOTAL_IMAGES} synthetic scalpel images...")
    print(f"Output: {OUTPUT_DIR}\n")

    generated = 0
    failures = 0

    while generated < TOTAL_IMAGES:
        source_image = valid_images[generated % len(valid_images)]
        scenario = SCENARIOS[generated % len(SCENARIOS)]
        if random.random() > 0.5:
            scenario = random.choice(SCENARIOS)

        output_path = os.path.join(OUTPUT_DIR, f"scalpel_{generated:04d}.png")
        print(f"[{generated+1}/{TOTAL_IMAGES}] {os.path.basename(source_image)} | {scenario[:45]}...")

        if generate_image(source_image, scenario, output_path):
            generated += 1
            print(f"Saved: scalpel_{generated:04d}.png")
            failures = 0
        else:
            failures += 1
            print(f"Failed (attempt {failures})")
            time.sleep(5)
            if failures > 5:
                print("Too many consecutive failures — is ComfyUI running?")
                break

    print(f"\nDone! Generated {generated}/{TOTAL_IMAGES} scalpel images")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
