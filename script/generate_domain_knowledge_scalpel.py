import json
import random
import urllib.request
import urllib.parse
import time
import os
import shutil
from PIL import Image

############### Configuration ###############
SERVER = "127.0.0.1:8188"
COMFYUI_INPUT_DIR = "/home/start/projects/ComfyUI/input"
IMAGE_DIR = "/home/start/assignments/HDS05/data/realScalpelDataset/train/images/"
BOWL_IMAGE = "/home/start/assignments/HDS05/data/metal_tray.png"
OUTPUT_DIR = "/home/start/projects/synthetic_dataset/scalpel_bowl/"
TOTAL_IMAGES = 150

WORKFLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_version_with_bowl_real.json")

BOWL_FILENAME = "metal_tray.png"

SELECTED_IMAGES = [
    "bisturi345.jpg", "bisturi346.jpg"
]

SCENARIOS = [
    "a surgical scalpel handle, in a stainless steel tray, high reflection of light on chrome, greyscale",
    "surgical scalpel in chrome instrument tray, OR lighting, grayscale, overhead",
]

POSITIVE_BASE = "a surgical scalpel handle, in a stainless steel tray, high reflection of light on chrome, greyscale"
NEGATIVE_PROMPT = "multiple instruments, extra objects, patterns, distorted, hallucination, text, writing, letters, words, watermark, logo, brand name, engraving, ugly, blurry, cropped, colour, colorful"

with open(WORKFLOW_PATH, "r") as f:
    workflow_template = json.load(f)


def queue_prompt(prompt):
    data = json.dumps({"prompt": prompt}).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=data)
    response = urllib.request.urlopen(req)
    return json.loads(response.read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_image(filename, subfolder, folder_type):
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    with urllib.request.urlopen(f"http://{SERVER}/view?{params}") as response:
        return response.read()

def wait_for_completion(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            if outputs:
                return history[prompt_id]
        time.sleep(2)
    return None

def generate_image(source_image, scenario, output_path):
    workflow = json.loads(json.dumps(workflow_template))

    shutil.copy2(BOWL_IMAGE, os.path.join(COMFYUI_INPUT_DIR, BOWL_FILENAME))

    img = Image.open(source_image)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        img = img.rotate(angle, expand=True)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    image_filename = f"aug_{angle}_{os.path.basename(source_image)}"
    img.save(os.path.join(COMFYUI_INPUT_DIR, image_filename))

    workflow["1"]["inputs"]["image"] = image_filename
    workflow["30"]["inputs"]["image"] = BOWL_FILENAME
    workflow["4"]["inputs"]["text"] = f"{POSITIVE_BASE}, {scenario}"
    workflow["5"]["inputs"]["text"] = NEGATIVE_PROMPT
    workflow["12"]["inputs"]["seed"] = random.randint(1, 999999999999999)
    workflow["14"]["inputs"]["batch_size"] = 1

    workflow["29"]["inputs"]["filename_prefix"] = "bowl_scalpel"

    result = queue_prompt(workflow)
    prompt_id = result["prompt_id"]
    print(f"  Queued: {prompt_id} | angle={angle}")

    history = wait_for_completion(prompt_id)

    if history and "outputs" in history:
        if "29" in history["outputs"]:
            for image_data in history["outputs"]["29"]["images"]:
                img_data = get_image(
                    image_data["filename"],
                    image_data["subfolder"],
                    image_data["type"]
                )
                with open(output_path, "wb") as f:
                    f.write(img_data)

                log_path = os.path.join(OUTPUT_DIR, "generation_log.txt")
                with open(log_path, "a") as log:
                    log.write(f"{output_path} | {os.path.basename(source_image)} | angle={angle} | {scenario}\n")

                return True
        else:
            print(f"  ERROR: Node 29 not in outputs: {list(history.get('outputs', {}).keys())}")

    return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_images = []
    for img in SELECTED_IMAGES:
        full_path = os.path.join(IMAGE_DIR, img)
        if os.path.exists(full_path):
            valid_images.append(full_path)
            print(f"✅ Found: {img}")
        else:
            print(f"❌ Missing: {img}")

    if not os.path.exists(BOWL_IMAGE):
        print(f"❌ Bowl image missing: {BOWL_IMAGE}")
        return
    else:
        print(f"✅ Bowl image found: {BOWL_IMAGE}")

    print(f"\nGenerating {TOTAL_IMAGES} kidney bowl synthetic images...")
    print(f"Output: {OUTPUT_DIR}\n")

    generated = 0
    failures = 0

    while generated < TOTAL_IMAGES:
        source_image = valid_images[generated % len(valid_images)]
        scenario = random.choice(SCENARIOS)
        output_path = f"{OUTPUT_DIR}bowl_scalpel_{generated:04d}.png"

        print(f"[{generated+1}/{TOTAL_IMAGES}] {os.path.basename(source_image)} | {scenario[:50]}...")

        success = generate_image(source_image, scenario, output_path)

        if success:
            generated += 1
            print(f"✅ Saved: bowl_scalpel_{generated:04d}.png")
            failures = 0
        else:
            failures += 1
            print(f"❌ Failed (attempt {failures})")
            time.sleep(5)
            if failures > 5:
                print("Too many failures — is ComfyUI running?")
                break

    print(f"\n🎉 Done! Generated {generated}/{TOTAL_IMAGES} kidney bowl images")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
