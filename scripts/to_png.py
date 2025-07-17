import numpy as np
from PIL import Image
import os

npz_path = "/tmp/openai-2025-07-17-12-58-02-234997/samples_10x64x64x3.npz"  # Replace with your .npz file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

data = np.load(npz_path)
images = data["arr_0"]  # shape: (N, H, W, C)

for i, img_array in enumerate(images):
    # Ensure uint8 type
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(os.path.join(output_dir, f"sample_{i:05d}.png"))

print(f"Saved {len(images)} images to {output_dir}")
