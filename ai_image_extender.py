from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
import torch

# === CONFIG ===
INPUT_IMAGE = "123.jpg"         # Your input image path
OUTPUT_IMAGE = "extended_image.png"  # Output file
PROMPT = "a realistic forest background"  # Text prompt to guide AI generation

TARGET_WIDTH = 900 # Desired width
TARGET_HEIGHT = 900  # Desired height


# === Load the inpainting pipeline ===
print("Loading AI model...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32
).to("cpu")
print("Model loaded.")


# === Step 1: Load and center the image on a bigger canvas ===
original = Image.open(INPUT_IMAGE).convert("RGBA")
ow, oh = original.size

print(f"Original size: {ow}x{oh}")
print(f"Expanding to: {TARGET_WIDTH}x{TARGET_HEIGHT}")

# Center the original image on a new white canvas
canvas = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (255, 255, 255, 255))
paste_x = (TARGET_WIDTH - ow) // 2
paste_y = (TARGET_HEIGHT - oh) // 2
canvas.paste(original, (paste_x, paste_y), mask=original)


# === Step 2: Create the mask ===
# The mask shows the **area we want AI to fill (inpaint)** — white = fill, black = keep
mask = Image.new("L", (TARGET_WIDTH, TARGET_HEIGHT), color=255)  # Entire area = white
mask.paste(0, (paste_x, paste_y, paste_x + ow, paste_y + oh))     # Original image = black

# === Step 3: Inpaint using AI ===
print("Running inpainting...")
result = pipe(
    prompt=PROMPT,
    image=canvas.convert("RGB"),
    mask_image=mask
).images[0]

# === Step 4: Save the output ===
result.save(OUTPUT_IMAGE)
print(f"Saved extended image to: {OUTPUT_IMAGE}")
