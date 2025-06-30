from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import os
import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Stable Diffusion Inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cpu")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extend-image', methods=['POST'])
def extend_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    width = int(request.form.get('width'))
    height = int(request.form.get('height'))

    original = Image.open(file.stream).convert("RGB")
    orig_w, orig_h = original.size

    # Center original image on new canvas
    new_img = Image.new("RGB", (width, height), (255, 255, 255))
    x_offset = (width - orig_w) // 2
    y_offset = (height - orig_h) // 2
    new_img.paste(original, (x_offset, y_offset))

    # Create mask: white = to generate
    mask = Image.new("L", (width, height), 255)
    mask.paste(0, (x_offset, y_offset, x_offset + orig_w, y_offset + orig_h))

    # Outpaint using prompt (optional: make dynamic)
    result = pipe(
        prompt="a realistic background matching the image",
        image=new_img,
        mask_image=mask
    ).images[0]

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
    result.save(output_path)

    return render_template('index.html', result_image=output_path)

@app.route('/download')
def download():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'output.png'), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
