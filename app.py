from flask import Flask, request, render_template, send_file
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import io

app = Flask(__name__)

# Load the SAM model
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
state_dict = torch.load('sam_model.pth')
# Filter out the unexpected keys
state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(state_dict, strict=False)
model.eval()

def remove_background(image):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)
    mask = output_predictions.byte().cpu().numpy()
    result = Image.fromarray(mask * 255).convert("L")
    return result

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        img = Image.open(file.stream).convert("RGB")
        mask = remove_background(img)
        img.putalpha(mask)
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)