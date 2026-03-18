import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from archs.ResNet import ResNet

app = FastAPI(title="FER API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
NUM_CLASSES = 8
WEIGHT_PATH = "weights/best_model.pth"

class_names = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise"]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

model = ResNet(num_classes=NUM_CLASSES, pretrained=False)
state = torch.load(WEIGHT_PATH, map_location=device)

if isinstance(state, dict) and "model_state_dict" in state:
    weights = state["model_state_dict"]
else:
    weights = state

weights = {k.replace("module.", ""): v for k, v in weights.items()}
model.load_state_dict(weights)
model.to(device)
model.eval()

def predict_pil_image(image: Image.Image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return {
        "class_index": pred_idx,
        "class_name": class_names[pred_idx],
        "confidence": round(confidence, 4)
    }


@app.get("/")
def root():
    return {"message": "FER Emotion Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = predict_pil_image(image)
    return result