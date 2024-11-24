import io
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define your preprocessing logic
def preprocess_input(request_body, content_type="application/x-image"):
    if content_type == "application/x-image":
        # Load image
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def input_fn(request_body, content_type):
    # Preprocess the input using the logic defined
    return preprocess_input(request_body, content_type)

def predict_fn(input_data, model):
    # Perform inference using the preprocessed data
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

def output_fn(prediction, accept="application/json"):
    # Convert the output tensor to JSON-serializable format
    return {"predictions": prediction.argmax(1).tolist()}
