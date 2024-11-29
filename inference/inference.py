import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import json

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_input(request_body, content_type="application/x-image"):
    # print("*"*60)
    # print(f"Pre-processing request_body {request_body}")
    # print("*"*60)

    try:
        # Decode base64 if the request body is a string
        if isinstance(request_body, str):
            request_body = base64.b64decode(request_body)
            # print(f"Pre-processing converted request_body {request_body}")
        
        if content_type in ["application/x-image", "text/csv"]:
            # Load the image from bytes
            image = Image.open(io.BytesIO(request_body)).convert("RGB")
            
            # Apply the transformations
            tensor = transform(image).unsqueeze(0)  # Add batch dimension
            return tensor
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        raise Exception(f"Error in preprocess_input {e}")
        

def input_fn(request_body, content_type):
    print("*"*60)
    print(f"-> Input content type {content_type}") 
    print(f"Payload size: {sys.getsizeof(request_body)} bytes")
    print("*"*60)
    
    # Preprocess the input using the logic defined
    return preprocess_input(request_body, content_type)

def predict_fn(input_data, model):
    print("*"*60)
    print(f"Predicting {input_data}")
    print("*"*60)
    
    # Perform inference using the preprocessed data
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

def output_fn(prediction, accept="application/json"):
    """
    Convert model prediction to a JSON format directly supported by SageMaker Clarify.

    Args:
        prediction (torch.Tensor): Model's raw output logits.
        accept (str): Desired content type for the response.

    Returns:
        str: JSON-encoded prediction with a flat structure.
    """
    try:
        # Convert logits to probabilities
        print(f"Predictions length {len(prediction)}")
        print(f"Example of prediction outputs: {prediction}")

        predicted_class = prediction.argmax(dim=1).tolist()

        print(f"Example of predicted class: {predicted_class}")

        return json.dumps(predicted_class)  # Array of arrays of numbers
    except Exception as e:
        raise ValueError(f"Error in output_fn: {str(e)}") from e

