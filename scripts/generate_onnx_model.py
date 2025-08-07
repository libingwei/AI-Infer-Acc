# scripts/generate_onnx_model.py
#
# Purpose: Generates a ResNet18 ONNX model using PyTorch.
# This script downloads a pretrained ResNet18 model, sets it to evaluation mode,
# and exports it to the ONNX format, saving it in the 'models' directory.

import torch
import torchvision
import os

def main():
    # Define the output directory and filename
    output_dir = os.path.join(os.path.dirname(__file__), "../models")
    onnx_file_path = os.path.join(output_dir, "resnet18.onnx")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    # Check if a CUDA-enabled GPU is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # 1. Load a pretrained ResNet18 model and move it to the GPU
    print("Loading pretrained ResNet18 model...")
    model = torchvision.models.resnet18(pretrained=True).eval().to(device)

    # 2. Create a dummy input tensor on the same device
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 3. Export the model to ONNX format
    print(f"Exporting model to ONNX at: {onnx_file_path}")
    torch.onnx.export(model,
                      dummy_input,
                      onnx_file_path,
                      verbose=False,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11,
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}}) # Optional: for dynamic batch size

    print("Model export complete.")
    print(f"File saved at: {onnx_file_path}")

if __name__ == "__main__":
    main()
