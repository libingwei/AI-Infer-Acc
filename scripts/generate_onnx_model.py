# scripts/generate_onnx_model.py
#
# Purpose: Generates a ResNet18 ONNX model using PyTorch.
# This script checks for a local copy of the pretrained weights first.
# If not found, it downloads them and saves a local copy for future use.
# Finally, it exports the model to the ONNX format.

import torch
import torchvision
import os

def main():
    # Define paths for the output directory, the PyTorch model file, and the ONNX model file.
    output_dir = os.path.join(os.path.dirname(__file__), "../models")
    pth_file_path = os.path.join(output_dir, "resnet18.pth")
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

    # 1. Load ResNet18 model structure (without weights)
    model = torchvision.models.resnet18(weights=None).eval().to(device)

    # 2. Load weights: check for a local .pth file first to avoid re-downloading.
    if os.path.exists(pth_file_path):
        print(f"Found local weights. Loading from: {pth_file_path}")
        model.load_state_dict(torch.load(pth_file_path))
    else:
        print("Local weights not found. Downloading pretrained ResNet18 model...")
        # Use the new 'weights' parameter for pretrained models
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
        print(f"Saving downloaded weights to: {pth_file_path}")
        torch.save(model.state_dict(), pth_file_path)

    # 3. Create a dummy input tensor on the same device
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 4. Export the model to ONNX format
    print(f"Exporting model to ONNX at: {onnx_file_path}")
    torch.onnx.export(model,
                      dummy_input,
                      onnx_file_path,
                      verbose=False,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11,
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})

    print("Model export complete.")
    print(f"PyTorch weights are saved at: {pth_file_path}")
    print(f"ONNX model is saved at: {onnx_file_path}")

if __name__ == "__main__":
    main()