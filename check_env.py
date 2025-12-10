import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("OpenCV not found")

try:
    import smplx
    print(f"SMPL-X version: {smplx.__version__}")
except ImportError:
    print("SMPL-X not found")

try:
    import pytorch3d
    print(f"PyTorch3D version: {pytorch3d.__version__}")
except ImportError:
    print("PyTorch3D not found")

try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
except ImportError:
    print("MediaPipe not found")
