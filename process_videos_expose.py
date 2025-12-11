import sys
import os
import cv2
import torch
import glob
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'expose'))

def check_models():

    expose_data = os.path.join('expose', 'data')
    required_files = [
        os.path.join(expose_data, 'all_means.pkl'),
        os.path.join(expose_data, 'shape_mean.npy'),
        os.path.join(expose_data, 'conf.yaml'),
        os.path.join(expose_data, 'checkpoints'), 
        os.path.join(expose_data, 'models', 'smplx') 
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
            
    if missing:
        print("ERROR: Missing ExPose/SMPL-X model files.")
        print("Please download them from https://expose.is.tue.mpg.de/ and https://smpl-x.is.tue.mpg.de/")
        print("And place them in 'expose/data/'.")
        print("Missing files/folders:")
        for m in missing:
            print(f" - {m}")
        return False
    return True

def main():
    if not check_models():
        print("\nFalling back to MediaPipe script is recommended if you cannot obtain these models.")
        return

    print("Models found! Starting ExPose processing... (Not fully implemented yet as models were missing during dev)")
    

if __name__ == "__main__":
    main()

