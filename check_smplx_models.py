"""
Check if SMPL-X models are installed and provide download instructions
"""
import os

def check_smplx_models():
    model_path = "expose/data/models/smplx"
    required_files = [
        "SMPLX_NEUTRAL.npz",
        "SMPLX_MALE.npz", 
        "SMPLX_FEMALE.npz"
    ]
    
    print("Checking for SMPL-X models...")
    print(f"Looking in: {os.path.abspath(model_path)}")
    print()
    
    if not os.path.exists(model_path):
        print("❌ SMPL-X model directory not found!")
        print()
        print_download_instructions()
        return False
    
    missing = []
    found = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            found.append(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            missing.append(f"✗ {file}")
    
    if found:
        print("Found models:")
        for f in found:
            print(f"  {f}")
        print()
    
    if missing:
        print("Missing models:")
        for m in missing:
            print(f"  {m}")
        print()
        print_download_instructions()
        return False
    
    print("✅ All SMPL-X models are installed!")
    print()
    print("You can now run:")
    print("  venv_expose\\Scripts\\python process_videos_mesh.py")
    return True

def print_download_instructions():
    print("=" * 60)
    print("SMPL-X MODEL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Register at: https://smpl-x.is.tue.mpg.de/")
    print("   - Click 'Register' (top right)")
    print("   - Fill in your details and accept license")
    print("   - Check email for confirmation")
    print()
    print("2. Download SMPL-X v1.1 (NPZ format)")
    print("   - Login at: https://smpl-x.is.tue.mpg.de/login")
    print("   - Go to Downloads section")
    print("   - Download 'SMPL-X v1.1' (~100MB)")
    print()
    print("3. Extract to: expose/data/models/smplx/")
    print()
    print("4. Run this script again to verify")
    print()
    print("See SMPLX_DOWNLOAD_GUIDE.md for detailed instructions")
    print("=" * 60)

if __name__ == "__main__":
    check_smplx_models()
