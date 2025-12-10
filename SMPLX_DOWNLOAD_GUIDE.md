# SMPL-X Model Download Guide

## Step 1: Register for SMPL-X

1. Go to: https://smpl-x.is.tue.mpg.de/
2. Click "Register" (top right)
3. Fill in the form:
   - Name, Email, Organization
   - Accept the license terms
4. Check your email for confirmation

## Step 2: Download SMPL-X Model

1. Login at: https://smpl-x.is.tue.mpg.de/login
2. Go to "Downloads" section
3. Download: **SMPL-X v1.1 (NPZ format)** - ~100MB
4. Save the zip file to your Downloads folder

## Step 3: Extract Models

After downloading, run this command:

```powershell
# Create the directory
mkdir expose\data\models\smplx

# Extract the zip (replace with your actual download path)
Expand-Archive -Path "$env:USERPROFILE\Downloads\smplx_v1_1.zip" -DestinationPath "expose\data\models\smplx"
```

Or manually:
1. Extract the downloaded zip file
2. Copy the contents to: `expose\data\models\smplx\`
3. You should have files like:
   - `SMPLX_NEUTRAL.npz`
   - `SMPLX_MALE.npz`
   - `SMPLX_FEMALE.npz`

## Step 4: Verify Installation

Run this to check:

```powershell
venv_expose\Scripts\python -c "import os; path='expose/data/models/smplx/SMPLX_NEUTRAL.npz'; print('✓ Found!' if os.path.exists(path) else '✗ Not found')"
```

## Step 5: Process Videos with Mesh Output

Once models are installed, run:

```powershell
venv_expose\Scripts\python process_videos_mesh.py
```

This will generate for each video:
- `vibe_output.pkl` - SMPL-X parameters
- `kps_3d.json` - 3D keypoints
- `rendered_video.mp4` - Video with mesh overlay
- `meshes/` - Folder with `.ply` mesh files (every 10th frame)

---

**Note**: The download requires manual registration due to licensing. There's no automated way to bypass this.
