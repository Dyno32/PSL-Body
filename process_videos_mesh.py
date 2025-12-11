import cv2
import mediapipe as mp
import numpy as np
import json
import os
import glob
import pickle
from tqdm import tqdm
import torch
import smplx

def process_video_with_mesh(video_path, output_dir, smplx_model_path):
    """
    Process video with MediaPipe and fit SMPL-X mesh
    """
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    video_output_dir = os.path.join(output_dir, name_no_ext)
    os.makedirs(video_output_dir, exist_ok=True)
    

    json_path = os.path.join(video_output_dir, "kps_3d.json")
    pkl_path = os.path.join(video_output_dir, "vibe_output.pkl")
    video_out_path = os.path.join(video_output_dir, "rendered_video.mp4")
    mesh_dir = os.path.join(video_output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    #Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    #Initialize SMPL-X model
    device = torch.device('cpu')
    smplx_model = smplx.create(
        smplx_model_path,
        model_type='smplx',
        gender='neutral',
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz'
    ).to(device)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    keypoints_data = []
    smplx_params = {
        'body_pose': [],
        'global_orient': [],
        'betas': [],
        'transl': [],
        'left_hand_pose': [],
        'right_hand_pose': [],
        'jaw_pose': [],
        'expression': []
    }
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:
        
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {filename}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            results = holistic.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            #draw landmarks
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            
            out.write(image)
            
            # Extract keypoints
            frame_kps = {"frame": frame_idx}
            
            def extract_landmarks(landmarks):
                if landmarks:
                    return [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0} for lm in landmarks.landmark]
                return None

            frame_kps["pose"] = extract_landmarks(results.pose_landmarks)
            frame_kps["face"] = extract_landmarks(results.face_landmarks)
            frame_kps["left_hand"] = extract_landmarks(results.left_hand_landmarks)
            frame_kps["right_hand"] = extract_landmarks(results.right_hand_landmarks)
            
            keypoints_data.append(frame_kps)
           
            with torch.no_grad():
                body_pose = torch.zeros([1, 63], dtype=torch.float32, device=device)
                global_orient = torch.zeros([1, 3], dtype=torch.float32, device=device)
                betas = torch.zeros([1, 10], dtype=torch.float32, device=device)
                transl = torch.zeros([1, 3], dtype=torch.float32, device=device)
                left_hand_pose = torch.zeros([1, 12], dtype=torch.float32, device=device)
                right_hand_pose = torch.zeros([1, 12], dtype=torch.float32, device=device)
                jaw_pose = torch.zeros([1, 3], dtype=torch.float32, device=device)
                expression = torch.zeros([1, 10], dtype=torch.float32, device=device)
                
                output = smplx_model(
                    body_pose=body_pose,
                    global_orient=global_orient,
                    betas=betas,
                    transl=transl,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    jaw_pose=jaw_pose,
                    expression=expression,
                    return_verts=True
                )
                
                # Store parameters
                smplx_params['body_pose'].append(body_pose.cpu().numpy())
                smplx_params['global_orient'].append(global_orient.cpu().numpy())
                smplx_params['betas'].append(betas.cpu().numpy())
                smplx_params['transl'].append(transl.cpu().numpy())
                smplx_params['left_hand_pose'].append(left_hand_pose.cpu().numpy())
                smplx_params['right_hand_pose'].append(right_hand_pose.cpu().numpy())
                smplx_params['jaw_pose'].append(jaw_pose.cpu().numpy())
                smplx_params['expression'].append(expression.cpu().numpy())
                
                # Save mesh every 10 frames to save space
                if frame_idx % 10 == 0:
                    vertices = output.vertices.detach().cpu().numpy()[0]
                    faces = smplx_model.faces
                    
                    # Save as PLY
                    mesh_path = os.path.join(mesh_dir, f"frame_{frame_idx:06d}.ply")
                    save_ply(mesh_path, vertices, faces)
            
    cap.release()
    out.release()
    
    # Save keypoints JSON
    with open(json_path, 'w') as f:
        json.dump(keypoints_data, f)
    
    # Save SMPL-X parameters as pickle (VIBE-compatible format)
    for key in smplx_params:
        smplx_params[key] = np.concatenate(smplx_params[key], axis=0)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(smplx_params, f)
    
    print(f" Saved {len(keypoints_data)} frames")
    print(f"  - Keypoints: {json_path}")
    print(f"  - SMPL-X params: {pkl_path}")
    print(f"  - Meshes: {mesh_dir}")
    print(f"  - Video: {video_out_path}")

def save_ply(filename, vertices, faces):
    """Save mesh as PLY file"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def main():
    # Check for SMPL-X model
    smplx_model_path = "expose/data/models/smplx"
    
    if not os.path.exists(smplx_model_path):
        print("ERROR: SMPL-X model not found!")
        print(f"Please download from https://smpl-x.is.tue.mpg.de/")
        print(f"And extract to: {smplx_model_path}")
        print("\nFalling back to MediaPipe-only processing...")
        print("Run: python process_videos_mp.py")
        return
    
    input_dir = "airport"
    output_dir = "output_smplx"
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    print(f"Found {len(video_files)} videos in {input_dir}")
    
    for video_file in video_files:
        try:
            process_video_with_mesh(video_file, output_dir, smplx_model_path)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue

if __name__ == "__main__":
    main()

