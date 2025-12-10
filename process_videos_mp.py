import cv2
import mediapipe as mp
import numpy as np
import json
import os
import glob
from tqdm import tqdm

def process_video(video_path, output_dir):
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, name_no_ext)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Output paths
    json_path = os.path.join(video_output_dir, "kps_3d.json")
    video_out_path = os.path.join(video_output_dir, "rendered_video.mp4")
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    keypoints_data = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:
        
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {filename}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process
            results = holistic.process(image)
            
            # Draw
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
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
            
    cap.release()
    out.release()
    
    # Save keypoints JSON
    with open(json_path, 'w') as f:
        json.dump(keypoints_data, f)
    
    # Calculate and save quality metrics
    print(f"\nCalculating quality metrics...")
    from calculate_metrics import calculate_and_save_metrics
    try:
        calculate_and_save_metrics(json_path)
    except Exception as e:
        print(f"Warning: Could not calculate metrics: {e}")

def main():
    input_dir = "airport"
    output_dir = "output_mediapipe"
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    print(f"Found {len(video_files)} videos in {input_dir}")
    
    # Summary statistics
    all_metrics = []
    
    for video_file in video_files:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(video_file)}")
        print(f"{'='*70}")
        process_video(video_file, output_dir)
        
        # Load metrics for summary
        name_no_ext = os.path.splitext(os.path.basename(video_file))[0]
        metrics_path = os.path.join(output_dir, name_no_ext, "quality_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                all_metrics.append({
                    'video': os.path.basename(video_file),
                    'metrics': metrics
                })
    
    # Generate summary report
    if all_metrics:
        summary_path = os.path.join(output_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Generate summary statistics
        generate_summary_report(all_metrics, output_dir)
        print(f"\n{'='*70}")
        print(f"Summary metrics saved to: {summary_path}")
        print(f"{'='*70}")

def generate_summary_report(all_metrics: list, output_dir: str):
    """Generate a summary report across all videos"""
    report = []
    report.append("=" * 70)
    report.append("SUMMARY REPORT - ALL VIDEOS")
    report.append("=" * 70)
    report.append(f"\nTotal Videos Processed: {len(all_metrics)}\n")
    
    # Aggregate statistics
    body_detection_rates = []
    face_detection_rates = []
    left_hand_detection_rates = []
    right_hand_detection_rates = []
    avg_confidences = []
    smoothness_scores = []
    limb_consistency_scores = []
    
    for item in all_metrics:
        m = item['metrics']['overall']
        t = item['metrics']['temporal']
        b = item['metrics']['body']
        
        body_detection_rates.append(m['body_detection_rate'])
        face_detection_rates.append(m['face_detection_rate'])
        left_hand_detection_rates.append(m['left_hand_detection_rate'])
        right_hand_detection_rates.append(m['right_hand_detection_rate'])
        avg_confidences.append(m['avg_body_confidence'])
        smoothness_scores.append(t['smoothness_score'])
        limb_consistency_scores.append(b['limb_consistency_score'])
    
    report.append("AVERAGE DETECTION RATES:")
    report.append(f"  Body: {np.mean(body_detection_rates):.2%} ± {np.std(body_detection_rates):.2%}")
    report.append(f"  Face: {np.mean(face_detection_rates):.2%} ± {np.std(face_detection_rates):.2%}")
    report.append(f"  Left Hand: {np.mean(left_hand_detection_rates):.2%} ± {np.std(left_hand_detection_rates):.2%}")
    report.append(f"  Right Hand: {np.mean(right_hand_detection_rates):.2%} ± {np.std(right_hand_detection_rates):.2%}")
    report.append("")
    
    report.append("AVERAGE QUALITY SCORES:")
    report.append(f"  Confidence: {np.mean(avg_confidences):.3f} ± {np.std(avg_confidences):.3f}")
    report.append(f"  Smoothness: {np.mean(smoothness_scores):.3f} ± {np.std(smoothness_scores):.3f}")
    report.append(f"  Limb Consistency: {np.mean(limb_consistency_scores):.3f} ± {np.std(limb_consistency_scores):.3f}")
    report.append("")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save summary report
    summary_report_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_report_path, 'w') as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
