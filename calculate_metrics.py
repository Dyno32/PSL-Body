import numpy as np
import json
from typing import Dict, List, Any

class PoseQualityMetrics:
    """Calculate quality metrics for pose estimation"""
    
    def __init__(self):
        #MediaPipe limb pairs (body landmarks)
        self.body_limbs = [
            (11, 12),  # Shoulders
            (11, 13),  # Left upper arm
            (13, 15),  # Left forearm
            (12, 14),  # Right upper arm
            (14, 16),  # Right forearm
            (11, 23),  # Left torso
            (12, 24),  # Right torso
            (23, 24),  # Hips
            (23, 25),  # Left thigh
            (25, 27),  # Left shin
            (24, 26),  # Right thigh
            (26, 28),  # Right shin
        ]
        
        # Hand finger bones
        self.hand_limbs = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        ]
    
    def calculate_metrics(self, keypoints_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate all quality metrics from keypoints data
        
        Args:
            keypoints_data: List of frame dictionaries with pose/face/hand landmarks
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'overall': {},
            'body': {},
            'face': {},
            'left_hand': {},
            'right_hand': {},
            'per_frame': []
        }
        
        # Calculate per-frame metrics
        for frame_data in keypoints_data:
            frame_metrics = self._calculate_frame_metrics(frame_data)
            metrics['per_frame'].append(frame_metrics)
        
        # Calculate overall statistics
        metrics['overall'] = self._calculate_overall_metrics(metrics['per_frame'])
        metrics['body'] = self._calculate_body_metrics(keypoints_data)
        metrics['face'] = self._calculate_face_metrics(keypoints_data)
        metrics['left_hand'] = self._calculate_hand_metrics(keypoints_data, 'left_hand')
        metrics['right_hand'] = self._calculate_hand_metrics(keypoints_data, 'right_hand')
        
        # Calculate temporal metrics (jitter, smoothness)
        temporal_metrics = self._calculate_temporal_metrics(keypoints_data)
        metrics['temporal'] = temporal_metrics
        
        return metrics
    
    def _calculate_frame_metrics(self, frame_data: Dict) -> Dict:
        """Calculate metrics for a single frame"""
        metrics = {
            'frame': frame_data['frame'],
            'body_detected': frame_data['pose'] is not None,
            'face_detected': frame_data['face'] is not None,
            'left_hand_detected': frame_data['left_hand'] is not None,
            'right_hand_detected': frame_data['right_hand'] is not None,
        }
        
        # Average confidence/visibility scores
        if frame_data['pose']:
            visibilities = [kp['visibility'] for kp in frame_data['pose']]
            metrics['body_confidence'] = np.mean(visibilities)
            metrics['body_min_confidence'] = np.min(visibilities)
        else:
            metrics['body_confidence'] = 0.0
            metrics['body_min_confidence'] = 0.0
        
        if frame_data['face']:
            # Face landmarks don't have visibility, assume 1.0 if detected
            metrics['face_confidence'] = 1.0
        else:
            metrics['face_confidence'] = 0.0
        
        if frame_data['left_hand']:
            metrics['left_hand_confidence'] = 1.0
        else:
            metrics['left_hand_confidence'] = 0.0
            
        if frame_data['right_hand']:
            metrics['right_hand_confidence'] = 1.0
        else:
            metrics['right_hand_confidence'] = 0.0
        
        return metrics
    
    def _calculate_overall_metrics(self, per_frame_metrics: List[Dict]) -> Dict:
        """Calculate overall statistics across all frames"""
        total_frames = len(per_frame_metrics)
        
        body_detected = sum(1 for m in per_frame_metrics if m['body_detected'])
        face_detected = sum(1 for m in per_frame_metrics if m['face_detected'])
        left_hand_detected = sum(1 for m in per_frame_metrics if m['left_hand_detected'])
        right_hand_detected = sum(1 for m in per_frame_metrics if m['right_hand_detected'])
        
        body_confidences = [m['body_confidence'] for m in per_frame_metrics if m['body_detected']]
        
        return {
            'total_frames': total_frames,
            'body_detection_rate': body_detected / total_frames,
            'face_detection_rate': face_detected / total_frames,
            'left_hand_detection_rate': left_hand_detected / total_frames,
            'right_hand_detection_rate': right_hand_detected / total_frames,
            'avg_body_confidence': np.mean(body_confidences) if body_confidences else 0.0,
            'min_body_confidence': np.min(body_confidences) if body_confidences else 0.0,
            'max_body_confidence': np.max(body_confidences) if body_confidences else 0.0,
            'std_body_confidence': np.std(body_confidences) if body_confidences else 0.0,
        }
    
    def _calculate_body_metrics(self, keypoints_data: List[Dict]) -> Dict:
        """Calculate body-specific metrics"""
        limb_lengths = {i: [] for i in range(len(self.body_limbs))}
        
        for frame_data in keypoints_data:
            if not frame_data['pose']:
                continue
            
            pose = frame_data['pose']
            for limb_idx, (start, end) in enumerate(self.body_limbs):
                if start < len(pose) and end < len(pose):
                    p1 = np.array([pose[start]['x'], pose[start]['y'], pose[start]['z']])
                    p2 = np.array([pose[end]['x'], pose[end]['y'], pose[end]['z']])
                    length = np.linalg.norm(p2 - p1)
                    limb_lengths[limb_idx].append(length)
        
        #Calculate limb consistency
        limb_consistency = {}
        for limb_idx, lengths in limb_lengths.items():
            if lengths:
                limb_consistency[f'limb_{limb_idx}'] = {
                    'mean_length': np.mean(lengths),
                    'std_length': np.std(lengths),
                    'coefficient_of_variation': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
                }
        
        # Overall consistency score 
        all_cvs = [v['coefficient_of_variation'] for v in limb_consistency.values()]
        
        return {
            'limb_consistency': limb_consistency,
            'avg_limb_consistency': np.mean(all_cvs) if all_cvs else 0.0,
            'limb_consistency_score': 1.0 - min(np.mean(all_cvs), 1.0) if all_cvs else 0.0  # 1.0 = perfect
        }
    
    def _calculate_hand_metrics(self, keypoints_data: List[Dict], hand_key: str) -> Dict:
        """Calculate hand-specific metrics"""
        finger_lengths = {i: [] for i in range(len(self.hand_limbs))}
        
        for frame_data in keypoints_data:
            if not frame_data[hand_key]:
                continue
            
            hand = frame_data[hand_key]
            for bone_idx, (start, end) in enumerate(self.hand_limbs):
                if start < len(hand) and end < len(hand):
                    p1 = np.array([hand[start]['x'], hand[start]['y'], hand[start]['z']])
                    p2 = np.array([hand[end]['x'], hand[end]['y'], hand[end]['z']])
                    length = np.linalg.norm(p2 - p1)
                    finger_lengths[bone_idx].append(length)
        
        # Calculate finger bone consistency
        bone_consistency = {}
        for bone_idx, lengths in finger_lengths.items():
            if lengths:
                bone_consistency[f'bone_{bone_idx}'] = {
                    'mean_length': np.mean(lengths),
                    'std_length': np.std(lengths),
                    'coefficient_of_variation': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
                }
        
        all_cvs = [v['coefficient_of_variation'] for v in bone_consistency.values()]
        
        return {
            'bone_consistency': bone_consistency,
            'avg_bone_consistency': np.mean(all_cvs) if all_cvs else 0.0,
            'bone_consistency_score': 1.0 - min(np.mean(all_cvs), 1.0) if all_cvs else 0.0
        }
    
    def _calculate_face_metrics(self, keypoints_data: List[Dict]) -> Dict:
        """Calculate face-specific metrics"""
        detected_frames = sum(1 for f in keypoints_data if f['face'] is not None)
        total_frames = len(keypoints_data)
        
        return {
            'detection_rate': detected_frames / total_frames if total_frames > 0 else 0.0,
            'num_landmarks': 468,  # MediaPipe face mesh
        }
    
    def _calculate_temporal_metrics(self, keypoints_data: List[Dict]) -> Dict:
        """Calculate temporal smoothness and jitter metrics"""
        # Calculate jitter for body keypoints
        body_jitter = self._calculate_jitter(keypoints_data, 'pose')
        left_hand_jitter = self._calculate_jitter(keypoints_data, 'left_hand')
        right_hand_jitter = self._calculate_jitter(keypoints_data, 'right_hand')
        
        return {
            'body_jitter': body_jitter,
            'left_hand_jitter': left_hand_jitter,
            'right_hand_jitter': right_hand_jitter,
            'overall_jitter': (body_jitter + left_hand_jitter + right_hand_jitter) / 3,
            'smoothness_score': 1.0 - min((body_jitter + left_hand_jitter + right_hand_jitter) / 3, 1.0)
        }
    
    def _calculate_jitter(self, keypoints_data: List[Dict], key: str) -> float:
        """
        Calculate jitter (average acceleration magnitude) for a set of keypoints
        Lower jitter = smoother motion
        """
        positions = []
        
        for frame_data in keypoints_data:
            if frame_data[key]:
                frame_positions = []
                for kp in frame_data[key]:
                    frame_positions.append([kp['x'], kp['y'], kp['z']])
                positions.append(np.array(frame_positions))
        
        if len(positions) < 3:
            return 0.0
        
        # Calculate acceleration (second derivative)
        accelerations = []
        for i in range(1, len(positions) - 1):
            velocity_prev = positions[i] - positions[i-1]
            velocity_next = positions[i+1] - positions[i]
            acceleration = velocity_next - velocity_prev
            accelerations.append(np.mean(np.linalg.norm(acceleration, axis=1)))
        
        return np.mean(accelerations) if accelerations else 0.0
    
    def generate_report(self, metrics: Dict) -> str:
        """Generate a human-readable report"""
        report = []
        report.append("=" * 70)
        report.append("3D POSE ESTIMATION QUALITY METRICS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall metrics
        overall = metrics['overall']
        report.append("OVERALL DETECTION RATES:")
        report.append(f"  Total Frames: {overall['total_frames']}")
        report.append(f"  Body Detection: {overall['body_detection_rate']:.2%}")
        report.append(f"  Face Detection: {overall['face_detection_rate']:.2%}")
        report.append(f"  Left Hand Detection: {overall['left_hand_detection_rate']:.2%}")
        report.append(f"  Right Hand Detection: {overall['right_hand_detection_rate']:.2%}")
        report.append("")
        
        # Confidence scores
        report.append("CONFIDENCE SCORES:")
        report.append(f"  Average Body Confidence: {overall['avg_body_confidence']:.3f}")
        report.append(f"  Min Body Confidence: {overall['min_body_confidence']:.3f}")
        report.append(f"  Max Body Confidence: {overall['max_body_confidence']:.3f}")
        report.append(f"  Std Body Confidence: {overall['std_body_confidence']:.3f}")
        report.append("")
        
        # Temporal metrics
        temporal = metrics['temporal']
        report.append("TEMPORAL QUALITY (Jitter & Smoothness):")
        report.append(f"  Body Jitter: {temporal['body_jitter']:.4f} (lower is better)")
        report.append(f"  Left Hand Jitter: {temporal['left_hand_jitter']:.4f}")
        report.append(f"  Right Hand Jitter: {temporal['right_hand_jitter']:.4f}")
        report.append(f"  Overall Smoothness Score: {temporal['smoothness_score']:.3f} (0-1, higher is better)")
        report.append("")
        
        # Limb consistency
        body = metrics['body']
        report.append("LIMB CONSISTENCY:")
        report.append(f"  Average Limb Consistency: {body['avg_limb_consistency']:.4f} (lower is better)")
        report.append(f"  Limb Consistency Score: {body['limb_consistency_score']:.3f} (0-1, higher is better)")
        report.append("")
        
        # Hand consistency
        left_hand = metrics['left_hand']
        right_hand = metrics['right_hand']
        report.append("HAND BONE CONSISTENCY:")
        report.append(f"  Left Hand Consistency Score: {left_hand['bone_consistency_score']:.3f}")
        report.append(f"  Right Hand Consistency Score: {right_hand['bone_consistency_score']:.3f}")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


def calculate_and_save_metrics(json_path: str, output_path: str = None):
    """
    Load keypoints JSON and calculate quality metrics
    
    Args:
        json_path: Path to kps_3d.json file
        output_path: Path to save metrics (default: same dir as json_path)
    """
    # Load keypoints
    with open(json_path, 'r') as f:
        keypoints_data = json.load(f)
    
    # Calculate metrics
    calculator = PoseQualityMetrics()
    metrics = calculator.calculate_metrics(keypoints_data)
    
    # Generate report
    report = calculator.generate_report(metrics)
    
    # Save metrics
    if output_path is None:
        import os
        base_dir = os.path.dirname(json_path)
        output_path = os.path.join(base_dir, 'quality_metrics.json')
        report_path = os.path.join(base_dir, 'quality_report.txt')
    else:
        report_path = output_path.replace('.json', '.txt')
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nMetrics saved to: {output_path}")
    print(f"Report saved to: {report_path}")
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python calculate_metrics.py <path_to_kps_3d.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    calculate_and_save_metrics(json_path)


