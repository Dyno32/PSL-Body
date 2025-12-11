
import json
import os
import glob
import numpy as np

def generate_summary_from_existing():
    output_dir = "output_mediapipe"
    

    metrics_files = glob.glob(os.path.join(output_dir, "*", "quality_metrics.json"))
    
    print(f"Found {len(metrics_files)} videos with metrics")
    
    all_metrics = []
    for metrics_file in metrics_files:
        video_name = os.path.basename(os.path.dirname(metrics_file))
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append({
                    'video': video_name,
                    'metrics': metrics
                })
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
    
    if not all_metrics:
        print("No metrics found!")
        return
    

    report = []
    report.append("=" * 70)
    report.append("SUMMARY REPORT - ALL VIDEOS")
    report.append("=" * 70)
    report.append(f"\nTotal Videos Processed: {len(all_metrics)}\n")
    
    body_detection_rates = []
    face_detection_rates = []
    left_hand_detection_rates = []
    right_hand_detection_rates = []
    avg_confidences = []
    smoothness_scores = []
    limb_consistency_scores = []
    left_hand_consistency = []
    right_hand_consistency = []
    
    for item in all_metrics:
        m = item['metrics']['overall']
        t = item['metrics']['temporal']
        b = item['metrics']['body']
        lh = item['metrics']['left_hand']
        rh = item['metrics']['right_hand']
        
        body_detection_rates.append(m['body_detection_rate'])
        face_detection_rates.append(m['face_detection_rate'])
        left_hand_detection_rates.append(m['left_hand_detection_rate'])
        right_hand_detection_rates.append(m['right_hand_detection_rate'])
        avg_confidences.append(m['avg_body_confidence'])
        smoothness_scores.append(t['smoothness_score'])
        limb_consistency_scores.append(b['limb_consistency_score'])
        left_hand_consistency.append(lh['bone_consistency_score'])
        right_hand_consistency.append(rh['bone_consistency_score'])
    
    report.append("AVERAGE DETECTION RATES:")
    report.append(f"  Body: {np.mean(body_detection_rates):.2%} ± {np.std(body_detection_rates):.2%}")
    report.append(f"  Face: {np.mean(face_detection_rates):.2%} ± {np.std(face_detection_rates):.2%}")
    report.append(f"  Left Hand: {np.mean(left_hand_detection_rates):.2%} ± {np.std(left_hand_detection_rates):.2%}")
    report.append(f"  Right Hand: {np.mean(right_hand_detection_rates):.2%} ± {np.std(right_hand_detection_rates):.2%}")
    report.append("")
    
    report.append("AVERAGE QUALITY SCORES:")
    report.append(f"  Body Confidence: {np.mean(avg_confidences):.3f} ± {np.std(avg_confidences):.3f}")
    report.append(f"  Temporal Smoothness: {np.mean(smoothness_scores):.3f} ± {np.std(smoothness_scores):.3f}")
    report.append(f"  Limb Consistency: {np.mean(limb_consistency_scores):.3f} ± {np.std(limb_consistency_scores):.3f}")
    report.append(f"  Left Hand Consistency: {np.mean(left_hand_consistency):.3f} ± {np.std(left_hand_consistency):.3f}")
    report.append(f"  Right Hand Consistency: {np.mean(right_hand_consistency):.3f} ± {np.std(right_hand_consistency):.3f}")
    report.append("")
    
    report.append("DETAILED STATISTICS:")
    report.append(f"  Min Body Detection: {np.min(body_detection_rates):.2%}")
    report.append(f"  Max Body Detection: {np.max(body_detection_rates):.2%}")
    report.append(f"  Min Confidence: {np.min(avg_confidences):.3f}")
    report.append(f"  Max Confidence: {np.max(avg_confidences):.3f}")
    report.append(f"  Min Smoothness: {np.min(smoothness_scores):.3f}")
    report.append(f"  Max Smoothness: {np.max(smoothness_scores):.3f}")
    report.append("")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
   
    summary_report_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_report_path, 'w') as f:
        f.write(report_text)
    
   
    summary_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nSummary saved to: {summary_report_path}")
    print(f"Detailed metrics saved to: {summary_path}")

if __name__ == "__main__":
    generate_summary_from_existing()

