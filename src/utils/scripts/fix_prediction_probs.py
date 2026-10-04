#!/usr/bin/env python3
"""
Fix unnormalized real_prob and fake_prob in existing prediction files.
This script:
1. Normalizes the probabilities so they sum to 1
2. Fixes the confidence field to be P(predicted class) instead of always real_prob
"""
import json
import glob
import sys
import os
import re
from pathlib import Path


def extract_answer(text):
    """Extract final answer from either reasoning or hard_label format."""
    if not text:
        return ""
    text_str = str(text).strip()
    # Reasoning format: <answer>Real/Fake</answer>
    match = re.search(r"<answer>(.*?)</answer>", text_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().capitalize()
    # Hard-label format: Final Answer: Real/Fake
    match = re.search(r"Final\s*Answer:\s*(Real|Fake)", text_str, re.IGNORECASE)
    if match:
        return match.group(1).strip().capitalize()
    # Fallback
    text_lower = text_str.lower()
    if "fake" in text_lower:
        return "Fake"
    if "real" in text_lower:
        return "Real"
    return text_str


def fix_prediction_file(input_path: str, output_path: str = None):
    """Fix unnormalized probabilities and confidence field in a prediction file."""
    if output_path is None:
        output_path = input_path
    
    fixed_predictions = []
    total_normalized = 0
    total_confidence_fixed = 0
    
    with open(input_path, 'r') as f:
        for line in f:
            try:
                pred = json.loads(line)
                
                # Check if we have real_prob and fake_prob
                if 'real_prob' in pred and 'fake_prob' in pred:
                    real_prob = pred['real_prob']
                    fake_prob = pred['fake_prob']
                    
                    # Normalize if they don't sum to ~1
                    total = real_prob + fake_prob
                    if total > 1e-12 and abs(total - 1.0) > 1e-6:
                        pred['real_prob'] = real_prob / total
                        pred['fake_prob'] = fake_prob / total
                        real_prob = pred['real_prob']
                        fake_prob = pred['fake_prob']
                        total_normalized += 1
                    
                    # Fix confidence field to be P(predicted class)
                    if 'confidence' in pred and 'output' in pred:
                        predicted_answer = extract_answer(pred['output']).lower()
                        
                        # Calculate correct confidence
                        if predicted_answer == "fake":
                            correct_confidence = fake_prob
                        elif predicted_answer == "real":
                            correct_confidence = real_prob
                        else:
                            # If we can't determine, use max probability
                            correct_confidence = max(real_prob, fake_prob)
                        
                        # Only update if different
                        if abs(pred['confidence'] - correct_confidence) > 1e-6:
                            pred['confidence'] = correct_confidence
                            total_confidence_fixed += 1
                
                fixed_predictions.append(pred)
            except json.JSONDecodeError:
                continue
    
    # Write back
    with open(output_path, 'w') as f:
        for pred in fixed_predictions:
            f.write(json.dumps(pred) + '\n')
    
    return len(fixed_predictions), total_normalized, total_confidence_fixed


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_prediction_probs.py <log_dir>")
        print("Example: python fix_prediction_probs.py /path/to/logs/test_hard_label_val_133k/")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    # Find all prediction files
    pred_files = glob.glob(os.path.join(log_dir, "predictions_*.jsonl"))
    if not pred_files:
        pred_files = glob.glob(os.path.join(log_dir, "samples_*.jsonl"))
    
    if not pred_files:
        print(f"No prediction files found in {log_dir}")
        sys.exit(1)
    
    print(f"Found {len(pred_files)} prediction file(s)")
    
    total_predictions = 0
    total_fixed = 0
    
    for pred_file in pred_files:
        print(f"\nProcessing: {pred_file}")
        n_preds, n_normalized, n_conf_fixed = fix_prediction_file(pred_file)
        total_predictions += n_preds
        total_fixed += n_normalized
        print(f"  Normalized probabilities: {n_normalized}/{n_preds}")
        print(f"  Fixed confidence field: {n_conf_fixed}/{n_preds}")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_predictions} predictions processed")
    print(f"  Normalized probabilities: {total_fixed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
