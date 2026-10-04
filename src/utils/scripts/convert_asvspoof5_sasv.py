#!/usr/bin/env python3
"""
Convert ASVspoof5 SASV dataset JSON files to the unified format.

Input format:
{
  "annotation": [
    {
      "id": "asv5_00000000",
      "enroll_path": "...",
      "test_path": "...",
      "label": 0 or 1,
      "pair_type": "BBT" | "BBF" | "BSF"
    }
  ]
}

Output format:
{
  "task_id": "...",
  "gt": "verified" | "rejected" | "spoof",
  "reference_audios": [{"audio_id": "...", "original_path": "...", "duration": 0.0}],
  "query_audios": [{"audio_id": "...", "original_path": "...", "duration": 0.0}]
}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import soundfile as sf
from tqdm import tqdm


def get_audio_duration_sec(path: str, verbose: bool = False) -> float:
    """Get audio duration from file headers only (no full audio load)."""
    if not os.path.exists(path):
        return 0.0
    
    try:
        with sf.SoundFile(path) as f:
            # len(f) = num frames; reads header only
            return len(f) / f.samplerate
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read duration for {path}: {e}", file=sys.stderr)
        return 0.0


def get_audio_id_from_path(path: str) -> str:
    """Extract audio_id from file path (basename without extension)."""
    return Path(path).stem


def map_pair_type_to_gt(pair_type: str, label: int) -> str:
    """
    Map pair_type and label to ground truth.
    - BBT (Bonafide-Bonafide True), label=1 → "verified"
    - BBF (Bonafide-Bonafide False), label=0 → "rejected"
    - BSF (Bonafide-Spoof False), label=0 → "spoof"
    """
    if pair_type == "BBT" and label == 1:
        return "verified"
    elif pair_type == "BBF" and label == 0:
        return "rejected"
    elif pair_type == "BSF" and label == 0:
        return "spoof"
    else:
        # Fallback: use label
        if label == 1:
            return "verified"
        else:
            # Default to rejected if we can't determine spoof
            return "rejected"


def convert_entry(entry: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Convert a single entry from input format to output format."""
    task_id = entry["id"]
    enroll_path = entry["enroll_path"]
    test_path = entry["test_path"]
    pair_type = entry.get("pair_type", "")
    label = entry.get("label", 0)
    
    # Get ground truth
    gt = map_pair_type_to_gt(pair_type, label)
    
    # Extract durations (from headers only)
    enroll_duration = get_audio_duration_sec(enroll_path, verbose=verbose)
    test_duration = get_audio_duration_sec(test_path, verbose=verbose)
    
    # Create audio IDs
    enroll_audio_id = get_audio_id_from_path(enroll_path)
    test_audio_id = get_audio_id_from_path(test_path)
    
    return {
        "task_id": task_id,
        "gt": gt,
        "reference_audios": [
            {
                "audio_id": enroll_audio_id,
                "original_path": enroll_path,
                "duration": enroll_duration,
            }
        ],
        "query_audios": [
            {
                "audio_id": test_audio_id,
                "original_path": test_path,
                "duration": test_duration,
            }
        ],
    }


def get_checkpoint_path(output_path: str) -> str:
    """Get checkpoint file path for a given output path."""
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)
    checkpoint_name = output_basename.replace(".json", "_checkpoint.json")
    return os.path.join(output_dir, checkpoint_name)


def load_checkpoint(checkpoint_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: str, converted_entries: List[Dict[str, Any]]):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(converted_entries, f, indent=2)


def convert_file_with_data(
    data: Dict[str, Any],
    output_path: str, 
    checkpoint_interval: int = 10000,
    checkpoint_time_interval: float = 300.0  # 5 minutes
):
    """
    Convert data with checkpoint support.
    
    Args:
        data: Already loaded JSON data
        output_path: Path to output JSON file
        checkpoint_interval: Save checkpoint every N entries
        checkpoint_time_interval: Save checkpoint every N seconds
    """
    checkpoint_path = get_checkpoint_path(output_path)
    
    annotations = data.get("annotation", [])
    num_entries = len(annotations)
    print(f"Found {num_entries} entries")
    
    # Try to load checkpoint
    converted_entries = load_checkpoint(checkpoint_path)
    start_idx = len(converted_entries) if converted_entries else 0
    
    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx}/{num_entries} entries already processed")
    else:
        converted_entries = []
    
    # Track time for time-based checkpoints
    last_checkpoint_time = time.time()
    
    # Process entries with tqdm progress bar
    with tqdm(total=num_entries, initial=start_idx, desc=f"Converting {os.path.basename(output_path)}") as pbar:
        for i in range(start_idx, num_entries):
            entry = annotations[i]
            converted_entry = convert_entry(entry, verbose=False)
            converted_entries.append(converted_entry)
            
            # Update progress bar
            pbar.update(1)
            
            # Save checkpoint periodically
            current_time = time.time()
            should_checkpoint = (
                (i + 1) % checkpoint_interval == 0 or
                (current_time - last_checkpoint_time) >= checkpoint_time_interval
            )
            
            if should_checkpoint:
                save_checkpoint(checkpoint_path, converted_entries)
                last_checkpoint_time = current_time
                pbar.set_postfix({"checkpoint": "saved"})
    
    # Final save
    print(f"Writing final output to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted_entries, f, indent=2)
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")
    
    print(f"Done! Converted {len(converted_entries)} entries")
    return len(converted_entries)


def main():
    parser = argparse.ArgumentParser(description="Convert ASVspoof5 SASV JSON annotations.")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./data"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    data_root = args.data_root
    input_files = {
        "train": os.path.join(data_root, "asvspoof5_sasv_train.json"),
        "val": os.path.join(data_root, "asvspoof5_sasv_dev.json"),
        "test": os.path.join(data_root, "asvspoof5_sasv_batch.json"),
    }
    output_dir = args.output_dir or os.path.join(data_root, "sasv_meta")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each file
    results = {}
    for split_name, input_path in input_files.items():
        if not os.path.exists(input_path):
            print(f"Warning: Input file not found: {input_path}", file=sys.stderr)
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")
        
        # Read file once to get count for filename
        with open(input_path, "r") as f:
            data = json.load(f)
        num_entries = len(data.get("annotation", []))
        
        # Format: sasv_pairs_train_1185k.json (with actual count)
        if num_entries >= 1000000:
            count_str = f"{num_entries // 1000}k"
        elif num_entries >= 1000:
            count_str = f"{num_entries // 1000}k"
        else:
            count_str = f"{num_entries}"
        
        output_filename = f"sasv_pairs_{split_name}_{count_str}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert entries with checkpoint support (pass data to avoid re-reading)
        count = convert_file_with_data(
            data,
            output_path,
            checkpoint_interval=10000,  # Save checkpoint every 10k entries
            checkpoint_time_interval=300.0  # Save checkpoint every 5 minutes
        )
        
        results[split_name] = {
            "input": input_path,
            "output": output_path,
            "count": count,
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    for split_name, info in results.items():
        print(f"{split_name:10s}: {info['count']:8d} entries -> {info['output']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
