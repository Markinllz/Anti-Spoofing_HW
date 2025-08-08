import os
import csv
import numpy as np
import torch
from copy import deepcopy

from src.metrics.eer import EERMetric

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(BASE_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")
SOLUTIONS_DIR = os.path.join(BASE_DIR, "students_solutions")
OUTPUT_CSV = os.path.join(BASE_DIR, "grades.csv")

# --- Load protocol ---
index = []
with open(PROTOCOL_PATH, "r") as protocol:
    for line in protocol:
        _, key, _, alg_id, label = line.strip().split()
        index.append({
            "key": key,
            "label": 1 if label == "bonafide" else 0
        })

# --- Prepare output ---
results = []

# --- Process each student file ---
for filename in os.listdir(SOLUTIONS_DIR):
    if filename.endswith(".csv"):
        name = filename.replace(".csv", "")
        filepath = os.path.join(SOLUTIONS_DIR, filename)

        # Load student scores into a dict
        student_scores = {}
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 2:
                    continue  # skip malformed lines
                key, score = row
                try:
                    student_scores[key] = float(score)
                except ValueError:
                    print(f"WARNING: Invalid score '{score}' for key '{key}' in {filename}")
                    continue

        # Build student index
        student_index = deepcopy(index)
        missing_keys = []
        for entry in student_index:
            key = entry["key"]
            if key not in student_scores:
                missing_keys.append(key)
            else:
                entry["score"] = student_scores[key]
        
        if missing_keys:
            print(f"WARNING: Missing {len(missing_keys)} keys in {filename}: {missing_keys[:5]}...")
            # Remove entries with missing scores
            student_index = [entry for entry in student_index if entry["key"] in student_scores]

        # Extract scores and labels
        scores = np.array([entry["score"] for entry in student_index])
        labels = np.array([entry["label"] for entry in student_index])

        if len(scores) != len(index):
            print(f"WARNING: {filename} has {len(scores)} scores, expected {len(index)}")
            if len(scores) == 0:
                print(f"ERROR: No valid scores in {filename}")
                continue

        # Additional check: verify score distribution
        if len(scores) > 0:
            print(f"Score statistics for {filename}:")
            print(f"   Min: {np.min(scores):.4f}")
            print(f"   Max: {np.max(scores):.4f}")
            print(f"   Mean: {np.mean(scores):.4f}")
            print(f"   Std: {np.std(scores):.4f}")
            
            # Check for reasonable score range (should be 0-1 for bonafide probabilities)
            if np.min(scores) < 0 or np.max(scores) > 1:
                print(f"WARNING: Scores outside [0,1] range! This might indicate wrong format.")
            if np.std(scores) < 0.01:
                print(f"WARNING: Very low score variance! Model might not be trained properly.")

        assert len(scores) == len(index), "Not enough / too many scores"

        # Scores are already bonafide probabilities from softmax (0-1 range)
        # No need to apply additional transformations since they're already probabilities
        scores_probs = scores

        # Compute EER
        bona_cm = scores_probs[labels == 1]
        spoof_cm = scores_probs[labels == 0]
        
        # Use the same EER computation as in training
        eer_metric = EERMetric()
        eer, _ = eer_metric.compute_eer_from_arrays(bona_cm, spoof_cm)

        eer *= 100 # in %

        # Grade calculation
        if eer > 9.5:
            grade = 0
        elif eer < 5.3:
            grade = 10
        else:
            # Linear interpolation between 4 and 10
            grade = 4 + (9.5 - eer) * (6 / (9.5 - 5.3))

        results.append({
            "name": name,
            "email": name + "@edu.hse.ru",
            "eer": round(eer, 4),
            "grade": round(grade, 2)
        })

# --- Write output CSV ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "email", "eer", "grade"])
    writer.writeheader()
    writer.writerows(results)

print(f"Grading complete. Results saved to {OUTPUT_CSV}")
