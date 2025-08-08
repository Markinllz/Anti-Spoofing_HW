import warnings
import csv
import os
import torch
import hydra
import numpy as np
from hydra.utils import instantiate
from pathlib import Path
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


class SubmissionInferencer(Inferencer):
    """
    Special Inferencer for generating predictions in submission format
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = False
    
    def generate_eval_predictions(self):
        """
        Generates predictions for eval dataset
        
        Returns:
            dict: dictionary {key: score} for all eval samples
        """
        predictions = {}
        
        # Load eval protocol to get correct trial IDs
        # Use environment variable for data path
        data_path = os.environ.get('DATA_PATH', 'data')
        # For Kaggle: /kaggle/input/asvpoof-2019-dataset/LA/
        if data_path == '/kaggle/input/asvspoof2019-la':
            data_path = '/kaggle/input/asvpoof-2019-dataset/LA'
        protocol_path = f"{data_path}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        eval_trial_ids = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    eval_trial_ids.append(parts[1])  # trial_id is the second column
        
        eval_dataloader = None
        for part, dataloader in self.evaluation_dataloaders.items():
            if part == "eval":
                eval_dataloader = dataloader
                break
        
        if eval_dataloader is None:
            print("Eval dataloader not found!")
            print("Available dataloaders:", list(self.evaluation_dataloaders.keys()))
            return predictions
        
        self.model.eval()
        print(f"Generating predictions for eval dataset...")
        print(f"   Number of batches: {len(eval_dataloader)}")
        print(f"   Expected trial IDs: {len(eval_trial_ids)}")
        
        trial_id_idx = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(eval_dataloader),
                desc="Generating predictions",
                total=len(eval_dataloader),
            ):
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)
                
                outputs = self.model(batch["data_object"])
                
                logits = outputs["logits"]
                # Convert logits to bonafide probabilities using softmax
                # For binary classification: softmax([logit_spoof, logit_bonafide]) = [prob_spoof, prob_bonafide]
                # We want prob_bonafide (second output)
                probs = torch.softmax(logits, dim=-1)
                bonafide_probs = probs[:, 1]  # Probability of bonafide class (second output)
                
                batch_size = logits.shape[0]
                
                # Use trial IDs from protocol
                if trial_id_idx + batch_size > len(eval_trial_ids):
                    print(f"WARNING: Not enough trial IDs! Expected {len(eval_trial_ids)}, got {trial_id_idx + batch_size}")
                    break
                    
                keys = eval_trial_ids[trial_id_idx:trial_id_idx + batch_size]
                trial_id_idx += batch_size
                
                for i, (key, prob) in enumerate(zip(keys, bonafide_probs)):
                    predictions[key] = prob.item()
        
        print(f"Generated {len(predictions)} predictions from {len(eval_trial_ids)} expected trial IDs")
        
        if len(predictions) != len(eval_trial_ids):
            print(f"WARNING: Mismatch! Generated {len(predictions)} predictions, expected {len(eval_trial_ids)}")
            print(f"This could cause incorrect EER calculation in grading.py!")
        
        return predictions


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for generating predictions and creating CSV file for submission
    """
    print("Starting prediction generation for submission...")
    
    set_random_seed(config.inferencer.seed)
    
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    print(f"Using device: {device}")

    print("Setting up dataloader...")
    dataloaders, batch_transforms = get_dataloaders(config, device)
    
    print(f"   Available dataloaders: {list(dataloaders.keys())}")

    print("Creating model...")
    from src.model.model import LCNN
    model = LCNN(
        in_channels=1,
        num_classes=2,  # 2 outputs for binary classification
        dropout_rate=0.75  # As in original paper
    ).to(device)
    print(f"   Model: {type(model).__name__}")

    best_model_path = "bestmodel/model_best.pth"
    print(f"Loading model from: {best_model_path}")
    
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded successfully")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Model loaded successfully (state_dict)")
            else:
                model.load_state_dict(checkpoint)
                print("Model loaded successfully (direct)")
            
            if 'best_eer' in checkpoint:
                print(f"   Best EER: {checkpoint['best_eer']:.4f}")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file not found: {best_model_path}")
        return

    print("Creating inferencer...")
    save_path = ROOT_PATH / "data" / "saved" / "submission"
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = SubmissionInferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=None,
        skip_model_load=True,
        writer=None,
    )

    print("Generating predictions...")
    predictions = inferencer.generate_eval_predictions()

    if not predictions:
        print("Failed to get predictions!")
        return

    print(f"Generated {len(predictions)} predictions")

    output_file = "students_solutions/aabagdasarian.csv"
    
    # Create directory if it doesn't exist
    os.makedirs("students_solutions", exist_ok=True)
    
    print(f"Saving predictions to {output_file}...")
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, score in predictions.items():
            writer.writerow([key, score])
    
    print(f"Predictions saved to {output_file}")
    
    scores = list(predictions.values())
    print(f"\nPrediction statistics:")
    print(f"   Total predictions: {len(scores)}")
    print(f"   Min score: {min(scores):.4f}")
    print(f"   Max score: {max(scores):.4f}")
    print(f"   Mean score: {np.mean(scores):.4f}")
    print(f"   Std score: {np.std(scores):.4f}")
    
    print(f"\nExample predictions:")
    for i, (key, score) in enumerate(list(predictions.items())[:5]):
        print(f"   {key}: {score:.4f}")
    
    print(f"\nDone! File {output_file} created for submission.")


if __name__ == "__main__":
    main() 