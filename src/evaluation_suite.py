import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import os
import random

# Import your custom modules
from models import MobileNetV2Custom
from dynamic_dataset_class import DynamicKeywordDataset
from utils.create_google_speech_dataset import get_background_speech_data
from models import MobileNetV2Custom



dataset_path = "/home/st1/Documents/training_cnn_for_waveform/data/audio_chunks"
# initialize class map to integers (sorted for deterministic mapping)
# Build class mapping, excluding 'google_speech' directory if it exists
all_classes = sorted(os.listdir(dataset_path))
local_classes = [c for c in all_classes if c != 'google_speech']
class_mapping = {x:i for i,x in enumerate(local_classes)}
class_mapping['google_speech'] = 13  # Google speech is always class 13

print(f"Class mapping: {class_mapping}")


def evaluate_model(model, test_loader, device, class_names, output_dir="../evaluation_results"):
    """
    Executes a high-speed, batch-wise forward pass over the entire test dataset,
    collects all predictions, and generates robust mathematical metrics.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Set model to evaluation mode (Freezes BatchNorm stats, disables Dropout)
    model.eval()
    
    all_true_labels = []
    all_pred_labels = []
    
    print("\nStarting batch inference on the test dataset...")
    
    # 2. Disable gradient tracking to massively reduce VRAM footprint and speed up inference
    with torch.no_grad():
        for batch_idx, (mels, labels) in enumerate(test_loader):
            # Move data to the target GPU (e.g., cuda:3)
            mels = mels.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(mels)
            
            # Extract the class with the highest probability score
            _, predicted = torch.max(logits, dim=1)
            print(f"Predicted index is {predicted}")
            
            # Move results back to CPU RAM to safely store them for Scikit-Learn
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    print("Inference complete. Calculating metrics...\n")

    # 3. Compute Confusion Matrix
    # This matrix mathematically proves where the model gets "confused" between specific classes
    cm = confusion_matrix(all_true_labels, all_pred_labels,labels=range(len(class_names)))
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        cm, 
        annot=True,       # Show the raw numbers inside the squares
        fmt='d',          # Format as integers
        cmap='Blues',     # Visual gradient scale
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Keyword Spotting Confusion Matrix', pad=20, fontsize=16)
    plt.ylabel('True Ground-Truth Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    
    # Rotate labels so they don't overlap if your Greek words are long
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300)
    print(f"Saved Confusion Matrix visualization to: {cm_plot_path}")
    
    # 4. Generate Classification Report (Precision, Recall, F1-Score)
    # Target names ensures the output text uses your actual Greek words, not just 0-13 integers
    report = classification_report(
        all_true_labels, 
        all_pred_labels, 
        labels=range(len(class_names)),
        target_names=class_names,  
        zero_division=0 # Prevents crash if a class was never predicted
    )
    
    print("======================================================")
    print("CLASSIFICATION METRICS")
    print("======================================================")
    print(report)
    
    # Save the text report to disk for record-keeping
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Keyword Spotting Classification Report\n")
        f.write("======================================\n\n")
        f.write(report)
        
    print(f"Saved textual metrics report to: {report_path}")

if __name__ == "__main__":
    

    # Configuration
    models_path = "/home/st1/Documents/training_cnn_for_waveform/models"
    DEVICE = 'cpu'
    MODEL_PATH = 'phase2_finetuned_best.pth'
    BATCH_SIZE = 16 # High batch size is perfectly safe here because we aren't updating gradients
    # class mappings ;
    CLASS_NAMES = [
        "anoikse_fos_kanape",   # 0
        "anoikse_fos_krevati",  # 1
        "anoikse_fos_saloni",   # 2
        "kleise_fos_kanape",    # 3
        "kleise_fos_krevati",   # 4
        "kleise_fos_saloni",    # 5
        "kleise_musiki",        # 6
        "kleise_ola",           # 7
        "noise",                # 8
        "paikse_musiki",        # 9
        "xamilose_fos_kanape",  # 10
        "xamilose_fos_krevati", # 11
        "xamilose_fos_saloni",  # 12
        "google_speech"         # 13
    ]
    
    # Define your actual 14 classes here (0 to 13)
    # Replace these with your exact Greek words
    data_list = [ (os.path.join(dataset_path,x,file),class_mapping.get(x)) for x in local_classes 
                        for file in os.listdir(os.path.join(dataset_path, x))]
    
    full_test_list = data_list
    
    if len(full_test_list) > 0:
        # 2. Instantiate the Test Dataset
        # CRITICAL: is_training=False ensures NO augmentations (pitch shift, noise) are applied.
        # We must evaluate the model strictly on clean, pristine reality.
        test_dataset = DynamicKeywordDataset(
            data_list=full_test_list, 
            noise_dir=None, # Not needed for testing
            is_training=False 
        )
        
        # 3. Create DataLoader
        # shuffle=False because order doesn't matter for evaluation and it keeps it deterministic
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=False # Speeds up RAM to VRAM transfer
        )
        
        # 4. Initialize and Load Model
        print(f"Loading weights from {MODEL_PATH}...")
        model = MobileNetV2Custom(num_classes=14, freeze_base=False)
        
        try:
            # Using weights_only=False here to bypass the strict unpickler bug you encountered earlier
            state_dict = torch.load(os.path.join(models_path,MODEL_PATH), map_location=DEVICE, weights_only=False)
            model.load_state_dict(state_dict)
            model = model.to(DEVICE)
            
            # 5. Trigger Evaluation
            evaluate_model(model, test_loader, DEVICE, CLASS_NAMES)
            
        except Exception as e:
            print(f"Failed to load model weights: {e}")
    else:
        print("No test data found. Please populate 'full_test_list' with valid (path, label) tuples.")