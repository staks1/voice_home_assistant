
from utils.create_google_speech_dataset import get_background_speech_data
from utils.training_logger import TrainingLogger
from utils.create_datasets import prepare_dataloaders, prepare_dataloaders_with_google_speech
from utils.gpu_utils import select_gpu
from dynamic_dataset_class import DynamicKeywordDataset
from torch.utils.data import DataLoader
import os
import sys
import torch
from training_loop import train_model
from models import MobileNetV2Custom


# ==========================================
# Initialization
# ==========================================

dataset_path = "/home/st1/Documents/training_cnn_for_waveform/data/audio_chunks"
noise_path = "noise"
path_to_logs = "/home/st1/Documents/training_cnn_for_waveform/logs"
models_path = "/home/st1/Documents/training_cnn_for_waveform/models"
added_data = "/home/st1/Documents/training_cnn_for_waveform/data/google_speech_root"


# initialize class map to integers (sorted for deterministic mapping)
# Build class mapping, excluding 'google_speech' directory if it exists
all_classes = sorted(os.listdir(dataset_path))
local_classes = [c for c in all_classes if c != 'google_speech']
class_mapping = {x:i for i,x in enumerate(local_classes)}
class_mapping['google_speech'] = 13  # Google speech is always class 13

print(f"Class mapping: {class_mapping}")

# Save class mapping to file
mapping_file = os.path.join(path_to_logs, 'class_mapping.txt')
os.makedirs(path_to_logs, exist_ok=True)
with open(mapping_file, 'w') as f:
    for class_name, class_id in sorted(class_mapping.items()):
        f.write(f"{class_name}: {class_id}\n")
print(f"Class mapping saved to {mapping_file}")

# initialize Logger 
train_logger = TrainingLogger(path_to_logs)


# feed the google speech 3000 samples 
google_speech_samples = get_background_speech_data(root_dir=os.path.join(added_data,"google_speech"), num_samples=3000, unknown_label_id=13)

data_list = [ (os.path.join(dataset_path,x,file),class_mapping.get(x)) for x in local_classes 
                        for file in os.listdir(os.path.join(dataset_path, x))]

total_list = data_list + google_speech_samples
print(f"Total dataset size after adding Google Speech samples: {len(total_list)}")

# lets print 3 sample from the total list to verify
print(f"Sample from total_list: {total_list[0:3]}")
print(f"Sample from the new ones {total_list[-1]}")


if __name__ == "__main__":



    train_loader, val_loader= prepare_dataloaders_with_google_speech(full_data_list=total_list, noise_dir=os.path.join(dataset_path, noise_path), batch_size=32, val_split=0.2,pin_memory=False)
    
    device = select_gpu()

    # 5. CRITICAL: Update the model initialization to expect 14 classes instead of 13
    model_phase1 = MobileNetV2Custom(num_classes=14, freeze_base=True).to(device)

    trained_model = train_model(model_phase1, train_loader, val_loader,lr=1e-3, num_epochs=10,step_size=5,device = device)

    # Phase 2 : train full model 
    #--------------------------------#
    print("\n========================================")
    print("PHASE 2: Fine-Tuning the Entire Network")
    print("========================================")
    # Re-instantiate the model, but this time UNFREEZE the base
    model_phase2 = MobileNetV2Custom(num_classes=14, freeze_base=False)
    
    
    # Load the aligned classifier weights you just achieved in Phase 1
    if os.path.exists('/home/st1/Documents/training_cnn_for_waveform/models/best_keyword_model.pth'):
        # CRITICAL FIX: Explicitly map the tensors directly to cuda:3 to prevent CPU RAM spikes
        # Set weights_only=True to comply with secure deserialization standards
        state_dict = torch.load('/home/st1/Documents/training_cnn_for_waveform/models/best_keyword_model.pth', map_location=device, weights_only=True)
        model_phase2.load_state_dict(state_dict)
        print("Successfully loaded Phase 1 weights.")
    else:
        print("Warning: Phase 1 weights not found.")
        
    # Train Phase 2 with a microscopic LR to protect the ImageNet filters
    final_finetuned_model = train_model(
        model=model_phase2, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        lr=1e-5,
        num_epochs=60, # We can train longer here because the tiny LR prevents optimizer thrashing
        step_size = 35,
        device=device, 
        save_path=os.path.join(models_path,'phase2_finetuned_best.pth'))
    