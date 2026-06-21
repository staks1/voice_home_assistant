
from utils.training_logger import TrainingLogger
from utils.create_datasets import prepare_dataloaders
from utils.gpu_utils import select_gpu
from dynamic_dataset_class import DynamicKeywordDataset
from torch.utils.data import DataLoader
import os
import sys
from training_loop import train_model
from models import KeywordSpottingCNN


# ==========================================
# Initialization
# ==========================================

dataset_path = "/home/st1/Documents/training_cnn_for_waveform/data/audio_chunks"
noise_path = "noise"
path_to_logs = "/home/st1/Documents/training_cnn_for_waveform/logs"



# initialize class map to integers 
class_mapping = {x:i for i,x in enumerate(os.listdir(dataset_path))}


# initialize Logger 
train_logger = TrainingLogger(path_to_logs)


#data_list = [("path/to/yes_01.wav", 0), ("path/to/no_01.wav", 1), ...]
data_list = [ (os.path.join(dataset_path,x,file),class_mapping.get(x)) for x in os.listdir(dataset_path) 
                        for file in os.listdir(os.path.join(dataset_path, x))]





if __name__ == "__main__":
    # this potentially needs thinking, what batch size we will use ??
    #prepare_dataloaders(full_data_list, noise_dir, class_mapping, batch_size=64, val_split=0.1, plot_distribution_per_class=False)
    train_loader, val_loader = prepare_dataloaders(
    full_data_list=data_list, 
    noise_dir=os.path.join(dataset_path, noise_path),
    class_mapping=class_mapping,  # Pass your class_mapping here
    batch_size=64, 
    val_split=0.1,
    plot_distribution_per_class=False,  # Enable plotting
    pin_memory=False
)
    
    device = select_gpu()
    model = KeywordSpottingCNN().to(device)
    

    trained_model = train_model(model, train_loader, val_loader, num_epochs=200, device = device)