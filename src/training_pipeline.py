
from utils.training_logger import TrainingLogger
from utils.create_datasets import prepare_dataloaders
from dynamic_dataset_class import DynamicKeywordDataset
from torch.utils.data import DataLoader
import os 

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
    batch_size=4, 
    val_split=0.1,
    plot_distribution_per_class=False,  # Enable plotting
    pin_memory=False
)
    # test train loader 
    print("==== train ====\n")
    train_loader = iter(train_loader)
    waveform, label = next(train_loader)
    print("tensor", waveform)
    print("label", label)
    print("tensor shape", waveform.shape)

    print("==== val ====\n")
    # test val loader 
    val_loader = iter(val_loader)
    waveform, label = next(val_loader)
    print("tensor", waveform)
    print("label", label)
    print("tensor shape", waveform.shape)