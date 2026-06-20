
from utils.training_logger import TrainingLogger
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
train_dataset = DynamicKeywordDataset(data_list, noise_dir=os.path.join(dataset_path,noise_path), is_training=True)





if __name__ == "__main__":
    # this potentially needs thinking, what batch size we will use ??
    train_loader = iter(DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4))


    #print(next(train_loader))
    print("tensor",next(train_loader)[0])
    print("label",next(train_loader[1]))
    print("tensor shape",next(train_loader)[0].shape)