import os 

class_freq = {}

def calculate_class_freq():
    all_chunks = "/home/st1/Documents/training_cnn_for_waveform/data/audio_chunks"
    for x in os.listdir(all_chunks):
        class_freq[x] = len(os.listdir(os.path.join(all_chunks,x)))
    


if __name__=="__main__":
    calculate_class_freq()
    print(class_freq)