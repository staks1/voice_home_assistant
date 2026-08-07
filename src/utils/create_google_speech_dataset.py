import torchaudio
import random
import os
import torchaudio




dataset_path = "/home/st1/Documents/training_cnn_for_waveform/data/audio_chunks"

def get_background_speech_data(root_dir=os.path.join(dataset_path,"google_speech"), num_samples=3000, unknown_label_id=13):
    """
    Downloads the Google Speech Commands dataset and extracts a random 
    subset to act as the negative/background class.
    """
    print(f"Checking for Google Speech Commands dataset in {root_dir}...")
    os.makedirs(root_dir, exist_ok=True)
    
    # Get all .wav files from all subfolders
    all_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                all_paths.append(full_path)

    print(f"Found {len(all_paths)} total .wav files")

    # Safely sample without exceeding the available dataset size
    safe_num_samples = min(num_samples, len(all_paths))
    selected_paths = random.sample(all_paths, safe_num_samples)

    # Create background list with tuples (path, label)
    background_list = []
    for path in selected_paths:
        background_list.append((path, unknown_label_id))

    print(f"Successfully sampled {safe_num_samples} files from {root_dir}")
    print(f"All assigned to class {unknown_label_id}")
    print(f"Background list length: {len(background_list)}")

    # Example of first few items
    print("\nFirst 3 items:")
    for i, (path, label) in enumerate(background_list[:3]):
        print(f"  {i+1}. {path} -> Label: {label}")
    return background_list

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Test the download and extraction
    bg_data = get_background_speech_data(num_samples=100)
    print(f"Sample entry: {bg_data[0]}")