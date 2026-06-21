# what gpus to use 
import torch 
import subprocess


def select_gpu():
    print(torch.cuda.device_count())
    avail_gpus = []
    for i in range(torch.cuda.device_count()):
        avail_gpus.append("cuda:"+str(i))
    if avail_gpus :
        print(f"Selected 1st gpu : {avail_gpus[0]}")
        return avail_gpus[0]
    return "cpu"







def run_gpu_monitor(script_path='/home/st1/Documents/nvidia-gpu-server-scripts/get_gpu_util_json_v4.sh'):
    """
    Run GPU monitoring script in a subprocess (non-blocking)
    """
    try:
        # Run the script and capture output
        result = subprocess.run(
            ['bash', script_path], 
            capture_output=False, 
            text=True, 
            timeout=1   # 5 second timeout to avoid blocking training
        )
    except subprocess.TimeoutExpired:
        print("⚠ GPU monitor script timed out")
    except FileNotFoundError:
        print(f"⚠ Script not found: {script_path}")
    except Exception as e:
        print(f"⚠ Error running GPU monitor: {e}")