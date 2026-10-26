


import os
import subprocess
import argparse
import xml.etree.ElementTree as ET

import torch
import time

NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
class gpu_waiter:


    def __init__(self, max_claimed_gpus: int = 1):

        self.claimed = dict()
        self.max_claimed_gpus = max_claimed_gpus

        




    def num_claimed(self):
        return sum([1 for holder in self.claimed.values() if holder is not None])

    def wait_loop(self):
        while True:
            free_gpus = get_free_gpus()
            if len(free_gpus) > 0:
                print(f"Free GPUs: {free_gpus}")
            
            # ask to claim free gpus            
            for gpu_id in free_gpus:
                if self.num_claimed() < self.max_claimed_gpus:
                    self.claim_gpu(gpu_id)
                    print(f"Claimed GPU {gpu_id}")

            # if there are claimed gpus, print them and sleep for 10 seconds
            if len(self.claimed) > 0:
                print("claimed gpus:", [gpu_id for gpu_id, holder in self.claimed.items() if holder is not None])
            

            time.sleep(10)



    def claim_gpu(self, gpu_id):
        self.claimed[gpu_id] = torch.rand(1000, 1000, device=f"cuda:{gpu_id}")  # Simulate claiming the GPU by allocating a tensor on it

    def release_gpu(self, gpu_id):
        self.claimed[gpu_id] = None


def get_free_gpus() -> list[int]:
    """
    Queries nvidia-smi via its built-in XML output interface and 
    returns a list of GPU indices that have exactly 0 running processes.
    """
    try:
        # Query nvidia-smi using XML format to reliably extract process trees
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"], 
            capture_output=True, 
            text=True, 
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: nvidia-smi is not accessible. Ensure NVIDIA drivers are installed.")
        return []

    root = ET.fromstring(result.stdout)
    free_gpus = []

    # Iterate cleanly over every physical GPU found in the system XML tree
    for index, gpu in enumerate(root.findall("gpu")):
        processes = gpu.find("processes")
        
        # Count total running process elements (both Compute and Graphics context)
        process_count = 0
        if processes is not None:
            # Safely captures both <process_info> tags
            process_count = len(processes.findall("process_info"))
        
        # If the process container is empty, the GPU is completely free
        if process_count == 0:
            free_gpus.append(index)

    return free_gpus

if __name__ == "__main__":
    # Execute and capture the array

    wait_loop = gpu_waiter()
    wait_loop.wait_loop()
