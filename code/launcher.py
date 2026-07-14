# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:14:13 2026

@author: ramigon
"""

import os
import subprocess
import time
import sys

experiments = [
    "C:/Users/ramigon/Downloads/8INF976/code/Cifar_hybrid.py",
    "C:/Users/ramigon/Downloads/8INF976/code/Cifar _SA.py",
    "C:/Users/ramigon/Downloads/8INF976/code/Cifar _abc.py",
    
    "C:/Users/ramigon/Downloads/8INF976/code/benchmark_technique.py",
   # "C:/Users/ramigon/Downloads/8INF976/code/test_bank_eva.py",
    "C:/Users/ramigon/Downloads/8INF976/code/random_search.py",
]

print("==================================================")
print("STARTING ALL EXPERIMENTS AUTOMATICALLY")
print("==================================================")

total_start = time.time()

for script in experiments:
    time.sleep(5)
    if not os.path.exists(script):
        print(f"\n[WARNING] {script} not found, skipping.")
        continue
        
    print(f"\n{'='*40}")
    print(f"RUNNING: {script}")
    print(f"{'='*40}\n")
    
    start_time = time.time()
    try:
        subprocess.run([sys.executable,"-u", script], check=True)
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {script} finished in {elapsed/60:.2f} minutes.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script} failed with exit code {e.returncode}")

total_elapsed = time.time() - total_start
print("\n==================================================")
print(f"ALL EXPERIMENTS COMPLETED IN {total_elapsed/3600:.2f} HOURS")
print("==================================================")