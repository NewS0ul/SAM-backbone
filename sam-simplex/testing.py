import os
from config import config
from tqdm import tqdm
import itertools

print(1)
num_files_to_process = config.n_runs
with os.scandir(config.runs_path) as entries:
    for entry in tqdm(itertools.islice(entries,10)):
        if entry.is_dir():
            run_directory_path = entry.path
            print(run_directory_path)