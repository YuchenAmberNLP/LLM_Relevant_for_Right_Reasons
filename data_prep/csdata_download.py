from huggingface_hub import hf_hub_download
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

base_repo = "rlitschk/csclir"
repo_type = "dataset"
local_base_dir = "data/csclir"

datasets = {
    "cs_0.5/xxxx": ["train.jsonl"],
    "cs_0.5/enxx": ["train.jsonl"],
    "finetuning/dede": ["train.jsonl"],
    "finetuning/ruru": ["train.jsonl"],
    "finetuning/arru": ["train.jsonl"],
    "finetuning/enen": ["train.jsonl"],
    "finetuning/arar": ["train.jsonl"],
    "finetuning/itit": ["train.jsonl"],
    "finetuning/ende": ["train.jsonl"],
    "finetuning/enit": ["train.jsonl"],
    "finetuning/enar": ["train.jsonl"],
    "finetuning/deit": ["train.jsonl"],
    "finetuning/deru": ["train.jsonl"],
    "finetuning/arit": ["train.jsonl"],

}


download_tasks = [
    (subdir, file)
    for subdir, files in datasets.items()
    for file in files
]

def download_file(subdir, file):
    try:
        print(f"Downloading: {subdir}/{file}")
        cached_file_path = hf_hub_download(
            repo_id=base_repo,
            filename=f"{subdir}/{file}",
            repo_type=repo_type
        )

        local_target_dir = os.path.join(local_base_dir, subdir)
        os.makedirs(local_target_dir, exist_ok=True)

        local_file_path = os.path.join(local_target_dir, file)
        shutil.copy(cached_file_path, local_file_path)

        return f"Saved: {local_file_path}"
    except Exception as e:
        return f"Failed: {subdir}/{file} ({e})"


max_workers = 5
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_task = {
        executor.submit(download_file, subdir, file): (subdir, file)
        for subdir, file in download_tasks
    }

    for future in as_completed(future_to_task):
        result = future.result()
        print(result)

print("All downloads finished.")