"""Download a Hugging Face model repo to a local directory.

Usage: python download_model.py <hf_repo> <local_dir>

Uses huggingface_hub.snapshot_download, which works across huggingface_hub versions
(the `huggingface-cli` command no longer works in huggingface_hub >= 1.0, and `hf`
does not exist in older versions).
"""
import os
import sys

from huggingface_hub import snapshot_download

if len(sys.argv) != 3:
    sys.exit("Usage: python download_model.py <hf_repo> <local_dir>")
repo, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=local_dir)
if not os.path.isfile(os.path.join(local_dir, "config.json")):
    sys.exit(f"Download of {repo} finished but {local_dir}/config.json is missing")
print(f"Downloaded {repo} -> {local_dir}")
