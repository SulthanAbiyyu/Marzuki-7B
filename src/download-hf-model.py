import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--local_dir_use_symlinks", type=bool, default=False)
    args = parser.parse_args()
    
    model_id= args.model_id
    local_dir= args.local_dir
    revision= args.revision
    local_dir_use_symlinks= args.local_dir_use_symlinks
    
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks, 
        revision=revision
    )