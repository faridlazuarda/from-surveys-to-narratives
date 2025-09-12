import os
import shutil
import tempfile
from huggingface_hub import HfApi, create_repo, upload_folder, whoami
from requests.exceptions import HTTPError

# Replace this with your actual Hugging Face token
HUGGINGFACE_TOKEN = ""

# Initialize the Hugging Face API
api = HfApi()

# Check if the token is valid and fetch user info
user_info = whoami(token=HUGGINGFACE_TOKEN)
print(f"Authenticated as {user_info['name']}")

# Base directory containing the language folders
# base_dir = "/tud/models/baseline/Meta-Llama-3.1-8B-Instruct"
# base_dir = "/tud/models/baseline/gemma-2-9b-it"
# base_dir = "/tud/models/baseline/Meta-Llama-3.1-8B"
base_dir = "/tud/models/baseline/Llama-2-7b-chat-hf"

# Model name for the repository
# model_name = "llama-3.1-8B-it"
# model_name = "gemma-2-9b-it"
model_name = base_dir.split("/")[-1].lower()
print(model_name)

# Get the list of language directories
language_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def get_latest_checkpoint(dir_path):
    """Get the latest checkpoint directory in the specified directory."""
    checkpoints = [d for d in os.listdir(dir_path) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(dir_path, latest_checkpoint)

for lang in language_dirs:
    # Handle comma-separated directories and split them
    if ',' in lang:
        individual_langs = lang.split(',')
        combined_repo_name = f"{user_info['name']}/valadapt-{model_name}-combined"
        
        latest_checkpoint_dir = get_latest_checkpoint(os.path.join(base_dir, lang))
        if not latest_checkpoint_dir:
            print(f"No checkpoints found in {os.path.join(base_dir, lang)}")
            continue

        # Create a temporary directory for the repository
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy only the files from the latest checkpoint directory to the temporary directory
            for item in os.listdir(latest_checkpoint_dir):
                s = os.path.join(latest_checkpoint_dir, item)
                d = os.path.join(tmp_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, False, None)
                else:
                    shutil.copy2(s, d)

            # Ensure the repository exists
            try:
                api.repo_info(repo_id=combined_repo_name, token=HUGGINGFACE_TOKEN)
                print(f"Repository {combined_repo_name} already exists.")
            except HTTPError:
                print(f"Creating repository {combined_repo_name}.")
                create_repo(repo_id=combined_repo_name, token=HUGGINGFACE_TOKEN, private=False)

            # Upload files to Hugging Face Hub
            upload_folder(
                repo_id=combined_repo_name,
                folder_path=tmp_dir,
                path_in_repo="",
                commit_message="Replace with latest checkpoint files",
                token=HUGGINGFACE_TOKEN
            )
    else:
        individual_langs = [lang]
    
    for individual_lang in individual_langs:
        lang_dir = os.path.join(base_dir, lang)
        latest_checkpoint_dir = get_latest_checkpoint(lang_dir)
        
        if not latest_checkpoint_dir:
            print(f"No checkpoints found in {lang_dir}")
            continue

        # Create a valid repository name
        repo_name = f"{user_info['name']}/valadapt-{model_name}-{individual_lang}".replace(',', '-')
        
        # Create a temporary directory for the repository
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy only the files from the latest checkpoint directory to the temporary directory
            for item in os.listdir(latest_checkpoint_dir):
                s = os.path.join(latest_checkpoint_dir, item)
                d = os.path.join(tmp_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, False, None)
                else:
                    shutil.copy2(s, d)

            # Ensure the repository exists
            try:
                api.repo_info(repo_id=repo_name, token=HUGGINGFACE_TOKEN)
                print(f"Repository {repo_name} already exists.")
            except HTTPError:
                print(f"Creating repository {repo_name}.")
                create_repo(repo_id=repo_name, token=HUGGINGFACE_TOKEN, private=False)

            # Upload files to Hugging Face Hub
            upload_folder(
                repo_id=repo_name,
                folder_path=tmp_dir,
                path_in_repo="",
                commit_message="Replace with latest checkpoint files",
                token=HUGGINGFACE_TOKEN
            )

print("Latest checkpoint files have been uploaded and replaced in the existing repositories.")
