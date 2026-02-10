import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Load environment variables
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables. Check your .env file.")

# Configuration
model_path = "./final_model_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf"
repo_name = "PhillipGre/llama2-13b-sctt-regression" 
commit_message = "Upload fine-tuned Llama-2-13b for SCTT creativity scoring (Pearson r=0.73)"

print("Uploading model to Hugging Face Hub...")
print(f"Model path: {model_path}")
print(f"Repository: {repo_name}")
print(f"Token loaded: {'Yes' if hf_token else 'No'}")

# Initialize API
api = HfApi(token=hf_token)

# Create repository if it doesn't exist
print("\nCreating/verifying repository...")
try:
    create_repo(
        repo_id=repo_name,
        token=hf_token,
        private=False,  # Set to True if you want a private repo
        exist_ok=True
    )
    print(f"✓ Repository created/verified: {repo_name}")
except Exception as e:
    print(f"Error creating repository: {e}")
    raise

# Upload the model
print("\nUploading model files (this may take a while for 25GB)...")
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_name,
    commit_message=commit_message
)

print(f"\n✓ Model uploaded successfully!")
print(f"View your model at: https://huggingface.co/{repo_name}")
