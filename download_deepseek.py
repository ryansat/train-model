from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import HfFolder
import os

def download_model():
    # Initialize model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            # Only use these options if accelerate is installed
            low_cpu_mem_usage=False,  # Changed from True
            device_map=None  # Changed from "auto"
        )
        
        # Save the model and tokenizer locally
        print("Saving model and tokenizer locally...")
        model.save_pretrained("./local_deepseek_model")
        tokenizer.save_pretrained("./local_deepseek_model")
        
        print("Download complete!")
        
    except Exception as e:
        print(f"Error occurred during download: {str(e)}")
        raise

if __name__ == "__main__":
    download_model()