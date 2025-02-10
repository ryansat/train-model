from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import load_dataset, Dataset  # Added Dataset import
import json

def train_model():
    # Load the locally saved model and tokenizer
    model_path = "./local_deepseek_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        
        # Load custom dataset
        with open("training_data/code_examples.json", "r") as f:
            raw_data = json.load(f)
        
        dataset = Dataset.from_dict({
            'code': [example['code'] for example in raw_data],
            'description': [example['description'] for example in raw_data]
        })
        
        def tokenize_function(examples):
            return tokenizer(
                examples["code"],
                truncation=True,
                padding="max_length",
                max_length=512
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        training_args = TrainingArguments(
            output_dir="./trained_model",
            per_device_train_batch_size=1,  # Reduce batch size for CPU
            gradient_accumulation_steps=8,  # Increase to simulate larger batch size
            num_train_epochs=3,
            save_steps=100,
            logging_steps=50,
            learning_rate=1e-5,
            warmup_steps=100,
            fp16=False,  # Disable mixed precision as it's not supported on CPU
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            ),
        )

        print("Starting training...")
        trainer.train()
        
        print("Saving final model...")
        trainer.save_model("./final_model")
        print("Training complete!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()