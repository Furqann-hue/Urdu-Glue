"""
Fine-tune XLM-RoBERTa-Large model with QLoRA (Quantized LoRA) technique on STS-B (Semantic Textual Similarity Benchmark) dataset
QLoRA uses 4-bit quantization for memory-efficient fine-tuning
Regression task - Predict similarity score (0-5) between sentence pairs
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# Try to import BitsAndBytesConfig - may not be available on all systems
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. QLoRA requires bitsandbytes for 4-bit quantization.")
    print("On macOS, bitsandbytes may not work. Consider using regular LoRA instead.")
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from scipy.stats import pearsonr, spearmanr

# Configuration
MODEL_NAME = "xlm-roberta-large"  # XLM-RoBERTa-Large model
MAX_LENGTH = 256  # Longer for sentence pairs
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./roberta_qlora_stsb"
DATA_DIR = "Documents/research"
STS_B_DIR = os.path.join(DATA_DIR, "sts-b-dataset")
TRAIN_FILE = os.path.join(STS_B_DIR, "Final_STSB_train.csv")
DEV_FILE = os.path.join(STS_B_DIR, "Final_dev_translated.csv")
TEST_FILE = os.path.join(STS_B_DIR, "STSB-test_urdu-translated - Final.tsv")
NUM_LABELS = 1  # Regression task - single continuous value

# QLoRA Configuration - 4-bit quantization
# Only create config if bitsandbytes is available
if BITSANDBYTES_AVAILABLE:
    BITSANDBYTES_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_quant_type="nf4",  # Normalized Float 4-bit
        bnb_4bit_compute_dtype=torch.float16,  # Compute dtype
        bnb_4bit_use_double_quant=True,  # Double quantization for better accuracy
    )
else:
    BITSANDBYTES_CONFIG = None

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification (works for regression too)
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)


class STSBDataset(Dataset):
    """Dataset class for STS-B (Semantic Textual Similarity) task"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 256,
    ):
        """
        Initialize the dataset
        
        Args:
            data: List of dictionaries with 'sentence1', 'sentence2', and 'score' keys
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for the pair
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get sentence pair and score
        sentence1 = str(example.get("sentence1", ""))
        sentence2 = str(example.get("sentence2", ""))
        score = float(example.get("score", 0.0))
        
        # Clean text
        if not sentence1 or sentence1 == "nan":
            sentence1 = ""
        if not sentence2 or sentence2 == "nan":
            sentence2 = ""
        
        # Ensure score is in valid range [0, 5]
        score = max(0.0, min(5.0, score))
        
        # Tokenize sentence pair
        encoding = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).flatten(),
            "labels": torch.tensor(score, dtype=torch.float32),
        }


def load_stsb_data(train_split: float = 0.8, random_state: int = 42) -> Dict:
    """
    Load STS-B dataset and create sentence pairs with similarity scores
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, and test datasets
    """
    print("Loading STS-B (Semantic Textual Similarity Benchmark) dataset...")
    print(f"Data directory: {STS_B_DIR}")
    
    datasets = {}
    
    # Load training data
    if os.path.exists(TRAIN_FILE):
        print(f"Loading training data from {TRAIN_FILE}...")
        train_df = pd.read_csv(TRAIN_FILE)
        
        # Check column names
        print(f"Training file columns: {train_df.columns.tolist()}")
        print(f"Training file shape: {train_df.shape}")
        print(f"First few rows:\n{train_df.head(3)}")
        
        # Find score column (could be 'score' or 'scores')
        score_col = None
        for col in ['score', 'scores', 'Score', 'Scores']:
            if col in train_df.columns:
                score_col = col
                break
        
        if score_col is None:
            raise ValueError(f"Could not find score column in {TRAIN_FILE}. Available columns: {train_df.columns.tolist()}")
        
        # Find sentence columns
        sent1_col = None
        sent2_col = None
        for col in train_df.columns:
            col_lower = str(col).lower()
            if 'sentence1' in col_lower or 'sent1' in col_lower:
                sent1_col = col
            elif 'sentence2' in col_lower or 'sent2' in col_lower:
                sent2_col = col
        
        if sent1_col is None or sent2_col is None:
            raise ValueError(f"Could not find sentence columns in {TRAIN_FILE}. Available columns: {train_df.columns.tolist()}")
        
        print(f"Using columns: sentence1='{sent1_col}', sentence2='{sent2_col}', score='{score_col}'")
        
        # Create data list
        train_data = []
        for _, row in train_df.iterrows():
            sent1 = str(row[sent1_col]) if pd.notna(row[sent1_col]) else ""
            sent2 = str(row[sent2_col]) if pd.notna(row[sent2_col]) else ""
            score = float(row[score_col]) if pd.notna(row[score_col]) else 0.0
            
            if sent1 and sent2:
                train_data.append({
                    "sentence1": sent1,
                    "sentence2": sent2,
                    "score": score,
                })
        
        print(f"✓ Loaded {len(train_data)} training examples")
        
        # Split into 80% train and 20% validation
        train_list, val_list = train_test_split(
            train_data,
            test_size=1 - train_split,
            random_state=random_state,
            shuffle=True,
        )
        
        datasets['train'] = train_list
        datasets['val'] = val_list
        print(f"✓ Train set (80%): {len(train_list)} pairs")
        print(f"✓ Validation set (20%): {len(val_list)} pairs")
        
        # Check score distribution
        scores = [d["score"] for d in train_data]
        print(f"✓ Score distribution:")
        print(f"   Min: {min(scores):.2f}, Max: {max(scores):.2f}, Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")
    else:
        raise FileNotFoundError(f"Could not find training file: {TRAIN_FILE}")
    
    # Load dev data (if available, can be used as additional validation)
    if os.path.exists(DEV_FILE):
        print(f"\nLoading dev data from {DEV_FILE}...")
        dev_df = pd.read_csv(DEV_FILE)
        
        print(f"Dev file columns: {dev_df.columns.tolist()}")
        print(f"Dev file shape: {dev_df.shape}")
        
        # Find score column
        score_col = None
        for col in ['score', 'scores', 'Score', 'Scores']:
            if col in dev_df.columns:
                score_col = col
                break
        
        # Find sentence columns
        sent1_col = None
        sent2_col = None
        for col in dev_df.columns:
            col_lower = str(col).lower()
            if 'sentence1' in col_lower or 'sent1' in col_lower:
                sent1_col = col
            elif 'sentence2' in col_lower or 'sent2' in col_lower:
                sent2_col = col
        
        if sent1_col and sent2_col and score_col:
            dev_data = []
            for _, row in dev_df.iterrows():
                sent1 = str(row[sent1_col]) if pd.notna(row[sent1_col]) else ""
                sent2 = str(row[sent2_col]) if pd.notna(row[sent2_col]) else ""
                score = float(row[score_col]) if pd.notna(row[score_col]) else 0.0
                
                if sent1 and sent2:
                    dev_data.append({
                        "sentence1": sent1,
                        "sentence2": sent2,
                        "score": score,
                    })
            
            if dev_data:
                datasets['dev'] = dev_data
                print(f"✓ Loaded {len(dev_data)} dev examples")
    
    # Load test data (if available, may not have scores)
    if os.path.exists(TEST_FILE):
        try:
            print(f"\nLoading test data from {TEST_FILE}...")
            # Try with tab separator first, then fallback to other options
            try:
                test_df = pd.read_csv(TEST_FILE, sep='\t', on_bad_lines='skip', engine='python')
            except Exception:
                # Try with comma separator
                try:
                    test_df = pd.read_csv(TEST_FILE, sep=',', on_bad_lines='skip', engine='python')
                except Exception:
                    # Try auto-detection
                    test_df = pd.read_csv(TEST_FILE, on_bad_lines='skip', engine='python')
            
            print(f"Test file columns: {test_df.columns.tolist()}")
            print(f"Test file shape: {test_df.shape}")
            
            # Find Urdu sentence columns
            sent1_col = None
            sent2_col = None
            for col in test_df.columns:
                col_lower = str(col).lower()
                if 'sentence1' in col_lower and 'urdu' in col_lower:
                    sent1_col = col
                elif 'sentence2' in col_lower and 'urdu' in col_lower:
                    sent2_col = col
            
            # If Urdu columns not found, try regular sentence columns
            if not sent1_col or not sent2_col:
                for col in test_df.columns:
                    col_lower = str(col).lower()
                    if 'sentence1' in col_lower and sent1_col is None:
                        sent1_col = col
                    elif 'sentence2' in col_lower and sent2_col is None:
                        sent2_col = col
            
            if sent1_col and sent2_col:
                test_data = []
                for _, row in test_df.iterrows():
                    sent1 = str(row[sent1_col]) if pd.notna(row[sent1_col]) else ""
                    sent2 = str(row[sent2_col]) if pd.notna(row[sent2_col]) else ""
                    
                    if sent1 and sent2:
                        test_data.append({
                            "sentence1": sent1,
                            "sentence2": sent2,
                            "score": 0.0,  # Placeholder, no scores in test set
                        })
                
                if test_data:
                    datasets['test'] = test_data
                    print(f"✓ Loaded {len(test_data)} test examples")
            else:
                print(f"⚠️  Warning: Could not find sentence columns in test file. Skipping test data.")
        except Exception as e:
            print(f"⚠️  Warning: Could not load test file: {e}")
            print(f"   Skipping test data. Training will continue with train/val data only.")
    
    return datasets


def compute_metrics(eval_pred):
    """Compute metrics for regression task"""
    predictions, labels = eval_pred
    
    # Flatten predictions if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(labels, predictions)
    if np.isnan(pearson_corr):
        pearson_corr = 0.0
    
    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(labels, predictions)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
    
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }


def setup_qlora_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
    """
    Setup XLM-RoBERTa-Large model with QLoRA (Quantized LoRA) adapters
    
    Args:
        model_name: Name of the base model
        num_labels: Number of classification labels (1 for regression)
        
    Returns:
        Model with QLoRA adapters, tokenizer, and a flag indicating if quantization was used
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if bitsandbytes is available
    if not BITSANDBYTES_AVAILABLE or BITSANDBYTES_CONFIG is None:
        print("⚠️  bitsandbytes not available. Falling back to regular LoRA (without quantization).")
        print("   For true QLoRA, install bitsandbytes: pip install bitsandbytes")
        print("   Note: bitsandbytes requires CUDA and may not work on macOS.")
        
        # Load model without quantization (regular LoRA)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="regression",  # Regression task
        )
        
        # Apply LoRA directly (no quantization)
        print("Applying LoRA adapters...")
        model = get_peft_model(model, LORA_CONFIG)
        return model, tokenizer, False  # False = no quantization used
    else:
        # Check if CUDA is available - quantization only makes sense on GPU
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available. Quantization is primarily for GPU memory savings.")
            print("   Falling back to regular LoRA on CPU (quantization not beneficial on CPU).")
            
            # Load model without quantization (regular LoRA)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="regression",  # Regression task
            )
            
            # Apply LoRA directly (no quantization)
            print("Applying LoRA adapters...")
            model = get_peft_model(model, LORA_CONFIG)
            return model, tokenizer, False  # False = no quantization used
        else:
            print(f"Loading model with 4-bit quantization: {model_name}")
            
            # Load base model with 4-bit quantization (GPU only)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="regression",  # Regression task
                quantization_config=BITSANDBYTES_CONFIG,  # Apply 4-bit quantization
                device_map="auto",  # Automatically handle device placement on GPU
            )
            
            # Prepare model for k-bit training (required for QLoRA)
            print("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
            
            # Apply LoRA
            print("Applying LoRA adapters...")
            model = get_peft_model(model, LORA_CONFIG)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer, True  # True = quantization used


def main():
    """Main training function"""
    print("=" * 60)
    print("XLM-RoBERTa-Large QLoRA Fine-tuning for STS-B (Semantic Textual Similarity Benchmark)")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_stsb_data()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    # Setup model and tokenizer
    model, tokenizer, use_quantization = setup_qlora_model(num_labels=NUM_LABELS)
    
    # Ensure model is on the correct device
    # If device_map wasn't used (CPU case), model should already be moved in setup_qlora_model
    # But we double-check here for safety - only move if device_map wasn't used
    if not hasattr(model, 'hf_device_map') or getattr(model, 'hf_device_map', None) is None:
        model = model.to(device)
    
    # Create datasets
    train_dataset = STSBDataset(
        datasets['train'],
        tokenizer,
        max_length=MAX_LENGTH,
    )
    
    eval_dataset = None
    if 'val' in datasets:
        eval_dataset = STSBDataset(
            datasets['val'],
            tokenizer,
            max_length=MAX_LENGTH,
        )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    # When using quantization, we need to disable Accelerate's device placement
    # because quantized models handle their own device placement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="pearson" if eval_dataset else None,  # Use Pearson correlation as best model metric
        greater_is_better=True,  # Higher correlation is better
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        seed=42,
        # Disable device placement when using quantization to avoid Accelerate conflicts
        dataloader_pin_memory=False if use_quantization else True,
    )
    
    # Initialize trainer
    # For quantized models, we need to handle device placement manually
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
    )
    
    # For quantized models on CPU, we need to ensure the model stays on CPU
    # and disable Accelerate's automatic device management
    if use_quantization and not torch.cuda.is_available():
        # Disable device placement in the accelerator
        if hasattr(trainer.accelerator, 'device_placement'):
            trainer.accelerator.device_placement = False
    
    # Train
    print("\nStarting training...")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_result = trainer.train()
    
    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metrics
    metrics = train_result.metrics
    with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
    
    # Evaluate on validation set if available
    if eval_dataset:
        print("\nEvaluating on validation set (20% split)...")
        eval_results = trainer.evaluate()
        print(f"Validation MSE: {eval_results.get('eval_mse', 'N/A'):.4f}")
        print(f"Validation RMSE: {eval_results.get('eval_rmse', 'N/A'):.4f}")
        print(f"Validation Pearson Correlation: {eval_results.get('eval_pearson', 'N/A'):.4f}")
        print(f"Validation Spearman Correlation: {eval_results.get('eval_spearman', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Evaluate on dev set if available
    if 'dev' in datasets:
        print("\nEvaluating on dev set...")
        dev_dataset = STSBDataset(
            datasets['dev'],
            tokenizer,
            max_length=MAX_LENGTH,
        )
        dev_results = trainer.evaluate(dev_dataset)
        print(f"Dev MSE: {dev_results.get('eval_mse', 'N/A'):.4f}")
        print(f"Dev RMSE: {dev_results.get('eval_rmse', 'N/A'):.4f}")
        print(f"Dev Pearson Correlation: {dev_results.get('eval_pearson', 'N/A'):.4f}")
        print(f"Dev Spearman Correlation: {dev_results.get('eval_spearman', 'N/A'):.4f}")
        
        # Save dev metrics
        with open(os.path.join(OUTPUT_DIR, "dev_metrics.json"), "w") as f:
            json.dump(dev_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

