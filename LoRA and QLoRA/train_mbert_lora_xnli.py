"""
Fine-tune mBERT model with LoRA technique on XNLI (Cross-lingual Natural Language Inference) dataset
3-class classification task - Entailment, Contradiction, Neutral
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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional
import json

# Configuration
MODEL_NAME = "bert-base-multilingual-cased"  # mBERT model
MAX_LENGTH = 256  # Longer for sentence pairs
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./mbert_lora_xnli"
DATA_DIR = "Documents/research"
XNLI_DIR = os.path.join(DATA_DIR, "xnli_dataset")
DEV_FILE = os.path.join(XNLI_DIR, "XNLI_Dev_Urdu - Sheet1 (1).tsv")
TEST_FILE = os.path.join(XNLI_DIR, "XNLI_test_urdu - Sheet1.tsv")
NUM_LABELS = 3  # 3-class classification: entailment, contradiction, neutral

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)

# Label mapping for XNLI
LABEL_MAP = {
    "entailment": 0,
    "contradiction": 1,
    "neutral": 2,
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class XNLIDataset(Dataset):
    """Dataset class for XNLI (Cross-lingual Natural Language Inference) task"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 256,
    ):
        """
        Initialize the dataset
        
        Args:
            data: List of dictionaries with 'sentence1', 'sentence2', and 'label' keys
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
        
        # Get sentence pair and label
        sentence1 = str(example.get("sentence1", ""))
        sentence2 = str(example.get("sentence2", ""))
        label = int(example.get("label", 0))
        
        # Clean text
        if not sentence1 or sentence1 == "nan":
            sentence1 = ""
        if not sentence2 or sentence2 == "nan":
            sentence2 = ""
        
        # Ensure label is in valid range [0, 2]
        label = max(0, min(2, label))
        
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
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_xnli_data(train_split: float = 0.8, random_state: int = 42) -> Dict:
    """
    Load XNLI dataset and create sentence pairs with labels
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, and test datasets
    """
    print("Loading XNLI (Cross-lingual Natural Language Inference) dataset...")
    print(f"Data directory: {XNLI_DIR}")
    
    datasets = {}
    
    # Load dev data (will be split into train/val)
    if os.path.exists(DEV_FILE):
        print(f"Loading dev data from {DEV_FILE}...")
        dev_df = pd.read_csv(DEV_FILE, sep='\t', on_bad_lines='skip', engine='python')
        
        # Check column names
        print(f"Dev file columns: {dev_df.columns.tolist()}")
        print(f"Dev file shape: {dev_df.shape}")
        print(f"First few rows:\n{dev_df.head(3)}")
        
        # Find columns (should be gold_label, sentence1, sentence2)
        label_col = None
        sent1_col = None
        sent2_col = None
        
        for col in dev_df.columns:
            col_lower = str(col).lower().strip()
            if 'label' in col_lower or 'gold' in col_lower:
                label_col = col
            elif 'sentence1' in col_lower or 'sent1' in col_lower or 'premise' in col_lower:
                sent1_col = col
            elif 'sentence2' in col_lower or 'sent2' in col_lower or 'hypothesis' in col_lower:
                sent2_col = col
        
        if label_col is None or sent1_col is None or sent2_col is None:
            raise ValueError(f"Could not find required columns in {DEV_FILE}. Available columns: {dev_df.columns.tolist()}")
        
        print(f"Using columns: label='{label_col}', sentence1='{sent1_col}', sentence2='{sent2_col}'")
        
        # Create data list
        dev_data = []
        label_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
        
        for _, row in dev_df.iterrows():
            sent1 = str(row[sent1_col]) if pd.notna(row[sent1_col]) else ""
            sent2 = str(row[sent2_col]) if pd.notna(row[sent2_col]) else ""
            label_str = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else ""
            
            # Skip if missing data
            if not sent1 or not sent2 or not label_str:
                continue
            
            # Map label string to integer
            if label_str in LABEL_MAP:
                label = LABEL_MAP[label_str]
                label_counts[label_str] += 1
            else:
                # Try to handle variations
                if "entail" in label_str:
                    label = 0
                    label_counts["entailment"] += 1
                elif "contradict" in label_str:
                    label = 1
                    label_counts["contradiction"] += 1
                elif "neutral" in label_str:
                    label = 2
                    label_counts["neutral"] += 1
                else:
                    print(f"Warning: Unknown label '{label_str}', skipping...")
                    continue
            
            dev_data.append({
                "sentence1": sent1,
                "sentence2": sent2,
                "label": label,
            })
        
        print(f"✓ Loaded {len(dev_data)} dev examples")
        print(f"✓ Label distribution: {label_counts}")
        
        # Split into 80% train and 20% validation
        train_list, val_list = train_test_split(
            dev_data,
            test_size=1 - train_split,
            random_state=random_state,
            shuffle=True,
            stratify=[d["label"] for d in dev_data],  # Stratify by label for balanced split
        )
        
        datasets['train'] = train_list
        datasets['val'] = val_list
        print(f"✓ Train set (80%): {len(train_list)} pairs")
        print(f"✓ Validation set (20%): {len(val_list)} pairs")
    else:
        raise FileNotFoundError(f"Could not find dev file: {DEV_FILE}")
    
    # Load test data (if available)
    if os.path.exists(TEST_FILE):
        try:
            print(f"\nLoading test data from {TEST_FILE}...")
            test_df = pd.read_csv(TEST_FILE, sep='\t', on_bad_lines='skip', engine='python')
            
            print(f"Test file columns: {test_df.columns.tolist()}")
            print(f"Test file shape: {test_df.shape}")
            
            # Find columns
            label_col = None
            sent1_col = None
            sent2_col = None
            
            for col in test_df.columns:
                col_lower = str(col).lower().strip()
                if 'label' in col_lower or 'gold' in col_lower:
                    label_col = col
                elif 'sentence1' in col_lower or 'sent1' in col_lower or 'premise' in col_lower:
                    sent1_col = col
                elif 'sentence2' in col_lower or 'sent2' in col_lower or 'hypothesis' in col_lower:
                    sent2_col = col
            
            if label_col and sent1_col and sent2_col:
                test_data = []
                for _, row in test_df.iterrows():
                    sent1 = str(row[sent1_col]) if pd.notna(row[sent1_col]) else ""
                    sent2 = str(row[sent2_col]) if pd.notna(row[sent2_col]) else ""
                    label_str = str(row[label_col]).strip().lower() if pd.notna(row[label_col]) else ""
                    
                    if not sent1 or not sent2 or not label_str:
                        continue
                    
                    # Map label string to integer
                    if label_str in LABEL_MAP:
                        label = LABEL_MAP[label_str]
                    else:
                        # Try to handle variations
                        if "entail" in label_str:
                            label = 0
                        elif "contradict" in label_str:
                            label = 1
                        elif "neutral" in label_str:
                            label = 2
                        else:
                            continue
                    
                    test_data.append({
                        "sentence1": sent1,
                        "sentence2": sent2,
                        "label": label,
                    })
                
                if test_data:
                    datasets['test'] = test_data
                    print(f"✓ Loaded {len(test_data)} test examples")
            else:
                print(f"⚠️  Warning: Could not find required columns in test file. Skipping test data.")
        except Exception as e:
            print(f"⚠️  Warning: Could not load test file: {e}")
            print(f"   Skipping test data. Training will continue with train/val data only.")
    
    return datasets


def compute_metrics(eval_pred):
    """Compute metrics for 3-class classification task"""
    predictions, labels = eval_pred
    
    # Get predicted class (argmax)
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_entailment": f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        "f1_contradiction": f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        "f1_neutral": f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
    }


def main():
    """Main training function"""
    print("=" * 60)
    print("mBERT LoRA Fine-tuning for XNLI (Cross-lingual Natural Language Inference)")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_xnli_data()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    # Create datasets
    train_dataset = XNLIDataset(
        datasets['train'],
        tokenizer,
        max_length=MAX_LENGTH,
    )
    
    eval_dataset = None
    if 'val' in datasets:
        eval_dataset = XNLIDataset(
            datasets['val'],
            tokenizer,
            max_length=MAX_LENGTH,
        )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
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
        metric_for_best_model="f1_macro" if eval_dataset else None,  # Use macro F1 as best model metric
        greater_is_better=True,  # Higher F1 is better
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        seed=42,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
    )
    
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
        print(f"Validation Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Validation F1 (Macro): {eval_results.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"Validation F1 (Weighted): {eval_results.get('eval_f1_weighted', 'N/A'):.4f}")
        print(f"Validation Precision (Macro): {eval_results.get('eval_precision_macro', 'N/A'):.4f}")
        print(f"Validation Recall (Macro): {eval_results.get('eval_recall_macro', 'N/A'):.4f}")
        print(f"Validation F1 (Entailment): {eval_results.get('eval_f1_entailment', 'N/A'):.4f}")
        print(f"Validation F1 (Contradiction): {eval_results.get('eval_f1_contradiction', 'N/A'):.4f}")
        print(f"Validation F1 (Neutral): {eval_results.get('eval_f1_neutral', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Evaluate on test set if available
    if 'test' in datasets:
        print("\nEvaluating on test set...")
        test_dataset = XNLIDataset(
            datasets['test'],
            tokenizer,
            max_length=MAX_LENGTH,
        )
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Test F1 (Macro): {test_results.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"Test F1 (Weighted): {test_results.get('eval_f1_weighted', 'N/A'):.4f}")
        print(f"Test Precision (Macro): {test_results.get('eval_precision_macro', 'N/A'):.4f}")
        print(f"Test Recall (Macro): {test_results.get('eval_recall_macro', 'N/A'):.4f}")
        print(f"Test F1 (Entailment): {test_results.get('eval_f1_entailment', 'N/A'):.4f}")
        print(f"Test F1 (Contradiction): {test_results.get('eval_f1_contradiction', 'N/A'):.4f}")
        print(f"Test F1 (Neutral): {test_results.get('eval_f1_neutral', 'N/A'):.4f}")
        
        # Save test metrics
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

