"""
Fine-tune XLM-RoBERTa-Large model with LoRA technique on Urdu Multi-Domain Classification dataset
Multi-class classification task - Sentiment domain only (3 classes)
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
from datasets import load_dataset

# Configuration
MODEL_NAME = "xlm-roberta-large"  # XLM-RoBERTa-Large model (multilingual)
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./roberta_lora_multiclass"
DATA_DIR = "Documents/research"
NUM_LABELS = 3  # 3-class sentiment classification

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)


class UrduMulticlassDataset(Dataset):
    """Dataset class for Urdu Multi-class Classification task"""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 128,
        text_column: str = "text",
        label_column: str = "label",
    ):
        """
        Initialize the dataset
        
        Args:
            data: Dataset or list of examples
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get text and label
        text = str(example[self.text_column])
        if not text or text == "nan":
            text = ""
        
        # Get label (should be 0, 1, or 2 for 3-class classification)
        label = int(example[self.label_column]) if self.label_column in example else 0
        # Ensure label is within valid range [0, NUM_LABELS-1]
        label = max(0, min(label, NUM_LABELS - 1))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_urdu_multiclass_data(train_split: float = 0.8, random_state: int = 42):
    """
    Load Urdu Multi-Domain Classification dataset from Hugging Face
    Filter for sentiment domain only (3 classes) and split into train/validation
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, and test datasets, and label column name
    """
    print("Loading Urdu Multi-Domain Classification dataset from Hugging Face...")
    print("Note: You may need to login using 'huggingface-cli login' to access this dataset")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("umar178/UrduMultiDomainClassification")
    
    datasets = {}
    
    # Check available splits
    print(f"Available splits: {dataset.keys()}")
    
    # Get the main split (usually 'train' or 'default')
    main_split = None
    if 'train' in dataset:
        main_split = 'train'
    elif len(dataset) > 0:
        main_split = list(dataset.keys())[0]
    
    if main_split is None:
        raise ValueError("Could not find a suitable split in the dataset")
    
    # Get the data
    full_data = dataset[main_split]
    print(f"✓ Full dataset: {len(full_data)} samples")
    
    # Check column names and structure
    if len(full_data) > 0:
        print(f"Available columns: {full_data[0].keys()}")
        print(f"Sample example: {full_data[0]}")
    
    # Convert to pandas for easier filtering and splitting
    df = pd.DataFrame(full_data)
    
    # Filter for sentiment domain only
    # Check what domain column is called
    domain_col = None
    for col in ['domain', 'Domain', 'DOMAIN', 'category', 'Category']:
        if col in df.columns:
            domain_col = col
            break
    
    if domain_col is None:
        print("⚠️  Warning: Could not find domain column. Using all data.")
        sentiment_df = df
    else:
        # Filter for sentiment domain
        print(f"Filtering for sentiment domain using column: '{domain_col}'")
        print(f"Available domains: {df[domain_col].unique()}")
        
        # Try different possible values for sentiment domain
        sentiment_keywords = ['sentiment', 'Sentiment', 'SENTIMENT', 'sent', 'Sent']
        sentiment_df = None
        
        for keyword in sentiment_keywords:
            if sentiment_df is None:
                sentiment_df = df[df[domain_col].astype(str).str.contains(keyword, case=False, na=False)]
                if len(sentiment_df) > 0:
                    print(f"✓ Found {len(sentiment_df)} samples in sentiment domain")
                    break
        
        if sentiment_df is None or len(sentiment_df) == 0:
            print("⚠️  Warning: Could not find sentiment domain. Using all data.")
            sentiment_df = df
        else:
            print(f"✓ Filtered to sentiment domain: {len(sentiment_df)} samples")
    
    # Check label distribution
    label_col = "label" if "label" in sentiment_df.columns else [col for col in sentiment_df.columns if 'label' in col.lower()][0] if any('label' in col.lower() for col in sentiment_df.columns) else sentiment_df.columns[-1]
    print(f"Using label column: '{label_col}'")
    print(f"Label distribution:\n{sentiment_df[label_col].value_counts().sort_index()}")
    
    # Ensure labels are 0, 1, 2 (3 classes)
    unique_labels = sorted(sentiment_df[label_col].unique())
    print(f"Unique labels found: {unique_labels}")
    
    # Create label mapping if needed (map to 0, 1, 2)
    if len(unique_labels) > NUM_LABELS:
        print(f"⚠️  Warning: Found {len(unique_labels)} unique labels, but expecting {NUM_LABELS}")
        print(f"   Using first {NUM_LABELS} labels")
        unique_labels = unique_labels[:NUM_LABELS]
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    print(f"Label mapping: {label_mapping}")
    
    # Apply label mapping
    sentiment_df[label_col] = sentiment_df[label_col].map(label_mapping)
    sentiment_df = sentiment_df.dropna(subset=[label_col])  # Remove any unmapped labels
    
    print(f"✓ Final dataset size: {len(sentiment_df)} samples")
    print(f"Final label distribution:\n{sentiment_df[label_col].value_counts().sort_index()}")
    
    # Split into 80% train and 20% validation
    train_data, val_data = train_test_split(
        sentiment_df,
        test_size=1 - train_split,
        random_state=random_state,
        shuffle=True,
        stratify=sentiment_df[label_col] if label_col in sentiment_df.columns else None  # Stratify if possible
    )
    
    # Convert back to list of dicts for dataset
    train_list = train_data.to_dict('records')
    val_list = val_data.to_dict('records')
    
    datasets['train'] = train_list
    datasets['val'] = val_list
    print(f"✓ Train set (80%): {len(train_list)} samples")
    print(f"✓ Validation set (20%): {len(val_list)} samples")
    
    # Check if test split exists
    if 'test' in dataset:
        test_data = dataset['test']
        test_df = pd.DataFrame(test_data)
        
        # Filter test set for sentiment domain if domain column exists
        if domain_col and domain_col in test_df.columns:
            for keyword in sentiment_keywords:
                test_sentiment = test_df[test_df[domain_col].astype(str).str.contains(keyword, case=False, na=False)]
                if len(test_sentiment) > 0:
                    test_df = test_sentiment
                    break
        
        # Apply same label mapping
        if label_col in test_df.columns:
            test_df[label_col] = test_df[label_col].map(label_mapping)
            test_df = test_df.dropna(subset=[label_col])
        
        test_list = test_df.to_dict('records')
        datasets['test'] = test_list
        print(f"✓ Test set: {len(test_list)} samples")
    
    return datasets, label_col


def compute_metrics(eval_pred):
    """Compute metrics for multi-class classification task"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def setup_lora_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
    """
    Setup XLM-RoBERTa-Large model with LoRA adapters
    
    Args:
        model_name: Name of the base model
        num_labels: Number of classification labels (3 for sentiment domain)
        
    Returns:
        Model with LoRA adapters and tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main training function"""
    print("=" * 60)
    print("XLM-RoBERTa-Large LoRA Fine-tuning for Urdu Multi-class Classification (Sentiment Domain)")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets, label_col = load_urdu_multiclass_data()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    # Setup model and tokenizer
    model, tokenizer = setup_lora_model(num_labels=NUM_LABELS)
    model.to(device)
    
    # Determine column names from first example
    first_example = datasets['train'][0]
    text_col = "text" if "text" in first_example else [col for col in first_example.keys() if col != label_col][0]
    
    print(f"Using text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    
    # Create datasets
    train_dataset = UrduMulticlassDataset(
        datasets['train'],
        tokenizer,
        max_length=MAX_LENGTH,
        text_column=text_col,
        label_column=label_col,
    )
    
    eval_dataset = None
    if 'val' in datasets:
        eval_dataset = UrduMulticlassDataset(
            datasets['val'],
            tokenizer,
            max_length=MAX_LENGTH,
            text_column=text_col,
            label_column=label_col,
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
        metric_for_best_model="f1_macro" if eval_dataset else None,
        greater_is_better=True,
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
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Test on test set if available
    if 'test' in datasets:
        print("\nEvaluating on test set...")
        test_dataset = UrduMulticlassDataset(
            datasets['test'],
            tokenizer,
            max_length=MAX_LENGTH,
            text_column=text_col,
            label_column=label_col,
        )
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Test F1 (Macro): {test_results.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"Test F1 (Weighted): {test_results.get('eval_f1_weighted', 'N/A'):.4f}")
        print(f"Test Precision (Macro): {test_results.get('eval_precision_macro', 'N/A'):.4f}")
        print(f"Test Recall (Macro): {test_results.get('eval_recall_macro', 'N/A'):.4f}")
        
        # Save test metrics
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

