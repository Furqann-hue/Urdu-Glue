"""
Fine-tune XLM-RoBERTa-Large model with LoRA technique on Urdu Sentiment Analysis dataset
Binary classification task (0 = negative, 1 = positive)
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
OUTPUT_DIR = "./roberta_lora_urdu_sentiment"
DATA_DIR = "Documents/research"

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)


class UrduSentimentDataset(Dataset):
    """Dataset class for Urdu Sentiment Analysis task"""
    
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
        
        # Get label (should be 0 or 1)
        label = int(example[self.label_column]) if self.label_column in example else 0
        # Ensure label is 0 or 1 for binary classification
        label = 1 if label > 0 else 0
        
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


def load_urdu_sentiment_data(train_split: float = 0.8, random_state: int = 42):
    """
    Load Urdu Sentiment dataset from Hugging Face and split into train/validation
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, and test datasets
    """
    print("Loading Urdu Sentiment dataset from Hugging Face...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("sepidmnorozy/Urdu_sentiment")
    
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
    
    # Check column names
    if len(full_data) > 0:
        print(f"Available columns: {full_data[0].keys()}")
    
    # Convert to pandas for easier splitting
    df = pd.DataFrame(full_data)
    
    # Split into 80% train and 20% validation
    train_data, val_data = train_test_split(
        df,
        test_size=1 - train_split,
        random_state=random_state,
        shuffle=True,
        stratify=df['label'] if 'label' in df.columns else None  # Stratify if possible
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
        test_list = pd.DataFrame(test_data).to_dict('records')
        datasets['test'] = test_list
        print(f"✓ Test set: {len(test_list)} samples")
    
    return datasets


def compute_metrics(eval_pred):
    """Compute metrics for sentiment analysis task"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def setup_lora_model(model_name: str = MODEL_NAME, num_labels: int = 2):
    """
    Setup XLM-RoBERTa-Large model with LoRA adapters
    
    Args:
        model_name: Name of the base model
        num_labels: Number of classification labels (2 for binary sentiment)
        
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
    print("XLM-RoBERTa-Large LoRA Fine-tuning for Urdu Sentiment Analysis")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_urdu_sentiment_data()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    # Setup model and tokenizer
    model, tokenizer = setup_lora_model()
    model.to(device)
    
    # Determine column names from first example
    first_example = datasets['train'][0]
    text_col = "text" if "text" in first_example else list(first_example.keys())[0]
    label_col = "label" if "label" in first_example else list(first_example.keys())[-1]
    
    print(f"Using text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    
    # Create datasets
    train_dataset = UrduSentimentDataset(
        datasets['train'],
        tokenizer,
        max_length=MAX_LENGTH,
        text_column=text_col,
        label_column=label_col,
    )
    
    eval_dataset = None
    if 'val' in datasets:
        eval_dataset = UrduSentimentDataset(
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
        metric_for_best_model="f1" if eval_dataset else None,
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
        print(f"Validation F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print(f"Validation Precision: {eval_results.get('eval_precision', 'N/A'):.4f}")
        print(f"Validation Recall: {eval_results.get('eval_recall', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Test on test set if available
    if 'test' in datasets:
        print("\nEvaluating on test set...")
        test_dataset = UrduSentimentDataset(
            datasets['test'],
            tokenizer,
            max_length=MAX_LENGTH,
            text_column=text_col,
            label_column=label_col,
        )
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Test F1 Score: {test_results.get('eval_f1', 'N/A'):.4f}")
        print(f"Test Precision: {test_results.get('eval_precision', 'N/A'):.4f}")
        print(f"Test Recall: {test_results.get('eval_recall', 'N/A'):.4f}")
        
        # Save test metrics
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

