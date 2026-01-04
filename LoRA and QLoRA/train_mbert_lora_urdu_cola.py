"""
Fine-tune mBERT model with LoRA technique on Urdu CoLA dataset
CoLA (Corpus of Linguistic Acceptability) - Binary classification task
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
    PeftModel,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional
import json

# Configuration
MODEL_NAME = "bert-base-multilingual-cased"  # mBERT model
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./mbert_lora_urdu_cola"
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


class UrduCoLADataset(Dataset):
    """Dataset class for Urdu CoLA task"""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        sentence_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ):
        """
        Initialize the dataset
        
        Args:
            dataframe: DataFrame containing the data
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            sentence_column: Name of the column containing sentences (auto-detected if None)
            label_column: Name of the column containing labels (auto-detected if None)
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Print available columns for debugging
        print(f"Available columns: {dataframe.columns.tolist()}")
        print(f"DataFrame shape: {dataframe.shape}")
        print(f"First few rows:\n{dataframe.head(3)}")
        
        # Auto-detect columns
        self.sentence_col = sentence_column or self._find_sentence_column(dataframe)
        self.label_col = label_column or self._find_label_column(dataframe)
        
        print(f"✓ Using sentence column: '{self.sentence_col}'")
        print(f"✓ Using label column: '{self.label_col}'")
        
        # Create label mapping for string labels
        self.label_map = self._create_label_mapping()
        print(f"✓ Label mapping: {self.label_map}")
    
    def _find_sentence_column(self, df: pd.DataFrame):
        """Find the sentence column - look for text-like columns"""
        # Common sentence column names
        sentence_candidates = [
            "sentence", "Sentence", "SENTENCE",
            "text", "Text", "TEXT",
            "sentence1", "sentence_1",
            "source", "Source",
            "urdu", "Urdu", "URDU",
        ]
        
        for col in sentence_candidates:
            if col in df.columns:
                return col
        
        # If not found, look for columns with string/text data
        # Exclude columns that look like IDs or labels
        for col in df.columns:
            # Convert column name to string for comparison
            col_str = str(col).lower()
            if col_str not in ["label", "id", "index", "idx", "acceptability", "gj04", "*"]:
                # Check if column contains text (sample a few rows)
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    # If values are strings and look like sentences (Urdu text)
                    avg_length = sample_values.astype(str).str.len().mean()
                    # Check if it contains Urdu characters or is long text
                    has_urdu = any(any('\u0600' <= char <= '\u06FF' for char in str(val)) for val in sample_values)
                    if avg_length > 15 or has_urdu:
                        return col
        
        # Last resort: use last column (usually contains the sentence)
        if len(df.columns) > 0:
            return df.columns[-1]
        return df.columns[0]
    
    def _find_label_column(self, df: pd.DataFrame):
        """Find the label column"""
        label_candidates = [
            "label", "Label", "LABEL",
            "acceptability", "Acceptability",
            "is_acceptable", "is_grammatical",
            "target", "Target",
        ]
        
        for col in label_candidates:
            if col in df.columns:
                return col
        
        # If not found, look for columns with numeric or categorical data
        for col in df.columns:
            col_str = str(col).lower()
            if col_str in ["label", "acceptability", "target", "y"]:
                return col
        
        # Check if any column has only 2 unique values (binary classification)
        # Also check for numeric columns (like 0, 1)
        for col in df.columns:
            unique_vals = df[col].nunique()
            # Check if column is numeric or has few unique values
            if unique_vals <= 3:  # Binary or ternary classification
                # Make sure it's not the sentence column
                col_str = str(col).lower()
                if col_str not in ["sentence", "text", "urdu"]:
                    # Check if values look like labels (numeric or short strings)
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        # If all values are numeric or short strings, likely a label
                        all_numeric_or_short = all(
                            (isinstance(val, (int, float)) or 
                             (isinstance(val, str) and len(str(val)) < 10))
                            for val in sample
                        )
                        if all_numeric_or_short:
                            return col
        
        # Last resort: use first numeric column or column with fewest unique values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                return col
        
        # If no numeric column, return column with fewest unique values
        if len(df.columns) > 0:
            col_with_fewest = min(df.columns, key=lambda c: df[c].nunique())
            return col_with_fewest
        
        return df.columns[0]
    
    def _create_label_mapping(self) -> Dict:
        """Create mapping from label values to integers"""
        unique_labels = self.data[self.label_col].dropna().unique()
        unique_labels = [str(label) for label in unique_labels]
        
        print(f"Found unique labels: {unique_labels}")
        
        # Try to detect if labels are already numeric
        numeric_labels = []
        for label in unique_labels:
            try:
                # Try to convert to int/float
                num_val = float(label)
                if num_val.is_integer():
                    numeric_labels.append(int(num_val))
                else:
                    numeric_labels.append(int(num_val))
            except (ValueError, TypeError):
                pass
        
        # If all labels are numeric, use them directly
        if len(numeric_labels) == len(unique_labels):
            label_map = {}
            for label in unique_labels:
                num_val = int(float(label))
                # Ensure binary: map to 0 or 1
                label_map[label] = 1 if num_val > 0 else 0
        else:
            # Create mapping from string labels to integers
            # For CoLA: 0 = ungrammatical, 1 = grammatical
            label_map = {}
            
            # First, try to extract numeric values from string labels (e.g., "c_13" -> 13)
            extracted_nums = {}
            for label in unique_labels:
                # Try to find numbers in the label string
                numbers = re.findall(r'\d+', str(label))
                if numbers:
                    # Use the last number found
                    extracted_nums[label] = int(numbers[-1])
            
            # If we found numbers in most labels, use them
            if len(extracted_nums) >= len(unique_labels) * 0.5:
                for label in unique_labels:
                    if label in extracted_nums:
                        # Map to binary: even -> 0, odd -> 1, or use modulo
                        label_map[label] = extracted_nums[label] % 2
                    else:
                        label_str = str(label).lower()
                        # Check for common grammatical indicators
                        if any(indicator in label_str for indicator in ["1", "grammatical", "accept", "correct", "true", "yes"]):
                            label_map[label] = 1
                        elif any(indicator in label_str for indicator in ["0", "ungrammatical", "reject", "incorrect", "false", "no"]):
                            label_map[label] = 0
                        else:
                            label_map[label] = 0  # Default to 0
            else:
                # Create mapping based on label patterns or alphabetical order
                sorted_labels = sorted(unique_labels)
                for idx, label in enumerate(sorted_labels):
                    label_str = str(label).lower()
                    # Check for common grammatical indicators
                    if any(indicator in label_str for indicator in ["1", "grammatical", "accept", "correct", "true", "yes", "gram"]):
                        label_map[label] = 1
                    elif any(indicator in label_str for indicator in ["0", "ungrammatical", "reject", "incorrect", "false", "no", "ungram"]):
                        label_map[label] = 0
                    else:
                        # Default: map to binary based on position
                        # For binary classification, alternate or use first half vs second half
                        label_map[label] = 1 if idx >= len(sorted_labels) // 2 else 0
        
        return label_map
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get sentence and label
        sentence = str(row[self.sentence_col])
        if pd.isna(sentence) or sentence == "nan":
            sentence = ""
        
        # Convert label using mapping
        label_value = row[self.label_col]
        if pd.isna(label_value):
            label = 0
        else:
            label_str = str(label_value)
            if label_str in self.label_map:
                label = self.label_map[label_str]
            else:
                # Try direct conversion
                try:
                    label = int(float(label_str))
                except (ValueError, TypeError):
                    # Default to 0 if conversion fails
                    label = 0
        
        # Ensure label is 0 or 1 for binary classification
        label = 1 if label > 0 else 0
        
        # Tokenize
        encoding = self.tokenizer(
            sentence,
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


def load_urdu_cola_data(data_dir: str = DATA_DIR, train_split: float = 0.8, random_state: int = 42):
    """
    Load Urdu CoLA dataset from Excel files and split into train/validation
    
    Args:
        data_dir: Directory containing the Excel files
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, and test DataFrames
    """
    train_path = os.path.join(data_dir, "final_train_urdu.xlsx")
    test_path = os.path.join(data_dir, "final_CoLA_test_urdu.xlsx")
    
    datasets = {}
    
    print("Loading Urdu CoLA dataset...")
    
    if os.path.exists(train_path):
        # Load full training data
        full_train_data = pd.read_excel(train_path)
        print(f"✓ Full training set: {len(full_train_data)} samples")
        
        # Split into 80% train and 20% validation
        train_data, val_data = train_test_split(
            full_train_data,
            test_size=1 - train_split,
            random_state=random_state,
            shuffle=True
        )
        
        datasets['train'] = train_data
        datasets['val'] = val_data
        print(f"✓ Train set (80%): {len(train_data)} samples")
        print(f"✓ Validation set (20%): {len(val_data)} samples")
    else:
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    
    if os.path.exists(test_path):
        datasets['test'] = pd.read_excel(test_path)
        print(f"✓ Test set: {len(datasets['test'])} samples")
    
    return datasets


def compute_metrics(eval_pred):
    """Compute metrics for CoLA task (Matthews Correlation Coefficient is the main metric)"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    mcc = matthews_corrcoef(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "matthews_correlation": mcc,
    }


def setup_lora_model(model_name: str = MODEL_NAME, num_labels: int = 2):
    """
    Setup mBERT model with LoRA adapters
    
    Args:
        model_name: Name of the base model
        num_labels: Number of classification labels (2 for CoLA)
        
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
    print("mBERT LoRA Fine-tuning for Urdu CoLA Dataset")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_urdu_cola_data()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    # Setup model and tokenizer
    model, tokenizer = setup_lora_model()
    model.to(device)
    
    # Create datasets
    train_dataset = UrduCoLADataset(
        datasets['train'],
        tokenizer,
        max_length=MAX_LENGTH,
    )
    
    eval_dataset = None
    if 'val' in datasets:
        eval_dataset = UrduCoLADataset(
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
        metric_for_best_model="matthews_correlation" if eval_dataset else None,
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
        print(f"Validation Matthews Correlation: {eval_results.get('eval_matthews_correlation', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Test on test set if available
    if 'test' in datasets:
        print("\nEvaluating on test set...")
        test_dataset = UrduCoLADataset(
            datasets['test'],
            tokenizer,
            max_length=MAX_LENGTH,
        )
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Test F1 Score: {test_results.get('eval_f1', 'N/A'):.4f}")
        print(f"Test Matthews Correlation: {test_results.get('eval_matthews_correlation', 'N/A'):.4f}")
        
        # Save test metrics
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

