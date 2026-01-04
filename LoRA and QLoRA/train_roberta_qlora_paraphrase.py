"""
Fine-tune XLM-RoBERTa-Large model with QLoRA (Quantized LoRA) technique on UPPC (Urdu Paraphrase Plagiarism Corpus) dataset
QLoRA uses 4-bit quantization for memory-efficient fine-tuning
Binary classification task - Paraphrase detection (P vs NP)
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import xml.etree.ElementTree as ET
import re

# Configuration
MODEL_NAME = "xlm-roberta-large"  # XLM-RoBERTa-Large model
MAX_LENGTH = 256  # Longer for sentence pairs
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./roberta_qlora_paraphrase"
DATA_DIR = "Documents/research"
UPPC_DIR = os.path.join(DATA_DIR, "UPPC Corpus")
UPPC_DATA_DIR = os.path.join(UPPC_DIR, "data")
ALL_FILES_PATH = os.path.join(UPPC_DIR, "all_files.txt")
NUM_LABELS = 2  # Binary classification: Paraphrase (1) or Non-Paraphrase (0)

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
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)


class UrduParaphraseDataset(Dataset):
    """Dataset class for Urdu Paraphrase Detection task"""
    
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


def extract_text_from_xml(xml_path: str) -> str:
    """
    Extract text content from UPPC XML file
    
    Args:
        xml_path: Path to the XML file
        
    Returns:
        Extracted text content
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get text from UPPC_document tag
        text = root.text if root.text else ""
        
        # Also get text from all child elements
        for elem in root.iter():
            if elem.text:
                text += " " + elem.text
            if elem.tail:
                text += " " + elem.tail
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return ""


def load_uppc_data(train_split: float = 0.8, random_state: int = 42) -> Dict:
    """
    Load UPPC dataset and create sentence pairs for paraphrase detection
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train and val datasets
    """
    print("Loading UPPC (Urdu Paraphrase Plagiarism Corpus) dataset...")
    print(f"Data directory: {UPPC_DATA_DIR}")
    
    # Read all_files.txt to get pairs
    pairs = []
    if not os.path.exists(ALL_FILES_PATH):
        raise FileNotFoundError(f"Could not find {ALL_FILES_PATH}")
    
    print(f"Reading pairs from {ALL_FILES_PATH}...")
    with open(ALL_FILES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 3:
                original_file = parts[0].strip()
                comparison_file = parts[1].strip()
                label_str = parts[2].strip().upper()
                
                # Convert label: P -> 1 (Paraphrase), NP -> 0 (Non-Paraphrase)
                label = 1 if label_str == "P" else 0
                
                pairs.append({
                    "original_file": original_file,
                    "comparison_file": comparison_file,
                    "label": label,
                })
    
    print(f"✓ Found {len(pairs)} pairs in all_files.txt")
    
    # Extract text from XML files and create sentence pairs
    print("Extracting text from XML files...")
    data = []
    missing_files = []
    
    for pair in pairs:
        original_path = os.path.join(UPPC_DATA_DIR, pair["original_file"])
        comparison_path = os.path.join(UPPC_DATA_DIR, pair["comparison_file"])
        
        # Check if files exist
        if not os.path.exists(original_path):
            missing_files.append(original_path)
            continue
        if not os.path.exists(comparison_path):
            missing_files.append(comparison_path)
            continue
        
        # Extract text from both files
        sentence1 = extract_text_from_xml(original_path)
        sentence2 = extract_text_from_xml(comparison_path)
        
        # Skip if either text is empty
        if not sentence1 or not sentence2:
            continue
        
        data.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": pair["label"],
        })
    
    if missing_files:
        print(f"⚠️  Warning: {len(missing_files)} files were missing. Skipped those pairs.")
        if len(missing_files) <= 10:
            print(f"Missing files: {missing_files}")
    
    print(f"✓ Successfully loaded {len(data)} sentence pairs")
    
    # Check label distribution
    labels = [d["label"] for d in data]
    paraphrase_count = sum(labels)
    non_paraphrase_count = len(labels) - paraphrase_count
    print(f"✓ Label distribution:")
    print(f"   Paraphrase (1): {paraphrase_count}")
    print(f"   Non-Paraphrase (0): {non_paraphrase_count}")
    
    # Split into 80% train and 20% validation
    train_data, val_data = train_test_split(
        data,
        test_size=1 - train_split,
        random_state=random_state,
        shuffle=True,
        stratify=labels,  # Stratify by label to maintain distribution
    )
    
    datasets = {
        'train': train_data,
        'val': val_data,
    }
    
    print(f"✓ Train set (80%): {len(train_data)} pairs")
    print(f"✓ Validation set (20%): {len(val_data)} pairs")
    
    return datasets


def compute_metrics(eval_pred):
    """Compute metrics for binary classification task"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    f1_macro = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="binary", zero_division=0)
    recall = recall_score(labels, predictions, average="binary", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
    }


def setup_qlora_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
    """
    Setup XLM-RoBERTa-Large model with QLoRA (Quantized LoRA) adapters
    
    Args:
        model_name: Name of the base model
        num_labels: Number of classification labels (2 for binary classification)
        
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
            problem_type="single_label_classification",
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
                problem_type="single_label_classification",
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
                problem_type="single_label_classification",
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
    print("XLM-RoBERTa-Large QLoRA Fine-tuning for Urdu Paraphrase Detection (UPPC Dataset)")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_uppc_data()
    
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
    train_dataset = UrduParaphraseDataset(
        datasets['train'],
        tokenizer,
        max_length=MAX_LENGTH,
    )
    
    eval_dataset = None
    if 'val' in datasets:
        eval_dataset = UrduParaphraseDataset(
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
        metric_for_best_model="f1" if eval_dataset else None,
        greater_is_better=True,
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
        print(f"Validation Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Validation F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print(f"Validation F1 (Macro): {eval_results.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"Validation Precision: {eval_results.get('eval_precision', 'N/A'):.4f}")
        print(f"Validation Recall: {eval_results.get('eval_recall', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

