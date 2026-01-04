"""
Fine-tune RoBERTa-XL model with QLoRA (Quantized LoRA) technique on Urdu POS (Part-of-Speech) Tagging dataset
QLoRA uses 4-bit quantization for memory-efficient fine-tuning
Token Classification task - Assign POS tag to each token
"""
import os
import tarfile
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
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
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Configuration
MODEL_NAME = "xlm-roberta-large"  # RoBERTa-XL model
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
OUTPUT_DIR = "./roberta_qlora_pos"
DATA_DIR = "/Users/phaedrasolutions/Documents/research"
POS_DIR = os.path.join(DATA_DIR, "pos")
TRAIN_FILE = os.path.join(POS_DIR, "train.txt.tar.gz")
TEST_FILE = os.path.join(POS_DIR, "test.txt.tar.gz")

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
    task_type=TaskType.TOKEN_CLS,  # Token Classification
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)


class POSDataset(Dataset):
    """Dataset class for POS Tagging task"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 256,
    ):
        """
        Initialize the dataset
        
        Args:
            data: List of dictionaries with 'tokens' and 'labels' keys
            tokenizer: Hugging Face tokenizer
            label_to_id: Mapping from label strings to integer IDs
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = example["tokens"]
        labels = example["labels"]
        
        # Tokenize and align labels
        # BERT tokenizer may split words into subwords
        # We need to align word-level labels with subword tokens
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Get word IDs for alignment
        word_ids = encoded.word_ids()
        
        # Align labels: first subword gets the label, others get -100 (ignored)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, PAD) get -100
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                label = labels[word_idx] if word_idx < len(labels) else self.label_to_id.get("O", 0)
                aligned_labels.append(self.label_to_id.get(label, 0))
            else:
                # Subsequent subwords get -100 (ignored)
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).flatten(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_pos_data(file_path: str) -> List[Dict]:
    """
    Load POS tagging data from CoNLL format file
    
    Format: word\tPOS_tag (one per line, blank lines separate sentences)
    
    Args:
        file_path: Path to the data file (can be .tar.gz or .txt)
    
    Returns:
        List of dictionaries with 'tokens' and 'labels' keys
    """
    print(f"Loading POS data from {file_path}...")
    
    # Extract if tar.gz
    if file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:gz') as tar:
            # Get the first file from the archive
            members = tar.getmembers()
            if members:
                extracted_file = tar.extractfile(members[0])
                lines = extracted_file.read().decode('utf-8').split('\n')
            else:
                raise ValueError(f"No files found in {file_path}")
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    sentences = []
    current_tokens = []
    current_labels = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # Blank line indicates end of sentence
            if current_tokens:
                sentences.append({
                    "tokens": current_tokens,
                    "labels": current_labels,
                })
                current_tokens = []
                current_labels = []
        else:
            # Parse word\tPOS_tag format
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[0].strip()
                pos_tag = parts[1].strip()
                if word and pos_tag:
                    current_tokens.append(word)
                    current_labels.append(pos_tag)
    
    # Add last sentence if file doesn't end with blank line
    if current_tokens:
        sentences.append({
            "tokens": current_tokens,
            "labels": current_labels,
        })
    
    print(f"✓ Loaded {len(sentences)} sentences")
    return sentences


def create_label_mapping(data: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID and ID to label mappings from the dataset
    
    Args:
        data: List of sentence dictionaries with 'labels' keys
    
    Returns:
        Tuple of (label_to_id, id_to_label) dictionaries
    """
    all_labels = set()
    for sentence in data:
        all_labels.update(sentence["labels"])
    
    # Sort labels for consistent mapping
    sorted_labels = sorted(all_labels)
    
    # Create mappings
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"✓ Found {len(label_to_id)} unique POS tags")
    print(f"  Tags: {sorted_labels[:20]}{'...' if len(sorted_labels) > 20 else ''}")
    
    return label_to_id, id_to_label


def load_pos_dataset(train_split: float = 0.8, random_state: int = 42) -> Dict:
    """
    Load POS tagging dataset and create train/validation splits
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, test datasets and label mappings
    """
    print("Loading Urdu POS Tagging dataset...")
    print(f"Data directory: {POS_DIR}")
    
    # Load training data
    train_data = []
    if os.path.exists(TRAIN_FILE):
        train_data = load_pos_data(TRAIN_FILE)
    elif os.path.exists(TRAIN_FILE.replace('.tar.gz', '')):
        train_data = load_pos_data(TRAIN_FILE.replace('.tar.gz', ''))
    else:
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    
    # Create label mappings from training data
    label_to_id, id_to_label = create_label_mapping(train_data)
    
    # Split training data into 80% train and 20% validation
    train_list, val_list = train_test_split(
        train_data,
        test_size=1 - train_split,
        random_state=random_state,
        shuffle=True,
    )
    
    print(f"✓ Train set (80%): {len(train_list)} sentences")
    print(f"✓ Validation set (20%): {len(val_list)} sentences")
    
    # Load test data if available
    test_data = []
    if os.path.exists(TEST_FILE):
        test_data = load_pos_data(TEST_FILE)
        print(f"✓ Test set: {len(test_data)} sentences")
    elif os.path.exists(TEST_FILE.replace('.tar.gz', '')):
        test_data = load_pos_data(TEST_FILE.replace('.tar.gz', ''))
        print(f"✓ Test set: {len(test_data)} sentences")
    
    return {
        "train": train_list,
        "val": val_list,
        "test": test_data,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }


def compute_metrics(eval_pred, id_to_label: Dict[int, str]):
    """
    Compute metrics for token classification task
    
    Args:
        eval_pred: Tuple of (predictions, labels) from the trainer
        id_to_label: Mapping from label IDs to label strings
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Get predicted class (argmax)
    predictions = np.argmax(predictions, axis=2)
    
    # Convert to label strings and remove ignored tokens (-100)
    true_labels = []
    pred_labels = []
    
    for i in range(len(predictions)):
        true_seq = []
        pred_seq = []
        for j in range(len(predictions[i])):
            if labels[i][j] != -100:  # Ignore padding and special tokens
                true_seq.append(id_to_label[labels[i][j]])
                pred_seq.append(id_to_label[predictions[i][j]])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    
    # Calculate metrics using seqeval
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def setup_qlora_model(model_name: str = MODEL_NAME, num_labels: int = None):
    """
    Setup RoBERTa-XL model with QLoRA (Quantized LoRA) adapters
    
    Args:
        model_name: Name of the base model
        num_labels: Number of classification labels
        
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
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
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
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
            )
            
            # Apply LoRA directly (no quantization)
            print("Applying LoRA adapters...")
            model = get_peft_model(model, LORA_CONFIG)
            return model, tokenizer, False  # False = no quantization used
        else:
            print(f"Loading model with 4-bit quantization: {model_name}")
            
            # Load base model with 4-bit quantization (GPU only)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
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
    print("RoBERTa-XL QLoRA Fine-tuning for Urdu POS Tagging")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_pos_dataset()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    label_to_id = datasets["label_to_id"]
    id_to_label = datasets["id_to_label"]
    num_labels = len(label_to_id)
    
    print(f"\nNumber of POS tags: {num_labels}")
    
    # Setup model and tokenizer
    model, tokenizer, use_quantization = setup_qlora_model(num_labels=num_labels)
    
    # Ensure model is on the correct device
    # If device_map wasn't used (CPU case), model should already be moved in setup_qlora_model
    # But we double-check here for safety - only move if device_map wasn't used
    if not hasattr(model, 'hf_device_map') or getattr(model, 'hf_device_map', None) is None:
        model = model.to(device)
    
    # Create datasets
    train_dataset = POSDataset(
        datasets['train'],
        tokenizer,
        label_to_id,
        max_length=MAX_LENGTH,
    )
    
    eval_dataset = None
    if 'val' in datasets and datasets['val']:
        eval_dataset = POSDataset(
            datasets['val'],
            tokenizer,
            label_to_id,
            max_length=MAX_LENGTH,
        )
    
    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )
    
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
        metric_for_best_model="f1" if eval_dataset else None,  # Use F1 as best model metric
        greater_is_better=True,  # Higher F1 is better
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        seed=42,
        # Disable device placement when using quantization to avoid Accelerate conflicts
        dataloader_pin_memory=False if use_quantization else True,
    )
    
    # Create compute_metrics function with label mapping
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, id_to_label)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
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
    
    # Save label mappings
    with open(os.path.join(OUTPUT_DIR, "label_to_id.json"), "w") as f:
        json.dump(label_to_id, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "id_to_label.json"), "w") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2, ensure_ascii=False)
    
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
        print(f"Validation Precision: {eval_results.get('eval_precision', 'N/A'):.4f}")
        print(f"Validation Recall: {eval_results.get('eval_recall', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Evaluate on test set if available
    if 'test' in datasets and datasets['test']:
        print("\nEvaluating on test set...")
        test_dataset = POSDataset(
            datasets['test'],
            tokenizer,
            label_to_id,
            max_length=MAX_LENGTH,
        )
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Test F1: {test_results.get('eval_f1', 'N/A'):.4f}")
        print(f"Test Precision: {test_results.get('eval_precision', 'N/A'):.4f}")
        print(f"Test Recall: {test_results.get('eval_recall', 'N/A'):.4f}")
        
        # Save test metrics
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

