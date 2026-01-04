"""
Fine-tune mBERT model with LoRA technique on OpenBookQA Urdu dataset
Question Answering task - Predict answer spans from context
"""
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import re

# Configuration
MODEL_NAME = "bert-base-multilingual-cased"  # mBERT model
MAX_LENGTH = 384  # Standard for QA tasks
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # As specified by user
NUM_EPOCHS = 10
OUTPUT_DIR = "./mbert_lora_qa"
DATA_DIR = "Documents/research"

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification (works for QA too)
    r=8,  # Rank - controls the rank of the low-rank matrices
    lora_alpha=16,  # LoRA alpha scaling parameter
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "key", "value", "dense"],  # Target attention and feed-forward layers
    bias="none",  # Don't train bias parameters
)


def preprocess_qa_data(examples, tokenizer, max_length=384):
    """
    Preprocess question answering data
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tokenized inputs with start and end positions
    """
    questions = [q.strip() if q else "" for q in examples["question"]]
    contexts = [c.strip() if c else "" for c in examples["context"]]
    
    # Tokenize questions and contexts
    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",  # Only truncate context, not question
        stride=128,  # Overlap when truncating
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Handle answers
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        
        # Get answer information
        answers = examples["answers"][sample_idx]
        
        # Handle different answer formats
        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            answer_starts = answers.get("answer_start", [])
        elif isinstance(answers, list):
            if len(answers) > 0 and isinstance(answers[0], dict):
                answer_texts = [a.get("text", "") for a in answers]
                answer_starts = [a.get("answer_start", 0) for a in answers]
            else:
                answer_texts = answers if answers else [""]
                answer_starts = [0] * len(answer_texts)
        else:
            answer_texts = [str(answers)] if answers else [""]
            answer_starts = [0]
        
        # Use first answer if multiple exist
        if len(answer_texts) > 0 and answer_texts[0]:
            answer_text = answer_texts[0]
            answer_start = answer_starts[0] if len(answer_starts) > 0 else 0
            answer_end = answer_start + len(answer_text)
        else:
            # No answer - set to impossible
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # Find context start in tokenized sequence
        sequence_ids = tokenized.sequence_ids(i)
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1
        
        # Check if answer is within the context span
        if context_start >= len(offsets) or context_end < 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # Find token positions for answer
        start_token = None
        end_token = None
        
        for token_idx in range(context_start, min(context_end + 1, len(offsets))):
            if offsets[token_idx] is None:
                continue
            token_start, token_end = offsets[token_idx]
            
            # Check if answer start is within this token
            if token_start <= answer_start < token_end and start_token is None:
                start_token = token_idx
            # Check if answer end is within this token
            if token_start < answer_end <= token_end:
                end_token = token_idx
        
        if start_token is not None and end_token is not None:
            start_positions.append(start_token)
            end_positions.append(end_token)
        else:
            # Answer not found in this chunk - set to impossible
            start_positions.append(0)
            end_positions.append(0)
    
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    
    return tokenized


def compute_qa_metrics(eval_pred):
    """
    Compute metrics for question answering task
    
    Args:
        eval_pred: Tuple of (predictions, labels)
                   predictions is a tuple of (start_logits, end_logits)
                   labels is a tuple of (start_positions, end_positions)
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Predictions is a tuple of (start_logits, end_logits)
    if isinstance(predictions, tuple) and len(predictions) == 2:
        start_logits, end_logits = predictions
    else:
        # Fallback: assume predictions is a single array
        start_logits = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        end_logits = predictions[1] if isinstance(predictions, (list, tuple)) and len(predictions) > 1 else predictions
    
    # Get predicted start and end positions
    start_pred = np.argmax(start_logits, axis=-1)
    end_pred = np.argmax(end_logits, axis=-1)
    
    # Labels is a tuple of (start_positions, end_positions)
    if isinstance(labels, tuple) and len(labels) == 2:
        start_true, end_true = labels
    else:
        # Fallback: assume labels is a 2D array
        start_true = labels[:, 0] if len(labels.shape) > 1 else labels
        end_true = labels[:, 1] if len(labels.shape) > 1 and labels.shape[1] > 1 else labels
    
    # Flatten if needed
    start_pred = start_pred.flatten()
    end_pred = end_pred.flatten()
    start_true = start_true.flatten()
    end_true = end_true.flatten()
    
    # Calculate exact match (start and end both match)
    exact_matches = np.sum((start_pred == start_true) & (end_pred == end_true))
    exact_match = exact_matches / len(start_true) if len(start_true) > 0 else 0.0
    
    # Calculate F1 score (approximate based on overlap)
    # For simplicity, we'll use a basic overlap metric
    overlaps = 0
    for i in range(len(start_pred)):
        pred_start, pred_end = start_pred[i], end_pred[i]
        true_start, true_end = start_true[i], end_true[i]
        
        # Check if there's any overlap
        if not (pred_end < true_start or pred_start > true_end):
            overlaps += 1
    
    f1_approx = overlaps / len(start_pred) if len(start_pred) > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "f1": f1_approx,
    }


def load_qa_data(train_split: float = 0.8, random_state: int = 42):
    """
    Load OpenBookQA Urdu dataset from Hugging Face
    
    Args:
        train_split: Proportion of data to use for training (default: 0.8 for 80-20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train, val, and test datasets
    """
    print("Loading OpenBookQA Urdu dataset from Hugging Face...")
    print("Note: You may need to login using 'huggingface-cli login' to access this dataset")
    
    try:
        dataset = load_dataset("large-traversaal/openbookqa_urdu")
    except Exception as e:
        raise ValueError(f"Could not load dataset: {e}. Make sure you have access to 'large-traversaal/openbookqa_urdu'")
    
    datasets = {}
    
    # Check available splits
    print(f"Available splits: {dataset.keys()}")
    
    # Get the main split (usually 'train' or first available)
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
    
    # Convert to list for splitting
    data_list = []
    for i in range(len(full_data)):
        example = full_data[i]
        data_list.append(example)
    
    # Split into 80% train and 20% validation
    train_list, val_list = train_test_split(
        data_list,
        test_size=1 - train_split,
        random_state=random_state,
        shuffle=True,
    )
    
    datasets['train'] = train_list
    datasets['val'] = val_list
    print(f"✓ Train set (80%): {len(train_list)} examples")
    print(f"✓ Validation set (20%): {len(val_list)} examples")
    
    # Check if test split exists
    if 'test' in dataset:
        test_data = dataset['test']
        datasets['test'] = [test_data[i] for i in range(len(test_data))]
        print(f"✓ Test set: {len(datasets['test'])} examples")
    
    return datasets


def main():
    """Main training function"""
    print("=" * 60)
    print("mBERT LoRA Fine-tuning for Question Answering (OpenBookQA Urdu)")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    datasets = load_qa_data()
    
    if 'train' not in datasets:
        raise ValueError("Training data not found!")
    
    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME,
    )
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    # Preprocess datasets
    print("\nPreprocessing training data...")
    # Handle OpenBookQA format: multiple-choice QA converted to extractive QA
    train_examples = datasets['train']
    questions = []
    contexts = []
    answers_list = []
    
    for ex in train_examples:
        # OpenBookQA format: urdu_question_stem, urdu_choices, answerKey
        question = ex.get("urdu_question_stem", ex.get("question_stem", ex.get("question", "")))
        
        # Create context from choices (for extractive QA)
        urdu_choices = ex.get("urdu_choices", ex.get("choices", {}))
        if isinstance(urdu_choices, dict) and "text" in urdu_choices:
            choice_texts = urdu_choices["text"]
            choice_labels = urdu_choices.get("label", [])
            # Combine choices into a context
            context_parts = []
            for label, text in zip(choice_labels, choice_texts):
                context_parts.append(f"{label}: {text}")
            context = " ".join(context_parts)
        else:
            context = str(ex.get("context", ""))
        
        # Get the correct answer
        answer_key = ex.get("answerKey", "")
        if answer_key and isinstance(urdu_choices, dict) and "text" in urdu_choices:
            choice_labels = urdu_choices.get("label", [])
            choice_texts = urdu_choices["text"]
            # Find the answer text corresponding to answerKey
            answer_text = ""
            for label, text in zip(choice_labels, choice_texts):
                if label == answer_key:
                    answer_text = text
                    break
            # Create answer dict in SQuAD format
            if answer_text:
                answer_start = context.find(answer_text)
                answers = {
                    "text": [answer_text],
                    "answer_start": [answer_start if answer_start >= 0 else 0]
                }
            else:
                answers = {"text": [""], "answer_start": [0]}
        else:
            answers = ex.get("answers", {"text": [""], "answer_start": [0]})
        
        questions.append(question if question else "")
        contexts.append(context if context else "")
        answers_list.append(answers if answers else {"text": [""], "answer_start": [0]})
    
    train_tokenized = preprocess_qa_data(
        {"question": questions, "context": contexts, "answers": answers_list},
        tokenizer,
        max_length=MAX_LENGTH,
    )
    
    # Convert to dataset format
    class QADataset(Dataset):
        def __init__(self, tokenized_data):
            self.input_ids = tokenized_data["input_ids"]
            self.attention_mask = tokenized_data["attention_mask"]
            self.start_positions = tokenized_data["start_positions"]
            self.end_positions = tokenized_data["end_positions"]
            if "token_type_ids" in tokenized_data:
                self.token_type_ids = tokenized_data["token_type_ids"]
            else:
                self.token_type_ids = None
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            item = {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                "start_positions": torch.tensor(self.start_positions[idx], dtype=torch.long),
                "end_positions": torch.tensor(self.end_positions[idx], dtype=torch.long),
            }
            if self.token_type_ids is not None:
                item["token_type_ids"] = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
            return item
    
    train_dataset = QADataset(train_tokenized)
    
    eval_dataset = None
    if 'val' in datasets:
        print("Preprocessing validation data...")
        val_examples = datasets['val']
        val_questions = []
        val_contexts = []
        val_answers_list = []
        
        for ex in val_examples:
            question = ex.get("urdu_question_stem", ex.get("question_stem", ex.get("question", "")))
            
            urdu_choices = ex.get("urdu_choices", ex.get("choices", {}))
            if isinstance(urdu_choices, dict) and "text" in urdu_choices:
                choice_texts = urdu_choices["text"]
                choice_labels = urdu_choices.get("label", [])
                context_parts = []
                for label, text in zip(choice_labels, choice_texts):
                    context_parts.append(f"{label}: {text}")
                context = " ".join(context_parts)
            else:
                context = str(ex.get("context", ""))
            
            answer_key = ex.get("answerKey", "")
            if answer_key and isinstance(urdu_choices, dict) and "text" in urdu_choices:
                choice_labels = urdu_choices.get("label", [])
                choice_texts = urdu_choices["text"]
                answer_text = ""
                for label, text in zip(choice_labels, choice_texts):
                    if label == answer_key:
                        answer_text = text
                        break
                if answer_text:
                    answer_start = context.find(answer_text)
                    answers = {
                        "text": [answer_text],
                        "answer_start": [answer_start if answer_start >= 0 else 0]
                    }
                else:
                    answers = {"text": [""], "answer_start": [0]}
            else:
                answers = ex.get("answers", {"text": [""], "answer_start": [0]})
            
            val_questions.append(question if question else "")
            val_contexts.append(context if context else "")
            val_answers_list.append(answers if answers else {"text": [""], "answer_start": [0]})
        
        val_tokenized = preprocess_qa_data(
            {"question": val_questions, "context": val_contexts, "answers": val_answers_list},
            tokenizer,
            max_length=MAX_LENGTH,
        )
        eval_dataset = QADataset(val_tokenized)
    
    # Data collator - for QA tasks, we use default_data_collator that preserves start_positions and end_positions
    data_collator = default_data_collator
    
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
        metric_for_best_model="f1" if eval_dataset else None,  # Use F1 as best model metric
        greater_is_better=True,  # Higher F1 is better
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        seed=42,
        remove_unused_columns=False,  # Keep start_positions and end_positions
    )
    
    # Custom compute_loss for QA models
    def compute_loss(model, inputs, return_outputs=False):
        """
        Custom loss computation for QA models that use start_positions and end_positions
        """
        # Remove any 'labels' key if present
        inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_qa_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
    )
    
    # Override compute_loss to handle QA properly
    trainer.compute_loss = compute_loss
    
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
        print(f"Validation Exact Match: {eval_results.get('eval_exact_match', 'N/A'):.4f}")
        print(f"Validation F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
        
        # Save evaluation metrics
        with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    
    # Evaluate on test set if available
    if 'test' in datasets:
        print("\nEvaluating on test set...")
        test_examples = datasets['test']
        test_questions = []
        test_contexts = []
        test_answers_list = []
        
        for ex in test_examples:
            question = ex.get("urdu_question_stem", ex.get("question_stem", ex.get("question", "")))
            
            urdu_choices = ex.get("urdu_choices", ex.get("choices", {}))
            if isinstance(urdu_choices, dict) and "text" in urdu_choices:
                choice_texts = urdu_choices["text"]
                choice_labels = urdu_choices.get("label", [])
                context_parts = []
                for label, text in zip(choice_labels, choice_texts):
                    context_parts.append(f"{label}: {text}")
                context = " ".join(context_parts)
            else:
                context = str(ex.get("context", ""))
            
            answer_key = ex.get("answerKey", "")
            if answer_key and isinstance(urdu_choices, dict) and "text" in urdu_choices:
                choice_labels = urdu_choices.get("label", [])
                choice_texts = urdu_choices["text"]
                answer_text = ""
                for label, text in zip(choice_labels, choice_texts):
                    if label == answer_key:
                        answer_text = text
                        break
                if answer_text:
                    answer_start = context.find(answer_text)
                    answers = {
                        "text": [answer_text],
                        "answer_start": [answer_start if answer_start >= 0 else 0]
                    }
                else:
                    answers = {"text": [""], "answer_start": [0]}
            else:
                answers = ex.get("answers", {"text": [""], "answer_start": [0]})
            
            test_questions.append(question if question else "")
            test_contexts.append(context if context else "")
            test_answers_list.append(answers if answers else {"text": [""], "answer_start": [0]})
        
        test_tokenized = preprocess_qa_data(
            {"question": test_questions, "context": test_contexts, "answers": test_answers_list},
            tokenizer,
            max_length=MAX_LENGTH,
        )
        test_dataset = QADataset(test_tokenized)
        test_results = trainer.evaluate(test_dataset)
        print(f"Test Exact Match: {test_results.get('eval_exact_match', 'N/A'):.4f}")
        print(f"Test F1: {test_results.get('eval_f1', 'N/A'):.4f}")
        
        # Save test metrics
        with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

