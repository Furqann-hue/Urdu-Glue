# 1. Urdu GLUE: Datasets & Fine-Tuning Code

This repository contains Urdu GLUE datasets curated by us, along with training code for multilingual transformer models. It supports benchmarking mBERT and XLM-RoBERTa on core Urdu NLU tasks using standard and parameter-efficient fine-tuning approaches.

## 2. Contents

### 2.1 Datasets (Urdu GLUE)

We provide four Urdu GLUE benchmark datasets covering grammaticality judgment, semantic textual similarity, natural language inference, and commonsense reasoning.

#### 2.1.1 U-CoLA
- Task: Grammatical acceptability classification
- Labels:
  - 0 (Ungrammatical)
  - 1 (Grammatical)

#### 2.1.2 U-STS-B
- Task: Semantic textual similarity
- Type: Regression
- Score Range: 0–5

#### 2.1.3 U-WNLI
- Task: Pronoun resolution / inference
- Labels:
  - 0 (Not entailment)
  - 1 (Entailment)

#### 2.1.4 U-XNLI
- Task: Natural language inference
- Labels:
  - entailment
  - neutral
  - contradiction

### 2.2 Models
- mBERT (Multilingual BERT)
- XLM-RoBERTa

### 2.3 Training Code
- Standard fine-tuning
- LoRA
- QLoRA
- ADAPT

All methods are implemented for both mBERT and XLM-RoBERTa.

ADAPT (Adaptive Dynamic Template Training) is a prompt-based fine-tuning strategy for masked language models that mitigates sensitivity to template selection by treating templates as stochastic training variables and incorporating multiple candidate templates during training.

## 3. Purpose
- Enable Urdu NLP benchmarking using GLUE-style tasks
- Support research in low-resource and multilingual settings
- Compare full fine-tuning with parameter-efficient and prompt-based methods
