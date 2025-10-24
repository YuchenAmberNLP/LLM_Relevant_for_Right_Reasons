# Relevant for the Right Reasons? Investigating Lexical Biases in Re-ranking with Large Language Models 

This repository contains the code, data and evaluation scripts to reporduce the results of the paper [Relevant for the Right Reasons? Investigating Lexical Biases in Re-ranking with Large Language Models], which has been accepted as _EMNLP 2025 Multilingual Representation Learning Workshop.


## Overview
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Model Training & Reranking & Evaluation](#model-training--reranking--evaluation)
4. [Overlap Correlation Calculation](#overlap-correlation-calculation)
5. [Lexical Perturbation Experiments](#lexical-perturbation-experiments)



## Installation
### (1) Download, unzip and enter the project folder
```bash
cd LLM_Right_for_Right_Reasons
```
### (2) Install dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation
To prepare multilingual MS MARCO and XQuAD-R/SQuAD datasets for MoIR and CLIR re-ranking experiments, run:
```bash
bash scripts/data_prep_mmarco.sh
bash scripts/data_prep_xquad.sh
```
If you would like to generate instruction-tuning data for model fine-tuning, use:
```bash
bash scripts/data_prep_mmarco_tune.sh
bash scripts/data_prep_squad_tune.sh
```

## Model Training & Reranking & Evaluation

This project is designed to support training and evaluating re-ranking models across **multiple languages**, including both monolingual and cross-lingual settings.

### Instruction Tuning
We provide multiple groups of instruction-tuning data, which have already been generated during the Data Preparationstage and stored in:

- `instruction/mmarco/` – for training on MS MARCO-style data across language pairs listed in `data/mmarco_langpair.txt`, as well as `enxx` and `xxxx`
- `instruction/squad/` – for training on MS MARCO-style data across language pairs listed in `data/xquad_langpair.txt`, as well as `enxx` and `xxxx`

To train a model, you can use the following command:

```bash
cd stanford_alpaca

torchrun --nproc_per_node=4 --master_port=29502 train_sft.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --data_path <PATH_TO_TRAINING_INSTRUCTION_DATA_JSON> \
    --bf16 True \
    --output_dir <PATH_TO_SAVE_FINETUNED_MODEL> \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --logging_strategy steps \
    --logging_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 12 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
```
Simply replace `<PATH_TO_TRAINING_INSTRUCTION_DATA_JSON>` with the path to the relevant instruction-tuning data (e.g., from `instruction/mmarco/` and `instruction/squad/`) and `<PATH_TO_SAVE_FINETUNED_MODEL>` with your desired output directory.

We also release our instruction-tuned models on [HuggingFace] (https://huggingface.co/ycmaonlp), allowing users to directly download and use them without additional training.

### Reranking
After training, you can run pairwise re-ranking on top-N candidates using one of the following scripts:

Using the original LLaMA3.1-8B-Instruct model without instruction tuning:
```bash
python pairwise_reranker_Llama3.py \
  --topn_path <PATH_TO_TOPN_TSV_FILE> \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --k 10 \
  --num_gpu 4
```
Using the instruction-tuned model:
```bash
python pairwise_reranker_tuned.py \
  --topn_path <PATH_TO_TOPN_TSV_FILE> \
  --model_name <PATH_TO_FINETUNED_MODEL> \
  --k 10 \
  --num_gpu 4
```
The `--topn_path` should point to a TSV file of retrieved candidate passages (e.g., from MS MARCO or XQuAD).  
These top-N files are already generated during the Data Preparation stage and stored under:
- `data/cl_topn/` – for MS MARCO-style queries across multilingual and cross-lingual language pairs
- `data/xquad-topn/` – for XQuAD-style queries used in multilingual evaluation

Each file corresponds to a specific language pair or evaluation setting.

### Evaluation

After reranking, you can compute IR metrics such as MRR@10 by running:

```bash
python ir_eval.py \
  --run_path <PATH_TO_RERANK_RESULTS_TSV>
```
The re-ranking result files are automatically saved in the `results/` directory during the reranking step.
Each result file contains query–passage pairs sorted by predicted relevance scores, and can be directly evaluated using this script.


## Overlap Correlation Calculation
To compute metrics **ALOP** and **AP-LOD correlation**, simply run:

```bash
python overlap_corr.py
```

This script calculates lexical overlap between query–document pairs and correlates it with reranking effectiveness.

## Lexical Perturbation Experiments

This section examines whether models rely on lexical overlap instead of semantic relevance when judging document relevance.
We conduct controlled experiments using synthetic data automatically generated by GPT-5, with all prompt templates provided in: `data_prep/causal_prompts.txt`.


### Evaluate on Original Synthetic Data
All original synthetic datasets are generated by GPT-5, then filtered and randomly shuffled using `data_prep/sync_data_preprocess.py`. The final processed data are available in: `data/synthetic_causal_data/original/`.

There are two files: one containing lexical–semantic conflict cases (queries overlap with irrelevant passages), and one containing non-conflict cases (queries overlap with relevant passages).

Run the following command to evaluate model performance on the original data:
```bash
python pairwise_classification_HF.py \
  --model_name <MODEL_NAME_OR_PATH> \
  --input_path <PATH_TO_ORIGINAL_JSONL>
```
Results will be saved in the `causal_results/` folder by default.
For each model, individual result files are generated, including the corresponding error cases (FP) and correct cases (TP) for later perturbation.

### Generate Perturbed Data
We then prompt GPT-5 to remove or modify overlapping keywords in the stored FP and TP cases while keeping the semantic meaning unchanged. 

We provide GPT-5–generated perturbed datasets, created based on our model’s predictions on the original data. These datasets are available in: `data/synthetic_causal_data/perturbed/`.
Users can also generate their own perturbed data according to their model results on the original datasets.

### Evaluate After Perturbation
Run the same classification script on the perturbed data:
```bash
python pairwise_classification_HF.py \
  --model_name <MODEL_NAME_OR_PATH> \
  --input_path <PATH_TO_PERTURBED_JSONL>
```
This evaluation computes overall accuracy, which corresponds to different metrics depending on the dataset:

- On **lexical–semantic non-conflict** data, accuracy represents the **Retention Rate**.  
- On **lexical–semantic conflict** data, accuracy represents the **Recovery Rate**.  

All results and detailed case logs are automatically saved in `causal_results/`.


## License
This repository is released under the MIT License.

## Note:

This project's training script is based on and modified from **[Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)**, See https://github.com/tatsu-lab/stanford_alpaca for details.


The following public datasets were used in this project:

- **SQuAD** (Stanford Question Answering Dataset)
- **XQuAD-R** (a retrieval version of the XQuAD dataset (a cross-lingual extractive QA dataset).)
- **Multilingual MS MARCO** (Microsoft Machine Reading Comprehension)
- **CSCLIR** (Code-switched Cross-Lingual IR) from ([Litschko et al., 2023](https://aclanthology.org/2023.findings-acl.193/))

Dataset usage complies with the respective licenses or terms provided by their authors and publishers.

For any questions regarding reuse or extension, please contact the author.
