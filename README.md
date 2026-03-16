# What Matters to an LLM? Behavioral and Computational Evidences from Summarization

This repository contains the code and resources for the paper [**"What Matters to an LLM? Behavioral and Computational Evidences from Summarization"**](https://arxiv.org/abs/2602.00459).

## Abstract

Large Language Models (LLMs) are now state-of-the-art at summarization, yet the internal notion of importance that drives their information selections remains hidden. We propose to investigate this by combining behavioral and computational analyses. 

**Behaviorally**, we generate a series of **length-controlled summaries** for each document and derive **empirical importance distributions** based on how often each information unit is selected. These reveal that LLMs converge on consistent importance patterns, sharply different from pre-LLM baselines, and that LLMs cluster more by family than by size. 

**Computationally**, we identify that certain attention heads align well with empirical importance distributions, and that middle-to-late layers are strongly predictive of importance. Together, these results provide initial insights into *what* LLMs prioritize in summarization and *how* this priority is internally represented, opening a path toward interpreting and ultimately controlling **information selection** in these models.

## 🔗 Data Availability

For reproducibility and ease of use, we provide the complete dataset on HuggingFace, including both the raw length-controlled summaries and the computed empirical importance distributions.

[**🤗 HuggingFace Dataset: yongxin2020/llm-importance-distributions**](https://huggingface.co/datasets/yongxin2020/llm-importance-distributions)

### Quick Load
You can load the data directly using the `datasets` library:

```python
from datasets import load_dataset

# Load entire dataset (all models, all splits)
dataset = load_dataset("yongxin2020/llm-importance-distributions")

# Load specific model subset (e.g., Llama-3.2-1B)
dataset = load_dataset(
    "yongxin2020/llm-importance-distributions", 
    data_files="meta-llama/Llama-3.2-1B-Instruct/**/*.json"
)
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yongxin2020/llm-importance-summarization
   cd llm-importance-summarization
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Copy the template configuration file and edit it to set your Python path and API keys (if using API models).
   ```bash
   cp config.json.template config.json
   nano config.json
   ```
   *Set `python_path` to your python executable (e.g., inside your virtualenv).*

**Prerequisites**:
- Python 3.8+
- PyTorch (tested with 2.0+)
- Transformers, Datasets, Accelerate
- 16GB+ RAM (32GB+ recommended for larger models)

## 🧪 Models & Datasets

We conducted experiments on the following models and datasets:

| Model Family | Models |
|--------------|--------|
| **Llama** | Llama-3.2-1B, Llama-3.1-8B |
| **Qwen** | Qwen2.5-1.5B, 3B, 7B, 14B |
| **DeepSeek** | DeepSeek-Chat (API) |

| Dataset | Type | Samples |
|---------|------|---------|
| **CNN/DailyMail** | News | 3,000 |
| **SAMSum** | Dialogue | 819 |
| **DECODA** | French Dialogue | 100 |

## 🚀 Reproduction Pipeline

The reproduction pipeline consists of four main stages. **Note that Step 1 and Step 2 can be skipped** if you use our pre-generated data from HuggingFace.

### Step 1: Data Generation (Optional)
Generate 10 length-controlled summaries per document for your target models. This establishes the "empirical importance" ground truth.

```bash
# Example: Generate summaries for Qwen2.5-7B on CNN/DailyMail
python scripts/_1_data_generation.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset cnn_dailymail \
    --save_fp "data/Qwen/Qwen2.5-7B-Instruct/cnn_dailymail/predictions.json"
```
*Output: `data/{model}/{dataset}/predictions.json`*

### Step 2: Preprocessing (Optional)
Calculate empirical importance distributions for each word based on its selection frequency across the generated summaries.

```bash
# Calculate word importance scores
python scripts/_2_data_preprocess.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail"
```
*Output: `data/{model}/{dataset}/generated_summaries_with_word_importance_deduplicated.json`*

### Step 3: Attention Analysis
Analyze the alignment between attention heads and the empirical importance distributions. This step identifies "Importance Heads" that focus on salient information.

```bash
# Extract attention weights and compute correlation metrics
python multi_head_attention/extract_attention_analysis.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail"
```
*Outputs include:*
- Spearman correlation scores for all attention heads
- NDCG@k metrics evaluating attention ranking quality
- Identification of the best-performing "Importance Heads"

### Step 4: Probing Analysis
We investigate how well the model's internal representations encode word importance through three distinct probing scenarios.

#### 1. Feature Extraction (Prerequisite)
First, extracting targeted hidden states for words with known importance scores. This is required for all probing scenarios.
```bash
# Extract hidden states for the target model
python probe_training/_0_targeted_hidden_extraction.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail" \
    --max_samples 1000
```

#### 2. Scenario A: Layer-wise Probing
Trains a separate probe for each layer to identify which specific layers best encode importance information.
```bash
# Train a probe for each layer individually
python probe_training/_1_train_targeted_word_importance_probe.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail" \
    comprehensive
```

#### 3. Scenario B: All-Layers Probing
Concatenates hidden states from all layers to train a powerful regression probe that utilizes the model's full capacity.
```bash
# Train a probe using concatenated features from all layers
python probe_training/_2_train_all_layers_word_importance_probe.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail" \
    comprehensive
```

#### 4. Scenario C: Article-Level Probing (Recommended)
Trains probes using a loss function calculated over the entire article distribution (e.g., KL Divergence). This approach captures the relative importance of words within their specific document context.
```bash
# Train with KL Divergence (Distribution Learning) - Recommended
python probe_training/_3_train_article_level_word_importance_probe.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail" \
    --loss_type kl \
    comprehensive

# Train with Mean Squared Error (Regression Learning)
python probe_training/_3_train_article_level_word_importance_probe.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "cnn_dailymail" \
    --loss_type mse \
    comprehensive
```

#### 5. Control Experiment: Randomized Weights
Validates that probe performance is due to learned knowledge, not just architectural induction biases, by training on initialized (untrained) weights.
```bash
# Run the full control experiment (extracts randomized features + trains probes)
# Note: Requires bash environment or manual execution of steps
./probe_training/run_control_experiment.sh "Qwen/Qwen2.5-7B-Instruct" "cnn_dailymail" "small" 100
```

## 📂 Repository Structure

- `scripts/`: Core pipeline for data generation and preprocessing.
- `multi_head_attention/`: Code for attention extraction and correlation analysis.
- `probe_training/`: Code for extracting hidden states and training linear probes.
- `behavioral_analysis_results/`: Stores output metrics from behavioral consistency checks.
- `data/`: Local storage for datasets (please use HuggingFace for the full release data).

## 📜 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@misc{zhou2026mattersllmbehavioralcomputational,
      title={What Matters to an LLM? Behavioral and Computational Evidences from Summarization}, 
      author={Yongxin Zhou and Changshun Wu and Philippe Mulhem and Didier Schwab and Maxime Peyrard},
      year={2026},
      eprint={2602.00459},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.00459}, 
}
```
