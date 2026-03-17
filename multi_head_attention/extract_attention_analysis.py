"""
This script:
1. Extracts attention weights from specified layers and heads of a model
2. Builds reference importance distributions from word_importance annotations
3. Handles token-word alignment to avoid mismatches
4. Computes KL divergence, Spearman correlation, NDCG, and IR metrics between attention weights and reference importance
5. Visualizes results to identify if attention weights reflect information importance


**Function:** `extract_attention_weights(model, tokenizer, text, layers_to_extract)`

**Attention Tensor Shape:**
```python
# attention[h, i, j] = attention from query position i to key position j
attention_weights: [num_heads, seq_len, seq_len]
```

**Method: Attention RECEIVED**
```python
# Sum attention received by each token across all query positions
attention_received = attention_weights.sum(dim=1)  # [num_heads, seq_len]
# Average across heads
avg_attention = attention_received.mean(dim=0)  # [seq_len]
```
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy, spearmanr
import math
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
import re
import unicodedata
from transformers import BitsAndBytesConfig
from metrics_extra import ndcg_at_k, renyi_divergence, precision_at_k_from_binary, recall_at_k_from_binary
warnings.filterwarnings('ignore')


def normalize_text(text: str, language: str = 'en') -> str:
    """
    Normalize text for matching (lowercase, strip, remove punctuation).
    IMPORTANT: Preserves apostrophes for contractions (can't, betty's, etc.)
    For French, also normalizes accents.
    """
    text = text.lower().strip()
    # Replace smart quotes with regular apostrophes
    text = text.replace(''', "'").replace(''', "'").replace("`", "'")
    # Remove punctuation EXCEPT apostrophes (keep ' for contractions)
    text = re.sub(r'[^\w\s\']', '', text)
    
    # For French, normalize accents
    if language == 'fr':
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')
    
    return text


def get_layer_indices(model_config, layer_selection: str = 'representative') -> List[int]:
    """
    Get layer indices based on selection strategy.
    
    Args:
        model_config: Model configuration
        layer_selection: Strategy for layer selection
            - 'representative': Early, middle, late (3 layers)
            - 'all': All layers
            - 'custom': Specify custom layers (pass as comma-separated string)
    
    Returns:
        List of layer indices to analyze
    """
    total_layers = model_config.num_hidden_layers
    
    if layer_selection == 'all':
        return list(range(total_layers))
    elif layer_selection == 'representative':
        early = 0
        middle = total_layers // 2
        late = total_layers - 1
        return [early, middle, late]
    else:
        # For custom selection, parse comma-separated string
        try:
            return [int(x.strip()) for x in layer_selection.split(',')]
        except:
            # Fallback to representative
            early = 0
            middle = total_layers // 2
            late = total_layers - 1
            return [early, middle, late]


def align_tokens_to_words(text: str, tokenizer) -> Tuple[List[str], List[List[int]]]:
    """
    Align tokens to words in the original text.
    
    This is crucial to avoid token-word mismatch issues.
    
    Args:
        text: Original text
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Tuple of (words, token_indices_per_word)
        - words: List of words in the text
        - token_indices_per_word: List of lists, each containing token indices for a word
    """
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokens = encoding['input_ids'][0].tolist()
    
    # Decode each token
    token_strings = [tokenizer.decode([t]).strip() for t in tokens]
    
    # Split text into words (simple whitespace split)
    words = text.split()
    
    # Align tokens to words
    word_to_token_indices = []
    token_idx = 0
    
    for word in words:
        word_tokens = []
        word_lower = normalize_text(word)
        
        # Accumulate tokens that form this word
        accumulated_text = ""
        while token_idx < len(token_strings):
            accumulated_text += token_strings[token_idx].replace(' ', '').replace('▁', '')
            word_tokens.append(token_idx)
            token_idx += 1
            
            # Check if we've accumulated enough to match the word
            accumulated_normalized = normalize_text(accumulated_text)
            if accumulated_normalized == word_lower:
                break
            elif len(accumulated_normalized) >= len(word_lower):
                # Overshot, might be a mismatch - keep the tokens we have
                break
        
        if word_tokens:
            word_to_token_indices.append(word_tokens)
    
    return words, word_to_token_indices


def build_reference_distribution_word_level(text: str, word_importance: Dict[str, float], 
                                             tokenizer, language: str = 'en') -> Tuple[List[str], np.ndarray, List[List[int]]]:
    """
    Build reference importance distribution at WORD level (not token level).
    
    This fixes the token-word mismatch issue.
    
    Args:
        text: Input text (dialogue or article)
        word_importance: Dict mapping words to importance scores
        tokenizer: Tokenizer to align tokens with words
    
    Returns:
        Tuple of (unique_words, normalized_scores, token_indices_per_word)
    """
    # Align tokens to words
    words, word_to_token_indices = align_tokens_to_words(text, tokenizer)
    
    # Normalize word_importance keys for consistent matching
    normalized_word_importance = {}
    for key, score in word_importance.items():
        normalized_key = normalize_text(key, language)
        normalized_word_importance[normalized_key] = score
    
    # Build word-score pairs, keeping only first occurrence of each word
    seen_words = {}
    unique_words = []
    unique_token_indices = []
    
    for word, token_indices in zip(words, word_to_token_indices):
        normalized_word = normalize_text(word, language)
        
        if normalized_word and normalized_word not in seen_words:
            # Check if word exists in normalized word_importance
            score = normalized_word_importance.get(normalized_word, 0.0)
            
            seen_words[normalized_word] = score
            unique_words.append(normalized_word)
            unique_token_indices.append(token_indices)
    
    # Convert to arrays
    scores = np.array([seen_words[w] for w in unique_words])
    
    # Normalize to probability distribution (sum to 1)
    if scores.sum() > 0:
        normalized_scores = scores / scores.sum()
    else:
        # Uniform distribution if no scores
        normalized_scores = np.ones(len(scores)) / len(scores)
    
    return unique_words, normalized_scores, unique_token_indices


def extract_attention_weights(model, tokenizer, text: str, 
                               layers_to_extract: List[int]) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights from specified layers.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        layers_to_extract: List of layer indices to extract from
    
    Returns:
        Dict mapping layer_idx -> attention tensor [num_heads, seq_len, seq_len]
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention weights for specified layers
    attention_weights = {}
    for layer_idx in layers_to_extract:
        # outputs.attentions is a tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
        attn = outputs.attentions[layer_idx].squeeze(0)  # Remove batch dimension
        attention_weights[layer_idx] = attn.cpu()
    
    return attention_weights


def aggregate_attention_for_words(attention_weights: torch.Tensor, 
                                   token_indices_per_word: List[List[int]],
                                   aggregation: str = 'mean') -> np.ndarray:
    """
    Aggregate attention weights at WORD level (not token level).
    
    This is the key fix for token-word mismatch.
    
    Args:
        attention_weights: Attention tensor [num_heads, seq_len, seq_len]
        token_indices_per_word: List of token indices for each word
        aggregation: 'max', 'min', or 'mean' for aggregating across tokens within a word
    
    Returns:
        Attention per word [num_heads, num_words]
    """
    num_heads, seq_len, _ = attention_weights.shape
    num_words = len(token_indices_per_word)
    
    # For each head, aggregate attention received by each WORD
    attention_per_head = np.zeros((num_heads, num_words))
    
    for head_idx in range(num_heads):
        # Sum attention received from all query positions for each token
        attention_received = attention_weights[head_idx].sum(dim=0).numpy()  # [seq_len]
        
        # Aggregate attention for each word
        for word_idx, token_indices in enumerate(token_indices_per_word):
            if not token_indices:
                continue
                
            # Get attention for all tokens in this word
            token_attentions = [attention_received[idx] for idx in token_indices if idx < len(attention_received)]
            
            if token_attentions:
                if aggregation == 'mean':
                    attention_per_head[head_idx, word_idx] = np.mean(token_attentions)
                elif aggregation == 'max':
                    attention_per_head[head_idx, word_idx] = np.max(token_attentions)
                elif aggregation == 'min':
                    attention_per_head[head_idx, word_idx] = np.min(token_attentions)
    
    # Normalize each head's distribution (sum to 1)
    for head_idx in range(num_heads):
        if attention_per_head[head_idx].sum() > 0:
            attention_per_head[head_idx] /= attention_per_head[head_idx].sum()
    
    return attention_per_head


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two distributions.
    
    KL(P || Q) = sum(P * log(P / Q))
    
    Args:
        p: Reference distribution (importance scores)
        q: Predicted distribution (attention weights)
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence score (lower is better)
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Renormalize after clipping
    p = p / p.sum()
    q = q / q.sum()
    
    return entropy(p, q)


def compute_kl_vs_uniform_baseline(p: np.ndarray, epsilon: float = 1e-10) -> dict:
    """
    Compute KL divergence between reference distribution and uniform distribution.
    This serves as a baseline to understand "how non-uniform is the true importance?"
    
    Args:
        p: Reference distribution (importance scores)
        epsilon: Small value to avoid log(0)
    
    Returns:
        Dictionary with baseline statistics
    """
    p = np.clip(p, epsilon, 1.0)
    p_norm = p / p.sum()
    n = len(p)
    
    # Uniform distribution
    uniform = np.ones(n) / n
    
    # KL(p || uniform) = sum(p * log(p / uniform))
    # Equivalent to: log(n) - H(p) where H(p) is entropy of p
    entropy_p = -np.sum(p_norm * np.log(p_norm + epsilon))
    kl_vs_uniform = np.log(n) - entropy_p
    
    # Alternative direct calculation for verification
    kl_direct = np.sum(p_norm * (np.log(p_norm + epsilon) - np.log(uniform + epsilon)))
    
    return {
        'kl_vs_uniform': kl_vs_uniform,
        'kl_direct': kl_direct,  # Should match kl_vs_uniform
        'entropy_reference': entropy_p,
        'max_entropy': np.log(n),
        'concentration_ratio': 1.0 - (entropy_p / np.log(n)),  # How concentrated vs uniform
        'n_tokens': n,
        'uniform_entropy': np.log(n)
    }


def compute_normalized_kl_metrics(attention_kl: float, reference_dist: np.ndarray, 
                                  verbose: bool = False) -> dict:
    """
    Compute normalized KL metrics using uniform baseline.
    
    Args:
        attention_kl: KL divergence between attention and reference
        reference_dist: Reference importance distribution
        verbose: Whether to include detailed breakdown
    
    Returns:
        Dictionary with normalized metrics and baseline info
    """
    baseline_stats = compute_kl_vs_uniform_baseline(reference_dist)
    kl_uniform = baseline_stats['kl_vs_uniform']
    
    # Normalized KL: how much better than uniform
    normalized_kl = 1.0 - (attention_kl / kl_uniform) if kl_uniform > 1e-10 else float('nan')
    
    # Improvement ratio: how many times better than uniform
    improvement_ratio = kl_uniform / attention_kl if attention_kl > 1e-10 else float('inf')
    
    result = {
        'attention_kl': attention_kl,
        'baseline_kl_uniform': kl_uniform,
        'normalized_kl': normalized_kl,
        'improvement_ratio': improvement_ratio,
        'reference_concentration': baseline_stats['concentration_ratio'],
        'reference_entropy': baseline_stats['entropy_reference'],
        'max_possible_entropy': baseline_stats['max_entropy']
    }
    
    if verbose:
        result.update(baseline_stats)
    
    return result


def compute_spearman_correlation(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between two distributions.
    
    This is more robust to distribution shape differences than KL divergence.
    Measures if attention and importance have similar RANKING of tokens.
    
    Args:
        p: Reference values (importance scores)
        q: Predicted values (attention weights)
    
    Returns:
        Spearman correlation coefficient (higher is better, range: -1 to 1)
    """
    if len(p) < 2:
        return 0.0
    
    try:
        corr, _ = spearmanr(p, q)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def analyze_attention_importance(model_name: str, 
                                  dataset_name: str,
                                  data_path: str,
                                  num_samples: int = 100,
                                  aggregation: str = 'mean',
                                  output_dir: str = './attention_analysis_results',
                                  test_heads_only: bool = False,
                                  filter_zero_importance: bool = False,
                                  layer_selection: str = 'all',
                                  save_detailed_log: bool = True,
                                  quantization: str = None):
    """
    Main analysis function with improvements.
    
    Args:
        model_name: Name of the model (e.g., 'Llama-3.2-1B-Instruct')
        dataset_name: Name of the dataset (e.g., 'samsum', 'cnn_dailymail')
        data_path: Path to the JSON data file
        num_samples: Number of samples to process
        aggregation: Method to aggregate attention ('max', 'min', 'mean')
        output_dir: Directory to save results
        test_heads_only: If True, only analyze first and last heads
        filter_zero_importance: If True, exclude words with zero importance
        layer_selection: Layer selection strategy ('all', 'representative', or custom)
        save_detailed_log: If True, save detailed log for first sample
        quantization: Quantization mode ('8bit', '4bit', or None)
    """
    print("="*80)
    print("ATTENTION-IMPORTANCE ANALYSIS (IMPROVED)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"Layer selection: {layer_selection}")
    print(f"Aggregation: {aggregation}")
    print(f"Filter zero-importance: {filter_zero_importance}")
    if quantization:
        print(f"Quantization: {quantization}")
    print("="*80)
    
    # Create output directory
    output_dir = os.path.join(output_dir, model_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("\n[1/6] Loading model and tokenizer...")
    # Determine model path based on model name
    if "Llama" in model_name:
        model_path = f"meta-llama/{model_name}"
    elif "Qwen" in model_name:
        model_path = f"Qwen/{model_name}"
    else:
        model_path = model_name  # Fallback to exact name
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure quantization
    quantization_config = None
    if quantization == '8bit':
        print("  ✓ Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif quantization == '4bit':
        print("  ✓ Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Use appropriate attention implementation based on model architecture
    # Qwen models use Sliding Window Attention which requires specific implementations
    if "Qwen" in model_name:
        # Qwen: Try flash_attention_2 first (best for SWA), fallback to float32+eager
        try:
            print("  Attempting to load with flash_attention_2...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config
            )
            print("  ✓ Using flash_attention_2 with float16")
        except Exception as e:
            print(f"  Flash attention not available, using float32 for numerical stability")
            # CRITICAL: Qwen + eager mode + float16 causes NaN in attention weights
            # Solution: Use float32 which provides numerical stability
            # If quantization is enabled, we must use float16 for compute type usually
            dtype = torch.float32 if quantization_config is None else torch.float16
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                quantization_config=quantization_config
            )
            print(f"  ✓ Using eager mode with {dtype} (prevents NaN in attention)")
            if quantization_config is None:
                print("  ⚠️  Note: float32 uses more memory but ensures accurate attention extraction")
    else:
        # Llama and others: Use eager mode (suppresses attention warnings)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            quantization_config=quantization_config
        )
    model.eval()
    
    # Get model info
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"  ✓ Model loaded: {num_layers} layers, {num_heads} heads per layer")
    
    # Get layers to extract
    layers_to_extract = get_layer_indices(model.config, layer_selection)
    
    # Generate layer names
    if len(layers_to_extract) == num_layers:
        layer_names = [f'L{i}' for i in layers_to_extract]
    elif len(layers_to_extract) == 3:
        layer_names = ['early', 'middle', 'late']
    else:
        layer_names = [f'L{i}' for i in layers_to_extract]
    
    print(f"  ✓ Extracting from {len(layers_to_extract)} layers: {layers_to_extract}")
    
    # Determine which heads to analyze
    if test_heads_only:
        heads_to_analyze = [0, num_heads - 1]
        print(f"  ✓ TEST MODE: Only analyzing heads: {heads_to_analyze}")
    else:
        heads_to_analyze = list(range(num_heads))
        print(f"  ✓ Analyzing ALL heads: 0 to {num_heads - 1}")
    
    # Load data
    print(f"\n[2/6] Loading data...")
    print(f"  Path: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Get text field and language based on dataset
    if dataset_name in ['samsum', 'decoda']:
        text_field = 'dialogue'
        language = 'fr' if dataset_name == 'decoda' else 'en'
    else:
        text_field = 'article'
        language = 'en'
    
    print(f"  ✓ Text field: '{text_field}', Language: {language}")
    
    # Limit to num_samples
    data = data[:num_samples]
    print(f"  ✓ Processing {len(data)} samples")
    
    # Storage for results
    kl_scores = {
        layer_idx: np.zeros((num_heads, len(data)))
        for layer_idx in layers_to_extract
    }
    spearman_scores = {
        layer_idx: np.zeros((num_heads, len(data)))
        for layer_idx in layers_to_extract
    }
    # Store per-sample Spearman p-values as well
    spearman_pvalues = {
        layer_idx: np.ones((num_heads, len(data))) * np.nan
        for layer_idx in layers_to_extract
    }
    # Extra per-sample top-k and reweighted metrics (option A)
    # We'll store per-sample, per-layer, per-head top-k indices (k in k_list)
    k_list = [1, 3, 5, 10]
    topk_indices_store = {
        layer_idx: [[[] for _ in range(num_heads)] for _ in range(len(data))]
        for layer_idx in layers_to_extract
    }
    # Storage for power-weighted KL and renyi per-sample (multiple alpha values)
    pw_kl_scores = {
        layer_idx: np.zeros((num_heads, len(data)))
        for layer_idx in layers_to_extract
    }
    # Storage for NDCG scores (at k=5 and k=10)
    ndcg_k_values = [5, 10]
    ndcg_scores = {
        k: {
            layer_idx: np.zeros((num_heads, len(data)))
            for layer_idx in layers_to_extract
        }
        for k in ndcg_k_values
    }
    # Storage for Precision and Recall at k (k=1,3,5,10)
    k_list = [1, 3, 5, 10]
    precision_scores = {
        k: {
            layer_idx: np.zeros((num_heads, len(data)))
            for layer_idx in layers_to_extract
        }
        for k in k_list
    }
    recall_scores = {
        k: {
            layer_idx: np.zeros((num_heads, len(data)))
            for layer_idx in layers_to_extract
        }
        for k in k_list
    }
    # Store Rényi for multiple alpha values: 1.5, 2, 3
    renyi_alpha_values = [1.5, 2.0, 3.0]
    renyi_scores_dict = {
        alpha: {
            layer_idx: np.zeros((num_heads, len(data)))
            for layer_idx in layers_to_extract
        }
        for alpha in renyi_alpha_values
    }
    
    valid_samples = []
    skipped_samples = []
    # Per-sample reference entropy and KL vs uniform (per-sample, independent of layer/head)
    per_sample_ref_entropy = [None] * len(data)
    per_sample_kl_uniform = [None] * len(data)
    per_sample_ref_distribution = [None] * len(data)
    
    # Process each sample
    print(f"\n[3/6] Processing samples...")
    for sample_idx, item in enumerate(tqdm(data, desc="Analyzing")):
        text = item[text_field]
        word_importance = item['word_importance']
        
        try:
            # Build reference distribution at WORD level (key improvement!)
            unique_words, ref_distribution, token_indices_per_word = build_reference_distribution_word_level(
                text, word_importance, tokenizer, language
            )
            
            if len(unique_words) == 0:
                if sample_idx < 3:  # Only print for first few samples to avoid spam
                    print(f"  ⚠️  Sample {sample_idx} skipped: no words matched")
                    print(f"      Text words (first 10): {[normalize_text(w, language) for w in text.split()[:10]]}")
                    print(f"      Word importance keys (first 10): {list(word_importance.keys())[:10]}")
                    print(f"      Normalized importance keys (first 10): {list(normalized_word_importance.keys())[:10]}")
                skipped_samples.append(sample_idx)
                continue
            
            # Filter zero-importance words if requested
            if filter_zero_importance:
                non_zero_mask = ref_distribution > 0
                if non_zero_mask.sum() == 0:
                    skipped_samples.append(sample_idx)
                    continue
                unique_words = [w for w, mask in zip(unique_words, non_zero_mask) if mask]
                ref_distribution = ref_distribution[non_zero_mask]
                token_indices_per_word = [idx for idx, mask in zip(token_indices_per_word, non_zero_mask) if mask]
                # Renormalize
                ref_distribution = ref_distribution / ref_distribution.sum()
            
            # Extract attention weights
            attention_weights = extract_attention_weights(
                model, tokenizer, text, layers_to_extract
            )
            
            # Store attention per head for all layers (for detailed logging)
            attention_per_head_dict = {}
            kl_scores_sample = {}
            spearman_scores_sample = {}
            
            # For each layer and head, compute KL divergence
            for layer_idx in layers_to_extract:
                attn = attention_weights[layer_idx]  # [num_heads, seq_len, seq_len]
                
                # Get attention distribution per head at WORD level (key improvement!)
                attention_per_head = aggregate_attention_for_words(
                    attn, token_indices_per_word, aggregation
                )
                attention_per_head_dict[layer_idx] = attention_per_head
                
                # Compute KL divergence and Spearman correlation for each head
                layer_kl_scores = []
                layer_spearman_scores = []
                
                # Compute uniform baseline for this sample (once per sample)
                baseline_stats = compute_kl_vs_uniform_baseline(ref_distribution)
                
                # Store per-sample baseline info
                per_sample_ref_distribution[sample_idx] = ref_distribution.tolist()
                per_sample_kl_uniform[sample_idx] = baseline_stats['kl_vs_uniform']
                
                # Log detailed baseline calculation for first few samples
                if sample_idx < 3:  # Log first 3 samples
                    print(f"    Sample {sample_idx+1} baseline calculation:")
                    print(f"      Reference entropy: {baseline_stats['entropy_reference']:.3f}")
                    print(f"      Max entropy (uniform): {baseline_stats['max_entropy']:.3f}")
                    print(f"      KL(ref||uniform): {baseline_stats['kl_vs_uniform']:.3f}")
                    print(f"      Concentration ratio: {baseline_stats['concentration_ratio']:.3f}")
                
                for head_idx in heads_to_analyze:
                    attn_dist = attention_per_head[head_idx]
                    kl = compute_kl_divergence(ref_distribution, attn_dist)
                    
                    # Compute normalized metrics with detailed breakdown
                    normalized_metrics = compute_normalized_kl_metrics(
                        kl, ref_distribution, verbose=(sample_idx < 3 and head_idx < 2)
                    )
                    
                    # Log detailed metrics for first few samples and heads
                    if sample_idx < 3 and head_idx < 2:
                        print(f"      Head {head_idx}: KL={kl:.3f}, Normalized={normalized_metrics['normalized_kl']:.3f}, " +
                              f"Improvement={normalized_metrics['improvement_ratio']:.2f}x better than uniform")
                    
                    # Get Spearman rho and p-value
                    try:
                        rho, pval = spearmanr(ref_distribution, attn_dist)
                        if np.isnan(rho):
                            rho = 0.0
                    except Exception:
                        rho, pval = 0.0, 1.0
                    kl_scores[layer_idx][head_idx, sample_idx] = kl
                    spearman_scores[layer_idx][head_idx, sample_idx] = float(rho)
                    spearman_pvalues[layer_idx][head_idx, sample_idx] = float(pval) if pval is not None else 1.0
                    layer_kl_scores.append(kl)
                    layer_spearman_scores.append(rho)
                    
                    # --- Option A: compute top-k and reweighted divergences ---
                    # Predict top-k indices by attention mass
                    pred_order = np.argsort(attn_dist)[::-1]
                    # store top-k indices (only up to k=10)
                    topk = pred_order[:max(k_list)].tolist()
                    topk_indices_store[layer_idx][sample_idx][head_idx] = topk

                    # Power-weighted KL (gamma=2)
                    p = ref_distribution
                    q = attn_dist
                    # avoid zero issues
                    p_w = p ** 2
                    p_w = p_w / (p_w.sum() + 1e-12)
                    pwkl = compute_kl_divergence(p_w, q)
                    pw_kl_scores[layer_idx][head_idx, sample_idx] = pwkl

                    # NDCG at k
                    for k_val in ndcg_k_values:
                        ndcg_val = ndcg_at_k(attn_dist, ref_distribution, k_val)
                        ndcg_scores[k_val][layer_idx][head_idx, sample_idx] = ndcg_val

                    # Precision and Recall at k (using binary relevance: importance > median)
                    median_importance = np.median(ref_distribution)
                    ref_binary = (ref_distribution > median_importance).astype(int)
                    
                    for k_val in k_list:
                        if k_val <= len(pred_order):
                            pred_topk_idx = pred_order[:k_val]
                            prec_val = precision_at_k_from_binary(pred_topk_idx, ref_binary)
                            rec_val = recall_at_k_from_binary(pred_topk_idx, ref_binary)
                            
                            precision_scores[k_val][layer_idx][head_idx, sample_idx] = prec_val
                            recall_scores[k_val][layer_idx][head_idx, sample_idx] = rec_val

                    # Rényi divergence for multiple alpha values (alpha > 1)
                    for alpha in renyi_alpha_values:
                        renyi_val = renyi_divergence(p, q, alpha)
                        renyi_scores_dict[alpha][layer_idx][head_idx, sample_idx] = renyi_val
                
                kl_scores_sample[layer_idx] = np.array(layer_kl_scores)
                spearman_scores_sample[layer_idx] = np.array(layer_spearman_scores)
            
            # Save detailed log for the first sample
            if save_detailed_log and sample_idx == 0:
                print(f"\n  💾 Saving detailed log for sample 0...")
                log_path = os.path.join(output_dir, "detailed_sample_log.txt")
                with open(log_path, 'w') as f:
                    f.write(f"Detailed Analysis for Sample 0\n")
                    f.write(f"Text: {text[:200]}...\n\n")
                    f.write(f"Reference Distribution (Top 10):\n")
                    top_indices = np.argsort(ref_distribution)[::-1][:10]
                    for idx in top_indices:
                        f.write(f"  {unique_words[idx]}: {ref_distribution[idx]:.4f}\n")
                    f.write("\n")
                    
                    for layer_idx in layers_to_extract:
                        f.write(f"Layer {layer_idx}:\n")
                        for head_idx in heads_to_analyze:
                            kl = kl_scores_sample[layer_idx][head_idx] if head_idx < len(kl_scores_sample[layer_idx]) else 0
                            spearman = spearman_scores_sample[layer_idx][head_idx] if head_idx < len(spearman_scores_sample[layer_idx]) else 0
                            f.write(f"  Head {head_idx}: KL={kl:.4f}, Spearman={spearman:.4f}\n")
                print(f"  ✓ Detailed sample log saved to: {log_path}")
            
            valid_samples.append(sample_idx)
            
        except Exception as e:
            # Check for OOM
            if "CUDA out of memory" in str(e):
                print(f"\n  Warning: Sample {sample_idx} failed: CUDA out of memory.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"\n  Warning: Sample {sample_idx} failed: {e}")
            skipped_samples.append(sample_idx)
            continue

    print(f"\n[4/6] Computing statistics...")
    
    # Aggregate results
    results = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'num_samples': len(valid_samples),
        'layers': layers_to_extract,
        'kl_scores': {},
        'spearman_scores': {},
        'spearman_pvalues': {},
        'pw_kl_scores': {},
        'ndcg_scores': {},
        'renyi_scores': {},
        # NEW: Per-sample scores for detailed analysis
        'per_sample_kl_scores': {},
        'per_sample_spearman_scores': {},
        'per_sample_ndcg_scores': {},
        'per_sample_reference_distributions': per_sample_ref_distribution,
        'per_sample_kl_uniform': per_sample_kl_uniform,
        'valid_sample_indices': valid_samples,
        'skipped_sample_indices': skipped_samples
    }
    
    if len(valid_samples) == 0:
        print(f"\n❌ ERROR: No valid samples were processed! All {len(data)} samples were skipped.")
        print("This usually indicates a problem with word importance data or text matching.")
        print("Check that the data file contains valid 'word_importance' dictionaries.")
        return results
    
    for layer_idx in layers_to_extract:
        # Filter valid samples
        valid_indices = [i for i in range(len(data)) if i in valid_samples]
        
        # KL
        kl_data = kl_scores[layer_idx][:, valid_indices]
        results['kl_scores'][layer_idx] = {
            'mean': np.mean(kl_data, axis=1).tolist(),
            'std': np.std(kl_data, axis=1).tolist(),
            'min': np.min(kl_data, axis=1).tolist(),
            'max': np.max(kl_data, axis=1).tolist()
        }
        
        # NEW: Save per-sample KL scores
        results['per_sample_kl_scores'][layer_idx] = kl_data.tolist()
        
        # Spearman
        spearman_data = spearman_scores[layer_idx][:, valid_indices]
        results['spearman_scores'][layer_idx] = {
            'mean': np.mean(spearman_data, axis=1).tolist(),
            'std': np.std(spearman_data, axis=1).tolist()
        }
        
        # NEW: Save per-sample Spearman scores
        results['per_sample_spearman_scores'][layer_idx] = spearman_data.tolist()
        
        # NEW: Save Spearman p-values
        spearman_p_data = spearman_pvalues[layer_idx][:, valid_indices]
        if layer_idx not in results['spearman_pvalues']:
            results['spearman_pvalues'][layer_idx] = []
        results['spearman_pvalues'][layer_idx] = spearman_p_data.tolist()
        
        # NDCG
        for k_val in ndcg_k_values:
            if k_val not in results['ndcg_scores']:
                results['ndcg_scores'][k_val] = {}
            ndcg_data = ndcg_scores[k_val][layer_idx][:, valid_indices]
            results['ndcg_scores'][k_val][layer_idx] = {
                'mean': np.mean(ndcg_data, axis=1).tolist()
            }
            
            # NEW: Save per-sample NDCG scores (only for k=5 and k=10)
            if k_val not in results['per_sample_ndcg_scores']:
                results['per_sample_ndcg_scores'][k_val] = {}
            results['per_sample_ndcg_scores'][k_val][layer_idx] = ndcg_data.tolist()
        
        # Precision and Recall
        for k_val in k_list:
            if f'precision_at_{k_val}' not in results:
                results[f'precision_at_{k_val}'] = {}
            prec_data = precision_scores[k_val][layer_idx][:, valid_indices]
            results[f'precision_at_{k_val}'][layer_idx] = {
                'mean': np.mean(prec_data, axis=1).tolist()
            }
            
            if f'recall_at_{k_val}' not in results:
                results[f'recall_at_{k_val}'] = {}
            rec_data = recall_scores[k_val][layer_idx][:, valid_indices]
            results[f'recall_at_{k_val}'][layer_idx] = {
                'mean': np.mean(rec_data, axis=1).tolist()
            }
        
        # Power-weighted KL
        pw_kl_data = pw_kl_scores[layer_idx][:, valid_indices]
        results['pw_kl_scores'][layer_idx] = {
            'mean': np.mean(pw_kl_data, axis=1).tolist(),
            'std': np.std(pw_kl_data, axis=1).tolist()
        }
        
        # NEW: Save per-sample power-weighted KL scores
        results['per_sample_pw_kl_scores'] = results.get('per_sample_pw_kl_scores', {})
        results['per_sample_pw_kl_scores'][layer_idx] = pw_kl_data.tolist()
        
        # Rényi divergences
        for alpha in renyi_alpha_values:
            if alpha not in results['renyi_scores']:
                results['renyi_scores'][alpha] = {}
            renyi_data = renyi_scores_dict[alpha][layer_idx][:, valid_indices]
            results['renyi_scores'][alpha][layer_idx] = {
                'mean': np.mean(renyi_data, axis=1).tolist(),
                'std': np.std(renyi_data, axis=1).tolist()
            }
            
            # NEW: Save per-sample Rényi scores
            if alpha not in results.get('per_sample_renyi_scores', {}):
                if 'per_sample_renyi_scores' not in results:
                    results['per_sample_renyi_scores'] = {}
                results['per_sample_renyi_scores'][alpha] = {}
            results['per_sample_renyi_scores'][alpha][layer_idx] = renyi_data.tolist()

    print(f"\n[5/6] Saving results...")
    results_path = os.path.join(output_dir, "metrics_scores.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved to: {results_path}")
    
    print(f"\n[6/6] Generating visualizations...")
    # Placeholder for visualization - assuming separate script or simple plot
    # Individual layer PNG files disabled - comprehensive visualizations generated by compute_extra_metrics_from_saved.py
    print("  ✓ Visualization generation skipped (use compute_extra_metrics_from_saved.py for comprehensive plots)")

    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze attention vs word importance')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., samsum, cnn_dailymail, decoda)')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data JSON')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'max', 'min'],
                        help='Aggregation method for token-to-word attention')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--layer_selection', type=str, default='all',
                        help='Layer selection: "all" (all layers), "representative" (early/middle/late), or comma-separated indices')
    parser.add_argument('--test_heads_only', action='store_true',
                        help='Only test first and last attention head (faster)')
    parser.add_argument('--filter_zero_importance', action='store_true',
                        help='Exclude words with zero importance from comparison')
    parser.add_argument('--save_detailed_log', default='True',
                        help='Whether to save detailed sample log (True/False)')
    parser.add_argument('--quantization', type=str, default=None, choices=['8bit', '4bit'],
                        help='Quantization mode: "8bit" or "4bit" (requires bitsandbytes)')
    
    args = parser.parse_args()

    # Allow boolean-like strings for save_detailed_log
    save_detailed_log_arg = str(args.save_detailed_log).lower()
    save_detailed_log_bool = save_detailed_log_arg in ('1', 'true', 't', 'yes', 'y', 'on')
    
    # Auto-construct data path if not provided
    if args.data_path is None:
        if "Llama" in args.model_name:
            model_family = "meta-llama"
        elif "Qwen" in args.model_name:
            model_family = "Qwen"
        else:
            model_family = "meta-llama"
        
        args.data_path = f'../data/{model_family}/{args.model_name}/{args.dataset_name}/generated_summaries_with_word_importance_deduplicated.json'
    
    # Set default num_samples
    if args.num_samples is None:
        if args.dataset_name == 'samsum':
            args.num_samples = 819
        elif args.dataset_name == 'cnn_dailymail':
            args.num_samples = 300
        elif args.dataset_name == 'decoda':
            args.num_samples = 100
        else:
            args.num_samples = 100
    
    print(f"Data path: {args.data_path}")
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        return
    
    analyze_attention_importance(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        num_samples=args.num_samples,
        aggregation=args.aggregation,
        output_dir=args.output_dir,
        test_heads_only=args.test_heads_only,
        filter_zero_importance=args.filter_zero_importance,
        layer_selection=args.layer_selection,
        save_detailed_log=save_detailed_log_bool,
        quantization=args.quantization
    )
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
