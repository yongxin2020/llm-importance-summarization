"""
Train Word Importance Probe with Article-Level Loss

This script trains probes by calculating losses at the article level rather than token level.
For each article, all word importance predictions are aggregated and the loss is calculated
on the entire article's word importance distribution.

Key differences from token-level training:
1. Batches contain complete articles (not individual words)
2. Loss is calculated on all words in an article simultaneously 
3. Better captures document-level importance relationships
4. More realistic for document summarization evaluation

Model Saving and Loading:
- Best models are automatically saved during training to ./saved_models/article_level_probes/
- Use load_saved_model(model_path) to load a trained model
- Use list_saved_models() to see all available models
- Use 'python script.py list_models' to list models from command line

Example usage:
    # Train models: python script.py small
    # List models: python script.py list_models
    # Load model: model, info = load_saved_model('./saved_models/article_level_probes/best_model_layer_-1_word_only_small.pt')
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import random
from tqdm import tqdm
import time
import sys
from typing import List, Dict, Tuple, Optional
import gc
from scipy.stats import spearmanr

import argparse
from config import MODEL_NAME, DATASET_NAME, get_model_config, get_config, SMALL_TEST_CONFIG, FULL_TRAINING_CONFIG, COMPREHENSIVE_TRAINING_CONFIG, CONTEXT_METHODS_CONFIG
from probe_utils import ndcg_at_k, renyi_divergence, kl_divergence, GPUTimeTracker

# Results subdirectory (baseline vs randomized)
PROBE_RESULTS_SUBDIR = "article_level_probe"


class ArticleLevelWordDataset(Dataset):
    """
    Dataset for article-level word importance probing.
    Each sample is a complete article with all its words and importance scores.
    """
    def __init__(self, 
                 hidden_states_dir: str,
                 original_json_path: str,
                 layer: int = -1,
                 context_method: str = "word_only",
                 selected_article_ids: set = None,
                 min_words_per_article: int = 5,
                 max_words_per_article: int = None,
                 hidden_dim: int = 1536,
                 normalize_scores: bool = False):
        """
        Args:
            hidden_states_dir: Directory with targeted hidden state files
            original_json_path: Path to original JSON with articles and word_importance
            layer: Which layer to use (-1 for last layer)
            context_method: "word_only" or "word_and_context"
            selected_article_ids: Set of article IDs to process (None = all articles)
            min_words_per_article: Minimum words required per article
            max_words_per_article: Maximum words to use per article (None = use all available words)
            hidden_dim: Hidden dimension size for the model
            normalize_scores: Whether to normalize scores to sum to 1 per article
        """
        self.hidden_states_dir = hidden_states_dir
        self.layer = layer
        self.context_method = context_method
        self.selected_article_ids = selected_article_ids
        self.min_words_per_article = min_words_per_article
        self.max_words_per_article = max_words_per_article
        self.hidden_dim = hidden_dim
        self.normalize_scores = normalize_scores
        
        # Load original articles to get word importance data
        with open(original_json_path, 'r') as f:
            self.original_data = json.load(f)
        
        # Filter by selected articles if specified
        if self.selected_article_ids is not None:
            self.original_data = [item for item in self.original_data if item['id'] in self.selected_article_ids]
            print(f"Filtered to {len(self.original_data)} articles from selection of {len(self.selected_article_ids)}")
        
        # Build article data with word importance and hidden states
        self.articles = []
        missing_count = 0
        insufficient_words_count = 0
        
        for item in self.original_data:
            if "word_importance" not in item or not item["word_importance"]:
                continue
                
            article_id = item["id"]
            hidden_path = os.path.join(self.hidden_states_dir, f"article_{article_id}.pt")
            
            if not os.path.exists(hidden_path):
                missing_count += 1
                continue
            
            try:
                # Load hidden states
                data = torch.load(hidden_path, map_location='cpu')
                available_words = set(data['word_hidden_states'].keys())
                
                # Use scores directly from hidden states file (includes zero-score words)
                valid_words = []
                for word, word_data in data['word_hidden_states'].items():
                    score = word_data.get('score', 0.0)  # Get score from hidden states
                    # Include both zero-score words (important baseline) and non-zero words
                    if score >= 0.0:  # Include all words including zero-score
                        valid_words.append((word, score))
                
                # Normalize scores if requested
                if self.normalize_scores and valid_words:
                    total_score = sum(score for _, score in valid_words)
                    if total_score > 0:
                        valid_words = [(w, s / total_score) for w, s in valid_words]
                    else:
                        # If all scores are 0, use uniform distribution
                        uniform_score = 1.0 / len(valid_words)
                        valid_words = [(w, uniform_score) for w, _ in valid_words]

                # Check if article has enough words
                if len(valid_words) < self.min_words_per_article:
                    insufficient_words_count += 1
                    continue
                
                # Use all available words unless a limit is explicitly set
                if self.max_words_per_article is not None and len(valid_words) > self.max_words_per_article:
                    # Separate zero-score and non-zero words
                    zero_score_words = [(word, score) for word, score in valid_words if score == 0.0]
                    non_zero_words = [(word, score) for word, score in valid_words if score > 0.0]
                    
                    # Calculate how many of each type to include
                    max_zero_words = min(len(zero_score_words), self.max_words_per_article // 3)  # Up to 1/3 zero words
                    max_non_zero_words = self.max_words_per_article - max_zero_words
                    
                    # Take random sample of zero-score words and top non-zero words
                    import random
                    if len(zero_score_words) > max_zero_words:
                        selected_zero_words = random.sample(zero_score_words, max_zero_words)
                    else:
                        selected_zero_words = zero_score_words
                    
                    # Sort non-zero words by importance and take top ones
                    non_zero_words.sort(key=lambda x: x[1], reverse=True)
                    selected_non_zero_words = non_zero_words[:max_non_zero_words]
                    
                    # Combine and shuffle
                    valid_words = selected_zero_words + selected_non_zero_words
                    random.shuffle(valid_words)
                
                # Store article data
                article_data = {
                    'article_id': article_id,
                    'words_and_scores': valid_words,
                    'hidden_states_data': data,
                    'num_words': len(valid_words)
                }
                
                self.articles.append(article_data)
                
            except Exception as e:
                print(f"Error loading {hidden_path}: {e}")
                missing_count += 1
                continue
        
        print(f"Loaded {len(self.articles)} articles with sufficient word data")
        if missing_count > 0:
            print(f"Warning: {missing_count} articles skipped due to missing/corrupted hidden states")
        if insufficient_words_count > 0:
            print(f"Warning: {insufficient_words_count} articles skipped due to insufficient words (< {min_words_per_article})")
        
        # Calculate statistics safely
        if len(self.articles) > 0:
            word_counts = [article['num_words'] for article in self.articles]
            print(f"Words per article - Min: {min(word_counts)}, Max: {max(word_counts)}, "
                  f"Mean: {np.mean(word_counts):.1f}, Median: {np.median(word_counts):.1f}")
        else:
            print("No valid articles after filtering; dataset is empty.")
        print(f"Using context method: {self.context_method}")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        article_id = article['article_id']
        words_and_scores = article['words_and_scores']
        data = article['hidden_states_data']
        
        try:
            # Select layer with safety check
            num_layers = data['num_layers']
            layer_idx = self.layer if self.layer >= 0 else num_layers + self.layer
            
            # Ensure layer index is valid
            if layer_idx >= num_layers:
                layer_idx = num_layers - 1
            elif layer_idx < 0:
                layer_idx = 0
            
            # Collect all word representations and scores for this article
            word_representations = []
            word_scores = []
            word_names = []
            
            for word, score in words_and_scores:
                try:
                    # Get word representation
                    word_data = data['word_hidden_states'][word]
                    word_repr = word_data['hidden_states'][layer_idx]  # [hidden_dim]
                    
                    # Get context representation
                    context_repr = data['context_hidden_states'][layer_idx]  # [hidden_dim]
                    
                    # Combine based on context method
                    if self.context_method == "word_only":
                        combined_repr = word_repr
                    elif self.context_method == "word_and_context":
                        combined_repr = torch.cat([word_repr, context_repr], dim=0)  # [2*hidden_dim]
                    else:
                        raise ValueError(f"Unknown context method: {self.context_method}")
                    
                    word_representations.append(combined_repr.float())
                    word_scores.append(score)
                    word_names.append(word)
                    
                except Exception as e:
                    print(f"Error processing word '{word}' in article {article_id}: {e}")
                    continue
            
            if len(word_representations) == 0:
                # Return dummy data if no valid words
                dummy_dim = self.hidden_dim * (2 if self.context_method == "word_and_context" else 1)
                return {
                    'representations': torch.zeros((1, dummy_dim), dtype=torch.float32),
                    'scores': torch.tensor([0.0], dtype=torch.float32),
                    'article_id': article_id,
                    'words': ['dummy'],
                    'num_words': 1
                }
            
            # Stack all representations and scores
            representations = torch.stack(word_representations)  # [num_words, hidden_dim]
            scores = torch.tensor(word_scores, dtype=torch.float32)  # [num_words]
            
            return {
                'representations': representations,
                'scores': scores,
                'article_id': article_id,
                'words': word_names,
                'num_words': len(word_representations)
            }
            
        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            # Return dummy data
            dummy_dim = self.hidden_dim * (2 if self.context_method == "word_and_context" else 1)
            return {
                'representations': torch.zeros((1, dummy_dim), dtype=torch.float32),
                'scores': torch.tensor([0.0], dtype=torch.float32),
                'article_id': article_id,
                'words': ['dummy'],
                'num_words': 1
            }

def collate_articles(batch):
    """
    Custom collate function to handle variable-length articles.
    Pads articles to the same length within a batch.
    """
    # Find maximum number of words in this batch
    max_words = max(item['num_words'] for item in batch)
    
    # Get representation dimension from first item
    repr_dim = batch[0]['representations'].shape[1]
    
    # Prepare batch tensors
    batch_representations = torch.zeros(len(batch), max_words, repr_dim)
    batch_scores = torch.zeros(len(batch), max_words)
    batch_masks = torch.zeros(len(batch), max_words, dtype=torch.bool)  # True for valid positions
    batch_article_ids = []
    batch_words = []
    batch_num_words = []
    
    for i, item in enumerate(batch):
        num_words = item['num_words']
        
        # Copy representations and scores
        batch_representations[i, :num_words] = item['representations']
        batch_scores[i, :num_words] = item['scores']
        batch_masks[i, :num_words] = True
        
        # Store metadata
        batch_article_ids.append(item['article_id'])
        batch_words.append(item['words'])
        batch_num_words.append(num_words)
    
    return {
        'representations': batch_representations,  # [batch_size, max_words, repr_dim]
        'scores': batch_scores,                   # [batch_size, max_words]
        'masks': batch_masks,                     # [batch_size, max_words]
        'article_ids': batch_article_ids,
        'words': batch_words,
        'num_words': batch_num_words
    }

class ArticleLevelMLPRegressor(nn.Module):
    """
    MLP for article-level word importance prediction.
    Processes all words in an article and outputs importance scores.
    """
    def __init__(self, input_dim: int, use_attention: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.use_attention = use_attention
        
        # Word-level processing
        hidden_size = max(512, input_dim // 4)
        self.word_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Article-level context (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size // 2,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 1)
            # Removed Sigmoid to output logits for Softmax
        )
    
    def forward(self, representations, masks):
        """
        Args:
            representations: [batch_size, max_words, input_dim]
            masks: [batch_size, max_words] - True for valid positions
        
        Returns:
            log_probs: [batch_size, max_words] - log probability scores
        """
        batch_size, max_words, _ = representations.shape
        
        # Process each word independently
        # Flatten for processing: [batch_size * max_words, input_dim]
        flat_repr = representations.view(-1, self.input_dim)
        processed = self.word_processor(flat_repr)  # [batch_size * max_words, hidden_size//2]
        
        # Reshape back: [batch_size, max_words, hidden_size//2]
        processed = processed.view(batch_size, max_words, -1)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over words in each article
            attended, _ = self.attention(processed, processed, processed, key_padding_mask=~masks)
            processed = attended + processed  # Residual connection
        
        # Predict importance scores (logits)
        # Flatten again for final prediction
        flat_processed = processed.view(-1, processed.shape[-1])
        logits = self.predictor(flat_processed).squeeze(-1)  # [batch_size * max_words]
        
        # Reshape back: [batch_size, max_words]
        logits = logits.view(batch_size, max_words)
        
        # Set logits of invalid positions to -inf so they don't contribute
        # For MSE (Sigmoid), -inf becomes 0.0, which is appropriate for padding
        logits = logits.masked_fill(~masks, float('-inf'))
        
        return logits

def article_level_loss(logits, targets, masks, loss_type='kl'):
    """
    Calculate article-level loss.
    
    Args:
        logits: [batch_size, max_words] - Raw output logits
        targets: [batch_size, max_words] - Target values
        masks: [batch_size, max_words] - True for valid positions
        loss_type: 'kl' (distribution) or 'mse' (regression)
    
    Returns:
        loss: scalar
    """
    if loss_type == 'kl':
        # KL Divergence / Cross Entropy
        # Targets must be probabilities (sum=1)
        # Logits -> LogSoftmax
        log_probs = torch.log_softmax(logits, dim=1)
        
        valid_log_probs = log_probs[masks]
        valid_targets = targets[masks]
        
        # Cross Entropy: -sum(p * log q)
        batch_size = logits.shape[0]
        loss = -torch.sum(valid_targets * valid_log_probs) / batch_size
        
    elif loss_type == 'mse':
        # Mean Squared Error
        # Targets are raw scores (0-1)
        # Logits -> Sigmoid
        probs = torch.sigmoid(logits)
        
        valid_probs = probs[masks]
        valid_targets = targets[masks]
        
        loss = nn.MSELoss()(valid_probs, valid_targets)
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss

def calculate_article_metrics(logits, targets, masks, loss_type='kl'):
    """
    Calculate metrics at article level.
    
    Args:
        logits: [batch_size, max_words] - Raw logits
        targets: [batch_size, max_words] - Target values
        masks: [batch_size, max_words] - True for valid positions
        loss_type: 'kl' or 'mse' - determines how to interpret logits/targets
    """
    batch_size = logits.shape[0]
    article_metrics = []
    
    all_valid_predictions = []
    all_valid_targets = []
    
    # Get probabilities for metrics
    if loss_type == 'kl':
        # Logits -> Softmax (sum=1)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
    else: # mse
        # Logits -> Sigmoid (0-1)
        probs = torch.sigmoid(logits)
        # For KL metric, we need normalized distribution
        # Normalize sigmoid outputs to sum to 1 per article
        # Add epsilon to avoid division by zero
        sums = probs.sum(dim=1, keepdim=True) + 1e-12
        normalized_probs = probs / sums
        log_probs = torch.log(normalized_probs + 1e-12)
        
        # If targets are raw scores, we also need to normalize them for KL metric
        target_sums = targets.sum(dim=1, keepdim=True) + 1e-12
        normalized_targets = targets / target_sums
    
    for i in range(batch_size):
        article_mask = masks[i]
        
        # Get values for this article
        article_probs = probs[i][article_mask].detach().cpu().numpy()
        article_targets = targets[i][article_mask].detach().cpu().numpy()
        
        if len(article_probs) > 1:
            # MAE and Correlation on raw predictions (whether probability or score)
            mae = np.mean(np.abs(article_probs - article_targets))
            correlation = np.corrcoef(article_probs, article_targets)[0, 1]
            
            # Spearman Correlation
            spearman_corr, _ = spearmanr(article_probs, article_targets)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
            
            # KL Divergence (always on normalized distributions)
            if loss_type == 'kl':
                p = article_targets # Already normalized
                log_q = log_probs[i][article_mask].detach().cpu().numpy()
                q = np.exp(log_q) # Get probabilities for Renyi/NDCG
            else:
                p = normalized_targets[i][article_mask].detach().cpu().numpy()
                log_q = log_probs[i][article_mask].detach().cpu().numpy()
                q = normalized_probs[i][article_mask].detach().cpu().numpy()
            
            # KL(P||Q) = sum(P * log(P/Q))
            # Calculate Cross Entropy term: - sum(P * log Q)
            cross_entropy = -np.sum(p * log_q)
            
            # Calculate Entropy term: - sum(P * log P)
            entropy = -np.sum(p * np.log(p + 1e-12))
            
            kl_div = cross_entropy - entropy
            
            # Calculate NDCG@10
            ndcg_10 = ndcg_at_k(article_probs, article_targets, k=10)
            
            # Calculate Renyi Divergence (alpha=2.0)
            # Note: Renyi requires probability distributions
            renyi_2 = renyi_divergence(p, q, alpha=2.0)
            
            article_metrics.append({
                'mae': mae,
                'correlation': correlation,
                'spearman': spearman_corr,
                'kl_div': kl_div,
                'ndcg_10': ndcg_10,
                'renyi_2': renyi_2,
                'num_words': len(article_probs)
            })
        
        # Collect for overall metrics
        all_valid_predictions.extend(article_probs)
        all_valid_targets.extend(article_targets)
    
    # Overall metrics
    overall_mae = np.mean(np.abs(np.array(all_valid_predictions) - np.array(all_valid_targets)))
    overall_correlation = np.corrcoef(all_valid_predictions, all_valid_targets)[0, 1] if len(all_valid_predictions) > 1 else 0.0
    
    # Calculate overall Spearman
    if len(all_valid_predictions) > 1:
        overall_spearman, _ = spearmanr(all_valid_predictions, all_valid_targets)
        if np.isnan(overall_spearman):
            overall_spearman = 0.0
    else:
        overall_spearman = 0.0
    
    return {
        'article_metrics': article_metrics,
        'overall_mae': overall_mae,
        'overall_correlation': overall_correlation,
        'overall_spearman': overall_spearman,
        'total_words': len(all_valid_predictions)
    }

def train_article_level_probe(config_key: str = "small", use_attention: bool = False, model_name: str = None, dataset_name: str = None, loss_type: str = 'kl', use_randomized_weights: bool = False):
    """
    Train word importance probe using article-level loss calculation.
    
    Args:
        config_key: Configuration to use ("small", "full", "comprehensive")
        use_attention: Whether to use attention mechanism for article context
        model_name: Name of the model to use (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')
        dataset_name: Name of the dataset to use
        loss_type: 'kl' (distribution learning) or 'mse' (regression learning)
        use_randomized_weights: Whether to use hidden states from randomized model weights (control experiment)
    """
    
    # Use provided model/dataset names or fall back to globals
    global PROBE_RESULTS_SUBDIR
    if model_name is None:
        model_name = MODEL_NAME
    if dataset_name is None:
        dataset_name = DATASET_NAME
    
    # Get model configuration
    model_config = get_model_config(model_name)
    
    print(f"🤖 Training with model: {model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Hidden dimension: {model_config['hidden_dim']}")
    print(f"📐 Number of layers: {model_config['num_layers']}")
    print(f"📉 Loss Type: {loss_type.upper()}")
    
    # Initialize GPU time tracker
    gpu_tracker = GPUTimeTracker()
    gpu_tracker.start_session(f"article_level_probe_training_{config_key}_{loss_type}")
    
    # Select configuration - use dynamic config functions with model_name
    if config_key == "small":
        from config import get_small_test_config
        config = get_small_test_config(model_name)
    elif config_key == "comprehensive":
        from config import get_comprehensive_training_config
        config = get_comprehensive_training_config(model_name, dataset_name)
    else:
        from config import get_full_training_config
        config = get_full_training_config(model_name)
    
    # Set up comprehensive random seeding for reproducibility
    random_seed = config.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    print(f"Using {config_key} configuration")
    print(f"Enabled: {config['enabled']}")
    print(f"Use attention: {use_attention}")
    print(f"Random seed: {random_seed}")
    
    if not config['enabled']:
        print(f"{config_key} configuration is disabled. Enable it in config.py")
        gpu_tracker.end_session()
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data paths
    base_hidden_states_dir = "article_with_zeros_RANDOMIZED" if use_randomized_weights else "article_with_zeros"
    hidden_states_dir = f"saved_features/hidden_states/{model_name}/{dataset_name}/{base_hidden_states_dir}"
    original_json_path = f"../data/{model_name}/{dataset_name}/generated_summaries_with_word_importance_deduplicated.json"
    PROBE_RESULTS_SUBDIR = "article_level_probe_RANDOMIZED" if use_randomized_weights else "article_level_probe"
    
    if use_randomized_weights:
        print("\n🎲 CONTROL EXPERIMENT: Using RANDOMIZED model weights")
        print("   This tests if probe performance comes from learned representations")
        print(f"   Reading from: {hidden_states_dir}")
    else:
        print("\n🧠 Using LEARNED model representations")
        print(f"   Reading from: {hidden_states_dir}")
    
    # Auto-detection information
    sample_files = [f for f in os.listdir(hidden_states_dir) if f.startswith("article_") and f.endswith(".pt")]
    print(f"\n🔍 Auto-detection results:")
    print(f"  Hidden states directory: {hidden_states_dir}")
    print(f"  Available hidden state files: {len(sample_files)}")
    
    max_samples = config.get('max_samples')
    if max_samples is None:
        print(f"  💡 Will use ALL {len(sample_files)} available samples")
    elif max_samples and len(sample_files) > max_samples:
        print(f"  📏 Will limit to {max_samples} samples (from {len(sample_files)} available)")
    else:
        print(f"  📊 Will use all {len(sample_files)} available samples")
    
    # Use optimized context methods configuration
    if CONTEXT_METHODS_CONFIG['use_only_best']:
        context_methods = CONTEXT_METHODS_CONFIG['default_methods']  # ["word_only"] - best performing
        print(f"🎯 Using optimized method: {context_methods}")
        print("   Running only word_only method for efficiency (best performing)")
    else:
        context_methods = CONTEXT_METHODS_CONFIG['all_methods']  # ["word_only", "word_and_context"]
        print(f"📊 Using comprehensive comparison: {context_methods}")
        print("   Running both methods for comparison")
    
    #layers_to_test = config.get('layers_to_test', [-1])  # Default to last layer only
    # New (skip embedding layer)
    # layers_to_test = list(range(1, model_config['num_layers'] + 1))
    layers_to_test = config.get('layers_to_test')
    if layers_to_test is None:
        layers_to_test = list(range(1, model_config['num_layers'] + 1))

    # Check how many layers of hidden states are saved in a sample .pt file
    sample_pt_file = None
    # Reuse the hidden_states_dir already constructed above
    sample_files = [f for f in os.listdir(hidden_states_dir) if f.startswith("article_") and f.endswith(".pt")]
    if sample_files:
        sample_pt_file = os.path.join(hidden_states_dir, sample_files[0])
        try:
            data = torch.load(sample_pt_file, map_location='cpu')
            num_layers_in_pt = data.get('num_layers', None)
            print(f"[INFO] Number of layers in sample hidden state file: {num_layers_in_pt}")
        except Exception as e:
            print(f"[WARN] Could not read sample .pt file: {e}")
            num_layers_in_pt = None
    else:
        print("[WARN] No sample .pt file found for layer check.")
        num_layers_in_pt = None

    print(f"[INFO] Number of layers to test: {len(layers_to_test)}")
    print(f"[INFO] layers_to_test: {layers_to_test}")

    results = {}
    
    for layer in layers_to_test:
        print(f"\n{'='*80}")
        print(f"Testing layer {layer} with article-level loss")
        print(f"{'='*80}")
        
        for context_method in context_methods:
            print(f"\nContext method: {context_method}")
            print("-" * 40)
            
            method_start_time = time.time()
            
            try:
                # Handle article-level sampling if max_samples is specified
                max_samples = config.get('max_samples')
                selected_article_ids = None
                
                if max_samples is not None:
                    # Same logic as other scripts for article sampling
                    print(f"Finding articles with extracted hidden states...")
                    available_hidden_states = [f for f in os.listdir(hidden_states_dir) 
                                             if f.startswith("article_") and f.endswith(".pt")]
                    
                    available_article_ids = []
                    for filename in available_hidden_states:
                        article_id = filename[8:-3]  # Remove 'article_' and '.pt'
                        available_article_ids.append(article_id)
                    
                    print(f"Found {len(available_article_ids)} articles with extracted hidden states")
                    
                    if len(available_article_ids) > max_samples:
                        random_seed = config.get('random_seed', 42)
                        torch.manual_seed(random_seed)
                        
                        article_indices = torch.randperm(len(available_article_ids))[:max_samples]
                        selected_article_ids = set(available_article_ids[i] for i in article_indices)
                        print(f"📊 Sampling {max_samples} articles from {len(available_article_ids)} with hidden states")
                        print(f"   Using random seed {random_seed} for reproducible article sampling")
                    else:
                        print(f"📊 Using all {len(available_article_ids)} articles with hidden states")
                        selected_article_ids = set(available_article_ids)
                else:
                    print(f"📊 Using ALL available articles with hidden states")

                # Create article-level dataset
                print("Creating article-level dataset...")
                # If using KL loss, we want normalized targets (probabilities)
                # If using MSE loss, we want raw targets (0-1 scores)
                normalize_scores = (loss_type == 'kl')
                
                dataset = ArticleLevelWordDataset(
                    hidden_states_dir=hidden_states_dir,
                    original_json_path=original_json_path,
                    layer=layer,
                    context_method=context_method,
                    selected_article_ids=selected_article_ids,
                    min_words_per_article=5,
                    max_words_per_article=None,  # Use all available words for complete training data
                    hidden_dim=model_config['hidden_dim'],
                    normalize_scores=normalize_scores  # Dynamic based on loss type
                )
                
                if len(dataset) == 0:
                    print("No valid articles found!")
                    continue
                
                print(f"📊 Final dataset contains {len(dataset)} articles")
                
                # Get input dimension from first sample
                sample = dataset[0]
                input_dim = sample['representations'].shape[1]
                print(f"Input dimension: {input_dim}")
                
                # Split dataset
                total_size = len(dataset)
                test_size = int(total_size * config.get('test_ratio', 0.2))
                dev_size = int(total_size * config.get('dev_ratio', 0.2))
                train_size = total_size - test_size - dev_size
                
                # Create generator with fixed seed for reproducible splits
                generator = torch.Generator().manual_seed(random_seed)
                train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(
                    dataset, [train_size, dev_size, test_size], generator=generator
                )
                
                print(f"Article split - Train: {train_size}, Dev: {dev_size}, Test: {test_size}")
                print(f"Using random seed {random_seed} for reproducible train/dev/test splits")
                
                # Create dataloaders with custom collate function
                batch_size = min(config.get('batch_size', 8), 4)  # Smaller batch for articles
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    collate_fn=collate_articles
                )
                dev_loader = DataLoader(
                    dev_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    collate_fn=collate_articles
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    collate_fn=collate_articles
                )
                
                # Create model
                model = ArticleLevelMLPRegressor(
                    input_dim=input_dim,
                    use_attention=use_attention
                ).to(device)
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
                
                # Optimizer
                optimizer = optim.AdamW(model.parameters(), 
                                      lr=config.get('learning_rate', 1e-4),
                                      weight_decay=1e-5)
                
                # Training loop
                model.train()
                train_losses = []
                dev_losses = []
                best_dev_loss = float('inf')
                patience_counter = 0
                early_stopping = config.get('early_stopping', 5)
                best_model_state = None
                
                training_start_time = time.time()
                
                print(f"\nStarting training for {config.get('num_epochs', 10)} epochs...")
                
                for epoch in range(config.get('num_epochs', 10)):
                    # Training phase
                    model.train()
                    epoch_train_loss = 0.0
                    num_train_batches = 0
                    
                    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Train")
                    for batch in train_pbar:
                        representations = batch['representations'].to(device)
                        scores = batch['scores'].to(device)
                        masks = batch['masks'].to(device)
                        
                        optimizer.zero_grad()
                        logits = model(representations, masks)
                        loss = article_level_loss(logits, scores, masks, loss_type=loss_type)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_train_loss += loss.item()
                        num_train_batches += 1
                        
                        train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                    
                    if num_train_batches > 0:
                        avg_train_loss = epoch_train_loss / num_train_batches
                        train_losses.append(avg_train_loss)
                    else:
                        avg_train_loss = float('inf')
                        print(f"Epoch {epoch+1}: No training batches processed!")
                    
                    # Validation phase
                    model.eval()
                    epoch_dev_loss = 0.0
                    num_dev_batches = 0
                    
                    with torch.no_grad():
                        for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1} - Dev"):
                            representations = batch['representations'].to(device)
                            scores = batch['scores'].to(device)
                            masks = batch['masks'].to(device)
                            
                            logits = model(representations, masks)
                            loss = article_level_loss(logits, scores, masks, loss_type=loss_type)
                            
                            epoch_dev_loss += loss.item()
                            num_dev_batches += 1
                    
                    if num_dev_batches > 0:
                        avg_dev_loss = epoch_dev_loss / num_dev_batches
                        dev_losses.append(avg_dev_loss)
                        
                        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Dev Loss = {avg_dev_loss:.6f}")
                        
                        # Early stopping check
                        if avg_dev_loss < best_dev_loss:
                            best_dev_loss = avg_dev_loss
                            patience_counter = 0
                            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            print(f"  → New best dev loss: {best_dev_loss:.6f}")
                        else:
                            patience_counter += 1
                            print(f"  → No improvement (patience: {patience_counter}/{early_stopping})")
                            
                            if patience_counter >= early_stopping:
                                print(f"  → Early stopping triggered at epoch {epoch+1}")
                                break
                    else:
                        dev_losses.append(float('nan'))
                        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Dev Loss = N/A (No dev samples)")
                        # If no dev set, always update best model (or just take the last one)
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                # Load best model for final evaluation
                if best_model_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                    print(f"Loaded best model with dev loss: {best_dev_loss:.6f}")
                
                # Final evaluation on test set
                print("Evaluating on test set...")
                model.eval()
                test_loss = 0.0
                num_test_batches = 0
                all_article_metrics = []
                
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Test Evaluation"):
                        representations = batch['representations'].to(device)
                        scores = batch['scores'].to(device)
                        masks = batch['masks'].to(device)
                        
                        logits = model(representations, masks)
                        loss = article_level_loss(logits, scores, masks, loss_type=loss_type)
                        
                        test_loss += loss.item()
                        num_test_batches += 1
                        
                        # Calculate metrics for this batch
                        batch_metrics = calculate_article_metrics(logits, scores, masks, loss_type=loss_type)
                        all_article_metrics.extend(batch_metrics['article_metrics'])
                
                if num_test_batches > 0:
                    avg_test_loss = test_loss / num_test_batches
                else:
                    avg_test_loss = float('nan')
                
                # Calculate overall metrics
                if all_article_metrics:
                    article_maes = [m['mae'] for m in all_article_metrics]
                    article_correlations = [m['correlation'] for m in all_article_metrics if not np.isnan(m['correlation'])]
                    article_spearmans = [m['spearman'] for m in all_article_metrics if not np.isnan(m['spearman'])]
                    article_ndcgs = [m['ndcg_10'] for m in all_article_metrics if not np.isnan(m['ndcg_10'])]
                    article_kls = [m['kl_div'] for m in all_article_metrics if not np.isnan(m['kl_div'])]
                    article_renyis = [m['renyi_2'] for m in all_article_metrics if not np.isnan(m['renyi_2']) and not np.isinf(m['renyi_2'])]
                    
                    overall_mae = np.mean(article_maes)
                    overall_correlation = np.mean(article_correlations) if article_correlations else 0.0
                    overall_spearman = np.mean(article_spearmans) if article_spearmans else 0.0
                    overall_ndcg = np.mean(article_ndcgs) if article_ndcgs else 0.0
                    overall_kl = np.mean(article_kls) if article_kls else 0.0
                    overall_renyi = np.mean(article_renyis) if article_renyis else 0.0
                    
                    print(f"\n🎯 Results for {context_method} (Layer {layer}):")
                    print(f"Final Test Loss: {avg_test_loss:.6f}")
                    print(f"Best Dev Loss: {best_dev_loss:.6f}")
                    print(f"Test MAE: {overall_mae:.6f}")
                    print(f"Test Correlation (Pearson): {overall_correlation:.6f}")
                    print(f"Test Correlation (Spearman): {overall_spearman:.6f}")
                    print(f"Test NDCG@10: {overall_ndcg:.6f}")
                    print(f"Test KL Divergence: {overall_kl:.6f}")
                    print(f"Test Renyi (alpha=2.0): {overall_renyi:.6f}")
                    print(f"Articles evaluated: {len(all_article_metrics)}")
                
                # Calculate timing
                method_end_time = time.time()
                method_duration = method_end_time - method_start_time
                training_duration = method_end_time - training_start_time
                
                print(f"Training time: {training_duration/60:.2f} minutes")
                print(f"Total method time: {method_duration/60:.2f} minutes")
                
                # Save the best model
                model_save_dir = f'./results/{model_name}/{dataset_name}/{PROBE_RESULTS_SUBDIR}/models'
                os.makedirs(model_save_dir, exist_ok=True)
                
                model_filename = f"{model_save_dir}/best_model_layer_{layer}_{context_method}_{config_key}.pt"
                model_info = {
                    'model_state_dict': best_model_state if best_model_state is not None else model.state_dict(),
                    'model_config': {
                        'input_dim': input_dim,
                        'use_attention': use_attention,
                        'layer': layer,
                        'context_method': context_method,
                        'config_key': config_key
                    },
                    'training_info': {
                        'best_dev_loss': best_dev_loss,
                        'test_loss': avg_test_loss,
                        'test_mae': overall_mae if all_article_metrics else 0.0,
                        'test_correlation': overall_correlation if all_article_metrics else 0.0,
                        'test_spearman': overall_spearman if all_article_metrics else 0.0,
                        'test_ndcg_10': overall_ndcg if all_article_metrics else 0.0,
                        'test_renyi_2': overall_renyi if all_article_metrics else 0.0,
                        'num_epochs_trained': len(train_losses),
                        'early_stopped': len(train_losses) < config.get('num_epochs', 10),
                        'train_losses': train_losses,
                        'dev_losses': dev_losses
                    }
                }
                
                torch.save(model_info, model_filename)
                print(f"💾 Saved best model: {model_filename}")
                
                # Store results
                key = f"layer_{layer}_{context_method}_article_level"
                results[key] = {
                    'test_loss': avg_test_loss,
                    'dev_loss': best_dev_loss,
                    'mae': overall_mae if all_article_metrics else 0.0,
                    'correlation': overall_correlation if all_article_metrics else 0.0,
                    'spearman': overall_spearman if all_article_metrics else 0.0,
                    'ndcg_10': overall_ndcg if all_article_metrics else 0.0,
                    'kl_div': overall_kl if all_article_metrics else 0.0,
                    'renyi_2': overall_renyi if all_article_metrics else 0.0,
                    'train_losses': train_losses,
                    'dev_losses': dev_losses,
                    'input_dim': input_dim,
                    'layer': layer,
                    'context_method': context_method,
                    'dataset_size': len(dataset),
                    'train_size': train_size,
                    'dev_size': dev_size,
                    'test_size': test_size,
                    'num_epochs_trained': len(train_losses),
                    'early_stopped': len(train_losses) < config.get('num_epochs', 10),
                    'model_parameters': total_params,
                    'use_attention': use_attention,
                    'num_articles_evaluated': len(all_article_metrics),
                    'model_path': model_filename,  # Add path to saved model
                    'timing': {
                        'total_method_time_seconds': method_duration,
                        'total_method_time_minutes': method_duration / 60,
                        'pure_training_time_seconds': training_duration,
                        'pure_training_time_minutes': training_duration / 60,
                    }
                }
                
                # Memory cleanup
                del model, optimizer, train_loader, dev_loader, test_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                print(f"Error with layer {layer}, context {context_method}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("ARTICLE-LEVEL PROBE TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for key, result in results.items():
        print(f"\n{key}:")
        print(f"  Input dimension: {result['input_dim']:,}")
        print(f"  Model parameters: {result['model_parameters']:,}")
        print(f"  Dev Loss: {result['dev_loss']:.6f}")
        print(f"  Test Loss: {result['test_loss']:.6f}")
        print(f"  Test MAE: {result['mae']:.6f}")
        print(f"  Test Correlation (Pearson): {result['correlation']:.6f}")
        print(f"  Test Correlation (Spearman): {result.get('spearman', 0.0):.6f}")
        print(f"  Articles evaluated: {result['num_articles_evaluated']}")
        print(f"  Training time: {result['timing']['pure_training_time_minutes']:.2f} min")
    
    # Save results
    save_article_level_results(results, config_key, gpu_tracker.get_summary(), model_name, dataset_name, model_config)
    
    # End GPU tracking
    gpu_tracker.end_session()
    gpu_tracker.get_summary()
    
    # Find best method
    if results:
        best_method = max(results.keys(), key=lambda k: results[k]['correlation'])
        print(f"\n🏆 Best method: {best_method}")
        print(f"Best correlation: {results[best_method]['correlation']:.6f}")
        print(f"Articles evaluated: {results[best_method]['num_articles_evaluated']}")
        
        # Show path to best model
        if 'model_path' in results[best_method]:
            print(f"💾 Best model saved at: {results[best_method]['model_path']}")
    
    # Show summary of all saved models
    print(f"\n📁 MODEL SUMMARY:")
    saved_models = []
    for key, result in results.items():
        if 'model_path' in result:
            saved_models.append({
                'path': result['model_path'],
                'method': key,
                'correlation': result['correlation'],
                'mae': result['mae']
            })
    
    if saved_models:
        print(f"Saved {len(saved_models)} models:")
        for model_info in sorted(saved_models, key=lambda x: x['correlation'], reverse=True):
            print(f"  📊 {model_info['method']}: {model_info['path']}")
            print(f"     Correlation: {model_info['correlation']:.3f}, MAE: {model_info['mae']:.3f}")
    
    return results

 

def save_article_level_results(results: Dict, config_key: str, gpu_summary: Dict, model_name: str, dataset_name: str, model_config: Dict):
    """Save comprehensive results for article-level probe training"""
    from datetime import datetime
    
    global PROBE_RESULTS_SUBDIR
    os.makedirs(f'./results/{model_name}/{dataset_name}/{PROBE_RESULTS_SUBDIR}/json', exist_ok=True)
    
    # Prepare results
    comprehensive_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_used': config_key,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'training_type': 'article_level_probe',
            'total_experiments': len(results),
            'hidden_dim': model_config['hidden_dim'],
            'loss_calculation': 'article_level'
        },
        'training_results': results,
        'gpu_usage': gpu_summary
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    comprehensive_results = convert_numpy(comprehensive_results)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'./results/{model_name}/{dataset_name}/{PROBE_RESULTS_SUBDIR}/json/article_level_training_results_{config_key}_{timestamp}.json'
    latest_filename = f'./results/{model_name}/{dataset_name}/{PROBE_RESULTS_SUBDIR}/json/latest_article_level_results_{config_key}.json'
    
    with open(filename, 'w') as f:
        json.dump(convert_numpy(comprehensive_results), f, indent=2)
    
    with open(latest_filename, 'w') as f:
        json.dump(convert_numpy(comprehensive_results), f, indent=2)
    
    print(f"\n💾 Article-level results saved:")
    print(f"  - {filename}")
    print(f"  - {latest_filename}")

 

def load_saved_model(model_path: str, device: str = "cuda"):
    """
    Load a saved article-level probe model.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        tuple: (model, model_info) where model is the loaded ArticleLevelMLPRegressor
               and model_info contains training information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model info
    model_info = torch.load(model_path, map_location=device)
    model_config = model_info['model_config']
    
    # Create model with same configuration
    model = ArticleLevelMLPRegressor(
        input_dim=model_config['input_dim'],
        use_attention=model_config['use_attention']
    )
    
    # Load state dict
    model.load_state_dict(model_info['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from {model_path}")
    print(f"   Layer: {model_config['layer']}")
    print(f"   Context method: {model_config['context_method']}")
    print(f"   Input dim: {model_config['input_dim']}")
    print(f"   Use attention: {model_config['use_attention']}")
    print(f"   Test correlation (Pearson): {model_info['training_info']['test_correlation']:.3f}")
    if 'test_spearman' in model_info['training_info']:
        print(f"   Test correlation (Spearman): {model_info['training_info']['test_spearman']:.3f}")
    print(f"   Test MAE: {model_info['training_info']['test_mae']:.3f}")
    
    return model, model_info

def list_saved_models():
    """
    List all saved article-level probe models.
    
    Returns:
        list: List of model file paths
    """
    model_dir = './saved_models/article_level_probes'
    if not os.path.exists(model_dir):
        print(f"No saved models directory found: {model_dir}")
        return []
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if not model_files:
        print(f"No saved models found in {model_dir}")
        return []
    
    print(f"📁 Found {len(model_files)} saved models in {model_dir}:")
    
    model_paths = []
    for model_file in sorted(model_files):
        model_path = os.path.join(model_dir, model_file)
        model_paths.append(model_path)
        
        try:
            # Load model info to show details
            model_info = torch.load(model_path, map_location='cpu')
            config = model_info['model_config']
            training = model_info['training_info']
            
            print(f"  📊 {model_file}")
            print(f"     Layer: {config['layer']}, Method: {config['context_method']}")
            print(f"     Pearson: {training['test_correlation']:.3f}, MAE: {training['test_mae']:.3f}")
            if 'test_spearman' in training:
                print(f"     Spearman: {training['test_spearman']:.3f}")
        except Exception as e:
            print(f"  ❌ {model_file} (Error loading: {e})")
    
    return model_paths

 

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train article-level word importance probe for multiple models")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='Model name to train probe for')
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, 
                        help='Dataset name')
    
    # Parse known args to handle mode as positional argument
    parser.add_argument('--loss_type', type=str, default='kl', choices=['kl', 'mse'],
                        help='Loss type: kl (distribution) or mse (regression)')
    parser.add_argument('--use_randomized_weights', action='store_true',
                        help='Use hidden states from randomized model weights (control experiment)')
    
    args, remaining = parser.parse_known_args()
    
    # Get model-specific configuration
    model_config = get_model_config(args.model_name)
    
    # Determine mode from remaining arguments or default
    if remaining:
        mode = remaining[0].lower()
        use_attention = len(remaining) > 1 and remaining[1].lower() == "attention"
    else:
        mode = "small"  # Default mode
        use_attention = False
    
    print(f"🤖 Model: {args.model_name}")
    print(f"📊 Dataset: {args.dataset_name}")
    print(f"🔧 Hidden Dimension: {model_config['hidden_dim']}")
    print(f"📐 Number of Layers: {model_config['num_layers']}")
    print(f"📉 Loss Type: {args.loss_type.upper()}")
    
    if mode == "list_models":
        # List all saved models
        print("📁 SAVED MODELS: Listing all article-level probe models...")
        list_saved_models()
        
    elif mode == "small":
        print("🧪 Starting SMALL TEST for ARTICLE-LEVEL probe...")
        print("This will train probes using article-level loss calculation")
        print("Expected runtime: ~10-15 minutes")
        results = train_article_level_probe("small", use_attention=use_attention, model_name=args.model_name, dataset_name=args.dataset_name, loss_type=args.loss_type, use_randomized_weights=args.use_randomized_weights)
    elif mode == "comprehensive":
        print("🚀 Starting COMPREHENSIVE ARTICLE-LEVEL probe training...")
        print("This will train probes using article-level loss with ALL available samples")
        print("Expected runtime: 60-120 minutes")
        results = train_article_level_probe("comprehensive", use_attention=use_attention, model_name=args.model_name, dataset_name=args.dataset_name, loss_type=args.loss_type, use_randomized_weights=args.use_randomized_weights)
    elif mode == "full":
        print("🚀 Starting FULL ARTICLE-LEVEL probe training...")
        print("This will train probes using article-level loss with ~1000 samples")
        print("Expected runtime: 30-60 minutes")
        results = train_article_level_probe("full", use_attention=use_attention, model_name=args.model_name, dataset_name=args.dataset_name, loss_type=args.loss_type, use_randomized_weights=args.use_randomized_weights)
    else:
        print(f"Unknown mode: {mode}")
        print("Valid options: small, full, comprehensive, list_models")
        print("Usage: python script.py --model_name MODEL --dataset_name DATASET --loss_type [kl|mse] <mode> [attention]")
        print("Examples:")
        print("  python script.py --model_name 'Qwen/Qwen2.5-1.5B-Instruct' full")
        print("  python script.py --model_name 'Qwen/Qwen2.5-1.5B-Instruct' --loss_type mse small attention")
        print("  python script.py list_models")
        sys.exit(1)
