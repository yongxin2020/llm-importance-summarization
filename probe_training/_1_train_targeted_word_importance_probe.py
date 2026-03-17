"""
Train Word Importance Probe with Targeted Hidden States

This script trains probes using the new targeted hidden state format that only contains
word-specific and context hidden states (much more memory efficient).

Detailed Functionality:
-----------------------
1. **Purpose**: Trains a linear/MLP probe to predict word importance scores from the internal hidden states of an LLM.
   It determines how well the model encodes "importance" at different layers of its architecture.

2. **Input Data**:
   - **Targeted Hidden States**: Loads `.pt` files containing hidden states for specific words (tokens) from `saved_features/hidden_states/...`.
     This is memory-efficient as it avoids loading full article representations.
   - **Ground Truth**: Loads `generated_summaries_with_word_importance_deduplicated.json` containing the target importance scores.

3. **Methodology**:
   - **Layer-wise Training**: A separate probe is trained for each layer specified in the configuration.
     - Typically ranges from Layer 0 (Embedding) to Layer N (Final Transformer Layer).
   - **Normalization**: 
     - If `loss_type='kl'` (default), importance scores are normalized per article to form a probability distribution (sum=1).
     - This treats importance prediction as a distribution matching problem.
   - **Probe Architecture**: 
     - A Multi-Layer Perceptron (MLP) with ReLU activation and Dropout.
     - Input: Hidden state dimension (e.g., 2048, 4096).
     - Output: Scalar importance score.

4. **Loss Functions**:
   - **KL Divergence (Default)**: Implemented via `BCEWithLogitsLoss` on normalized targets. This effectively minimizes the divergence between the predicted importance distribution and the ground truth distribution.
   - **MSE**: Mean Squared Error regression (optional).

5. **Configuration**:
   - Controlled via `config.py` (e.g., `get_layerwise_training_config`).
   - Parameters include: layers to test, max samples, batch size, learning rate, etc.

6. **Output**:
   - Saves detailed metrics (MAE, Pearson, Spearman, NDCG) in JSON format.
   - Generates visualization plots (heatmaps, performance trends) in `results/...`.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import time
import sys
import traceback
import argparse
from typing import List, Dict, Tuple
from config import MODEL_NAME, DATASET_NAME, get_model_config, get_config
from probe_utils import ndcg_at_k, renyi_divergence, kl_divergence, GPUTimeTracker


class TargetedWordDataset(Dataset):
    """
    Dataset for word importance probing using targeted hidden states.
    Loads only the specific word and context representations (memory efficient).
    """
    def __init__(self, 
                 hidden_states_dir: str,
                 original_json_path: str,
                 layer: int = -1,
                 context_method: str = "word_only",
                 selected_article_ids: set = None,
                 hidden_dim: int = 2048,
                 normalize_scores: bool = False):
        """
        Args:
            hidden_states_dir: Directory with targeted hidden state files
            original_json_path: Path to original JSON with articles and word_importance
            layer: Which layer to use (-1 for last layer)
            context_method: "word_only" or "word_and_context"
            selected_article_ids: Set of article IDs to process (None = all articles)
            hidden_dim: Hidden dimension size for the model
            normalize_scores: Whether to normalize scores to probabilities (sum=1) per article
        """
        self.hidden_states_dir = hidden_states_dir
        self.layer = layer
        self.context_method = context_method
        self.selected_article_ids = selected_article_ids
        self.hidden_dim = hidden_dim
        self.normalize_scores = normalize_scores
        
        # Load original articles to get word importance data
        with open(original_json_path, 'r') as f:
            self.original_data = json.load(f)
        
        # Filter by selected articles if specified
        if self.selected_article_ids is not None:
            self.original_data = [item for item in self.original_data if item['id'] in self.selected_article_ids]
            print(f"Filtered to {len(self.original_data)} articles from selection of {len(self.selected_article_ids)}")
        
        # Create lookup dict for faster access
        self.data_lookup = {item['id']: item for item in self.original_data}
        
        # Build list of all word-article pairs with available hidden states
        self.pairs = []
        missing_count = 0
        
        # Store article-level word lists for normalization
        self.article_words = {}  # article_id -> list of (word, score)
        
        for item in self.original_data:
            if "word_importance" not in item or not item["word_importance"]:
                continue
                
            article_id = item["id"]
            hidden_path = os.path.join(self.hidden_states_dir, f"article_{article_id}.pt")
            
            if not os.path.exists(hidden_path):
                missing_count += 1
                continue
            
            # Check what words are available in the saved file
            try:
                data = torch.load(hidden_path, map_location='cpu')
                available_words = set(data['word_hidden_states'].keys())
                
                # Collect words and scores for this article
                article_word_scores = []
                for word, word_data in data['word_hidden_states'].items():
                    score = word_data.get('score', 0.0)  # Get score from hidden states
                    # Include both zero-score words (important baseline) and non-zero words
                    if score >= 0.0:  # Include all words including zero-score
                        article_word_scores.append((word, score))
                
                # Normalize scores per article if requested
                if self.normalize_scores and article_word_scores:
                    total_score = sum(score for _, score in article_word_scores)
                    if total_score > 0:
                        article_word_scores = [(w, s / total_score) for w, s in article_word_scores]
                    else:
                        # If all scores are 0, use uniform distribution
                        uniform_score = 1.0 / len(article_word_scores)
                        article_word_scores = [(w, uniform_score) for w, _ in article_word_scores]
                
                # Store for this article
                self.article_words[article_id] = article_word_scores
                
                # Add to pairs list
                for word, normalized_score in article_word_scores:
                    self.pairs.append((article_id, word, normalized_score))
                        
            except Exception as e:
                print(f"Error loading {hidden_path}: {e}")
                missing_count += 1
                continue
        
        print(f"Loaded {len(self.pairs)} word pairs with available targeted hidden states")
        if missing_count > 0:
            print(f"Warning: {missing_count} articles skipped due to missing/corrupted hidden states")
        print(f"Using context method: {self.context_method}")
        if self.normalize_scores:
            print(f"✓ Scores normalized to probabilities (sum=1) per article")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        article_id, word, score = self.pairs[idx]
        
        # Load targeted hidden states
        hidden_path = os.path.join(self.hidden_states_dir, f"article_{article_id}.pt")
        
        try:
            data = torch.load(hidden_path, map_location='cpu')
            
            # Select layer with safety check
            num_layers = data['num_layers']
            layer_idx = self.layer if self.layer >= 0 else num_layers + self.layer
            
            # Ensure layer index is valid
            if layer_idx >= num_layers:
                print(f"Warning: Layer {layer_idx} >= {num_layers}, using last layer {num_layers-1}")
                layer_idx = num_layers - 1
            elif layer_idx < 0:
                print(f"Warning: Layer {layer_idx} < 0, using layer 0")
                layer_idx = 0
            
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
                raise ValueError(f"Unknown context method: {self.context_method}. Use 'word_only' or 'word_and_context'")
            
            return combined_repr.float(), torch.tensor(score, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading data for article {article_id}, word '{word}': {e}")
            # Return dummy data
            if self.context_method == "word_and_context":
                dummy_repr = torch.zeros(2 * self.hidden_dim, dtype=torch.float32)
            else:
                dummy_repr = torch.zeros(self.hidden_dim, dtype=torch.float32)
            return dummy_repr, torch.tensor(0.0, dtype=torch.float32)

class MLPImportanceRegressor(nn.Module):
    """MLP for importance score regression with flexible input size"""
    def __init__(self, input_dim: int, use_sigmoid: bool = True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        hidden_size = max(512, input_dim // 4)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        if use_sigmoid:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x):
        logits = self.network(x).squeeze(-1)
        return self.output_activation(logits)

def train_targeted_probe(config_key: str = "small", model_name: str = None, dataset_name: str = None, loss_type: str = 'kl'):
    """Train word importance probe using targeted hidden states
    
    Args:
        config_key: Configuration to use ("small", "full", "comprehensive")
        model_name: Model name
        dataset_name: Dataset name
        loss_type: Loss type - 'kl' (distribution) or 'mse' (regression)
    """
    
    # Use defaults if not provided
    if model_name is None:
        model_name = MODEL_NAME
    if dataset_name is None:
        dataset_name = DATASET_NAME
    
    # Get model-specific configuration
    model_config = get_model_config(model_name)
    hidden_dim = model_config["hidden_dim"]
    
    # Initialize GPU time tracker
    gpu_tracker = GPUTimeTracker()
    gpu_tracker.start_session(f"probe_training_{config_key}")
    
    # Select configuration
    config = get_config(model_name)  # This automatically chooses based on enabled flag
    
    # Override with specific config if requested
    if config_key == "comprehensive":
        from config import get_comprehensive_training_config
        config = get_comprehensive_training_config(model_name)
    elif config_key == "full":
        from config import get_full_training_config  
        config = get_full_training_config(model_name)
    elif config_key == "small":
        from config import get_small_test_config
        config = get_small_test_config(model_name)
    elif config_key == "layerwise":
        from config import get_layerwise_training_config
        config = get_layerwise_training_config(model_name)
    
    # Set up random seeding for reproducibility
    random_seed = config.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    print(f"🤖 Model: {model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Hidden Dimension: {hidden_dim}")
    print(f"Using {config_key} configuration")
    print(f"Enabled: {config['enabled']}")
    print(f"Random seed: {random_seed}")
    
    # Display sample configuration information
    max_samples = config.get('max_samples')
    if max_samples is None:
        print(f"Sample limit: AUTO-DETECT (use all available hidden states)")
    else:
        print(f"Sample limit: {max_samples}")
    
    print(f"Layers to test: {len(config['layers_to_test'])} layers {config['layers_to_test']}")
    print(f"Batch size: {config['batch_size']}, Epochs: {config['num_epochs']}")
    
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
    
    # Data paths - updated to use model and dataset parameters
    hidden_states_dir = f"saved_features/hidden_states/{model_name}/{dataset_name}/article_with_zeros"
    original_json_path = f"../data/{model_name}/{dataset_name}/generated_summaries_with_word_importance_deduplicated.json"
    
    # Auto-detect number of layers from saved hidden states
    num_layers = None
    sample_files = [f for f in os.listdir(hidden_states_dir) if f.startswith("article_") and f.endswith(".pt")]
    
    print(f"\n🔍 Auto-detection results:")
    print(f"  Hidden states directory: {hidden_states_dir}")
    print(f"  Available hidden state files: {len(sample_files)}")
    
    if config.get('max_samples') is None:
        print(f"  💡 COMPREHENSIVE mode: Will use ALL {len(sample_files)} available samples")
    elif config.get('max_samples') and len(sample_files) > config['max_samples']:
        print(f"  📏 Will limit to {config['max_samples']} samples (from {len(sample_files)} available)")
    else:
        print(f"  📊 Will use all {len(sample_files)} available samples (less than limit)")
    
    if sample_files:
        sample_path = os.path.join(hidden_states_dir, sample_files[0])
        try:
            sample_data = torch.load(sample_path, map_location='cpu')
            num_layers = sample_data['num_layers']
            print(f"Detected {num_layers} layers in model")
        except Exception as e:
            print(f"Error detecting layers from {sample_path}: {e}")
            num_layers = 17  # Fallback for Llama-3.2-1B-Instruct
    else:
        print("No saved hidden state files found, using fallback layer count")
        num_layers = 17  # Fallback for Llama-3.2-1B-Instruct
    
    # Adapt layers to test based on actual model layers
    config_layers = config['layers_to_test']
    # Adapt layers to test based on actual model layers
    layers_to_test = [layer if layer < num_layers else num_layers - 1 for layer in config_layers]
    layers_to_test = sorted(list(set(layers_to_test)))
    if not layers_to_test:
        layers_to_test = [4, num_layers // 2, num_layers - 1]  # Early, middle, late
    print(f"Original layers config: {config_layers}")
    print(f"Adapted layers for model: {layers_to_test}")
    # Test different context methods and layers
    from config import CONTEXT_METHODS_CONFIG
    if CONTEXT_METHODS_CONFIG['use_only_best']:
        context_methods = CONTEXT_METHODS_CONFIG['default_methods']
        print("Using optimized method: word_only (best performance based on experiments)")
    else:
        context_methods = CONTEXT_METHODS_CONFIG['all_methods']
        print("Using comprehensive comparison: both word_only and word_and_context methods")
    
    results = {}
    
    for layer in layers_to_test:
        print(f"\n{'='*60}")
        print(f"Testing layer {layer}")
        print(f"{'='*60}")
        
        for context_method in context_methods:
            # Check if model already exists to skip training
            models_dir = f'./results/{model_name}/{dataset_name}/layer_wise_probe/models'
            expected_model_path = os.path.join(models_dir, f"layer_{layer}_{context_method}_best.pt")
            
            if os.path.exists(expected_model_path):
                print(f"⏩ Skipping layer {layer}, method {context_method}: Model already exists at {expected_model_path}")
                
                # Try to load existing results if possible to include in summary
                # This is optional but helpful for the final summary
                try:
                    checkpoint = torch.load(expected_model_path, map_location='cpu')
                    if 'performance' in checkpoint:
                        key = f"layer_{layer}_{context_method}"
                        perf = checkpoint['performance']
                        results[key] = {
                            'test_loss': perf.get('test_loss', 0.0),
                            'dev_loss': perf.get('dev_loss', 0.0),
                            'mae': perf.get('mae', 0.0),
                            'correlation': perf.get('correlation', 0.0),
                            'spearman': perf.get('spearman', 0.0),
                            'input_dim': checkpoint.get('input_dim', 0),
                            'layer': layer,
                            'context_method': context_method,
                            'loss_type': loss_type,
                            'train_losses': [],
                            'dev_losses': [],
                            'skipped': True
                        }
                except Exception as e:
                    print(f"  (Could not load existing result for summary: {e})")
                
                continue

            print(f"\nContext method: {context_method}")
            print("-" * 40)
            
            # Start timing for this specific layer/method combination
            session_name = f"layer_{layer}_{context_method}"
            method_start_time = time.time()
            
            try:
                # Handle article-level sampling if max_samples is specified
                max_samples = config.get('max_samples')
                selected_article_ids = None
                
                if max_samples is not None:
                    # First, find all articles that have extracted hidden states
                    print(f"Finding articles with extracted hidden states...")
                    available_hidden_states = [f for f in os.listdir(hidden_states_dir) 
                                             if f.startswith("article_") and f.endswith(".pt")]
                    
                    # Extract article IDs from filenames (article_<id>.pt)
                    available_article_ids = []
                    for filename in available_hidden_states:
                        # Remove 'article_' prefix and '.pt' suffix to get the ID
                        article_id = filename[8:-3]  # Remove 'article_' (8 chars) and '.pt' (3 chars)
                        available_article_ids.append(article_id)
                    
                    print(f"Found {len(available_article_ids)} articles with extracted hidden states")
                    
                    if len(available_article_ids) > max_samples:
                        # Set random seed for reproducible article sampling
                        random_seed = config.get('random_seed', 42)
                        torch.manual_seed(random_seed)
                        
                        # Sample articles randomly from those that have hidden states
                        article_indices = torch.randperm(len(available_article_ids))[:max_samples]
                        selected_article_ids = set(available_article_ids[i] for i in article_indices)
                        print(f"📊 Sampling {max_samples} articles from {len(available_article_ids)} with hidden states")
                        print(f"   Using random seed {random_seed} for reproducible article sampling")
                    else:
                        print(f"📊 Using all {len(available_article_ids)} articles with hidden states (less than max_samples={max_samples})")
                        selected_article_ids = set(available_article_ids)
                else:
                    print(f"📊 Using ALL available articles with hidden states (no max_samples limit)")

                # Create dataset with potentially filtered articles
                # Normalize scores if using KL loss
                normalize_scores = (loss_type == 'kl')
                
                dataset = TargetedWordDataset(
                    hidden_states_dir=hidden_states_dir,
                    original_json_path=original_json_path,
                    layer=layer,
                    context_method=context_method,
                    selected_article_ids=selected_article_ids,
                    hidden_dim=hidden_dim,
                    normalize_scores=normalize_scores
                )
                
                if len(dataset) == 0:
                    print("No valid samples found!")
                    continue
                
                print(f"📊 Final dataset contains {len(dataset)} word pairs")
                
                # Get input dimension from first sample
                sample_repr, _ = dataset[0]
                input_dim = sample_repr.shape[0]
                print(f"Input dimension: {input_dim}")
                
                # Split dataset into train/dev/test (60/20/20)
                total_size = len(dataset)
                test_size = int(total_size * config.get('test_ratio', 0.2))
                dev_size = int(total_size * config.get('dev_ratio', 0.2))
                train_size = total_size - test_size - dev_size
                
                # Create generator with fixed seed for reproducible splits
                generator = torch.Generator().manual_seed(random_seed)
                train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(
                    dataset, [train_size, dev_size, test_size], generator=generator
                )
                
                print(f"Train size: {train_size}, Dev size: {dev_size}, Test size: {test_size}")
                print(f"Using random seed {random_seed} for reproducible train/dev/test splits")
                
                # Create dataloaders
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=config.get('batch_size', 8), 
                    shuffle=True
                )
                dev_loader = DataLoader(
                    dev_dataset, 
                    batch_size=config.get('batch_size', 8), 
                    shuffle=False
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=config.get('batch_size', 8), 
                    shuffle=False
                )
                
                # Create model
                use_sigmoid = (loss_type == 'mse')  # Sigmoid for MSE, raw logits for KL
                model = MLPImportanceRegressor(input_dim, use_sigmoid=use_sigmoid).to(device)
                
                # Create loss function based on loss_type
                if loss_type == 'kl':
                    # For KL loss, we need cross-entropy on distributions
                    # Since we're doing word-level (not article-level), we treat each prediction independently
                    # We'll use BCE loss with normalized targets
                    criterion = nn.BCEWithLogitsLoss()  # Expects logits, applies sigmoid internally
                elif loss_type == 'mse':
                    criterion = nn.MSELoss()
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}. Use 'kl' or 'mse'")
                
                optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
                
                # Training loop with early stopping
                model.train()
                train_losses = []
                dev_losses = []
                best_dev_loss = float('inf')
                patience_counter = 0
                early_stopping = config.get('early_stopping', 3)
                best_model_state = None
                
                # Track training time for this specific experiment
                training_start_time = time.time()
                
                for epoch in range(config.get('num_epochs', 10)):
                    # Training phase
                    model.train()
                    epoch_train_loss = 0.0
                    num_train_batches = 0
                    
                    for batch_repr, batch_scores in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
                        batch_repr = batch_repr.to(device)
                        batch_scores = batch_scores.to(device)
                        
                        optimizer.zero_grad()
                        predictions = model(batch_repr)
                        loss = criterion(predictions, batch_scores)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_train_loss += loss.item()
                        num_train_batches += 1
                    
                    avg_train_loss = epoch_train_loss / num_train_batches
                    train_losses.append(avg_train_loss)
                    
                    # Validation phase
                    model.eval()
                    epoch_dev_loss = 0.0
                    num_dev_batches = 0
                    dev_predictions = []
                    dev_targets = []
                    
                    with torch.no_grad():
                        for batch_repr, batch_scores in dev_loader:
                            batch_repr = batch_repr.to(device)
                            batch_scores = batch_scores.to(device)
                            
                            predictions = model(batch_repr)
                            loss = criterion(predictions, batch_scores)
                            
                            epoch_dev_loss += loss.item()
                            num_dev_batches += 1
                            
                            dev_predictions.extend(predictions.cpu().numpy().flatten())
                            dev_targets.extend(batch_scores.cpu().numpy().flatten())
                    
                    avg_dev_loss = epoch_dev_loss / num_dev_batches
                    dev_losses.append(avg_dev_loss)
                    
                    # Calculate validation metrics
                    dev_predictions_np = np.array(dev_predictions)
                    dev_targets_np = np.array(dev_targets)
                    dev_mae = np.mean(np.abs(dev_predictions_np - dev_targets_np))
                    dev_correlation = np.corrcoef(dev_predictions_np, dev_targets_np)[0, 1] if len(dev_predictions_np) > 1 else 0.0
                    dev_spearman, _ = spearmanr(dev_predictions_np, dev_targets_np) if len(dev_predictions_np) > 1 else (0.0, 0.0)
                    
                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Dev Loss = {avg_dev_loss:.6f}, Dev MAE = {dev_mae:.6f}, Dev Corr (P) = {dev_correlation:.6f}, Dev Corr (S) = {dev_spearman:.6f}")
                    
                    # Early stopping check
                    if avg_dev_loss < best_dev_loss:
                        best_dev_loss = avg_dev_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        print(f"  → New best dev loss: {best_dev_loss:.6f}")
                    else:
                        patience_counter += 1
                        print(f"  → No improvement (patience: {patience_counter}/{early_stopping})")
                        
                        if patience_counter >= early_stopping:
                            print(f"  → Early stopping triggered at epoch {epoch+1}")
                            break
                
                # Load best model for final evaluation
                if best_model_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                    print(f"Loaded best model with dev loss: {best_dev_loss:.6f}")
                
                # Final evaluation on test set
                model.eval()
                test_loss = 0.0
                num_test_batches = 0
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_repr, batch_scores in test_loader:
                        batch_repr = batch_repr.to(device)
                        batch_scores = batch_scores.to(device)
                        
                        predictions = model(batch_repr)
                        loss = criterion(predictions, batch_scores)
                        
                        test_loss += loss.item()
                        num_test_batches += 1
                        
                        # Store predictions and targets
                        batch_predictions = predictions.cpu().numpy()
                        batch_targets = batch_scores.cpu().numpy()
                        
                        all_predictions.extend(batch_predictions)
                        all_targets.extend(batch_targets)
                
                avg_test_loss = test_loss / num_test_batches
                
                # Calculate comprehensive metrics

                # Compute per-article metrics
                # Group predictions and targets by article (using test_dataset indices)
                from collections import defaultdict
                article_pred = defaultdict(list)
                article_target = defaultdict(list)
                for idx, dataset_idx in enumerate(test_dataset.indices):
                    article_id, word, true_score = dataset.pairs[dataset_idx]
                    article_pred[article_id].append(all_predictions[idx])
                    article_target[article_id].append(all_targets[idx])

                # Per-article MAE and Pearson
                mae_list = []
                pearson_list = []
                for aid in article_pred:
                    preds = np.array(article_pred[aid])
                    targs = np.array(article_target[aid])
                    if len(preds) > 1:
                        mae_list.append(np.mean(np.abs(preds - targs)))
                        pearson = np.corrcoef(preds, targs)[0, 1]
                        pearson_list.append(pearson)
                mae = float(np.mean(mae_list)) if mae_list else 0.0
                correlation = float(np.mean(pearson_list)) if pearson_list else 0.0

                spearman_list = []
                ndcg_list = []
                for aid in article_pred:
                    preds = np.array(article_pred[aid])
                    targs = np.array(article_target[aid])
                    if len(preds) > 1:
                        s, _ = spearmanr(preds, targs)
                        spearman_list.append(s)
                    # NDCG@10 per article (assume articles have >=10 tokens)
                    ndcg_list.append(ndcg_at_k(preds, targs, k=10))

                spearman_corr = float(np.mean([x for x in spearman_list if x is not None])) if spearman_list else 0.0
                ndcg_10 = float(np.mean(ndcg_list)) if ndcg_list else 0.0

                # KL and Rényi divergences (normalize predictions and targets to probabilities)

                # Per-article KL and Rényi divergence
                kl_list = []
                renyi_list = []
                EPS = 1e-12
                for aid in article_pred:
                    preds = np.array(article_pred[aid])
                    targs = np.array(article_target[aid])
                    if len(preds) > 1:
                        pred_sum = preds.sum() + EPS
                        targ_sum = targs.sum() + EPS
                        pred_probs = preds / pred_sum
                        targ_probs = targs / targ_sum
                        kl = kl_divergence(targ_probs, pred_probs)
                        renyi = renyi_divergence(targ_probs, pred_probs, alpha=2.0)
                        kl_list.append(kl)
                        renyi_list.append(renyi)
                kl_div = float(np.mean(kl_list)) if kl_list else 0.0
                renyi_2 = float(np.mean(renyi_list)) if renyi_list else 0.0

                print(f"Final Test Loss: {avg_test_loss:.6f}")
                print(f"Best Dev Loss: {best_dev_loss:.6f}")
                print(f"Test MAE: {mae:.6f}")
                print(f"Test Correlation (Pearson): {correlation:.6f}")
                print(f"Test Correlation (Spearman, avg per article): {spearman_corr:.6f}")
                print(f"Test NDCG@10 (avg per article): {ndcg_10:.6f}")
                print(f"Test KL Divergence: {kl_div:.6f}")
                print(f"Test Rényi Divergence (α=2.0): {renyi_2:.6f}")
                
                # Calculate timing for this experiment
                method_end_time = time.time()
                method_duration = method_end_time - method_start_time
                training_duration = method_end_time - training_start_time
                
                print(f"\n⏱️  Training completed for {session_name}")
                print(f"   Method time: {method_duration/60:.2f} minutes")
                print(f"   Pure training time: {training_duration/60:.2f} minutes")
                
                # Store comprehensive results
                key = f"layer_{layer}_{context_method}"
                results[key] = {
                    'test_loss': avg_test_loss,
                    'dev_loss': best_dev_loss,
                    'mae': mae,
                    'correlation': correlation,
                    'spearman': spearman_corr,
                    'ndcg_10': ndcg_10,
                    'kl_div': kl_div,
                    'renyi_2': renyi_2,
                    'train_losses': train_losses,
                    'dev_losses': dev_losses,
                    'input_dim': input_dim,
                    'layer': layer,
                    'context_method': context_method,
                    'loss_type': loss_type,
                    'dataset_size': len(dataset),
                    'train_size': train_size,
                    'dev_size': dev_size,
                    'test_size': test_size,
                    'num_epochs_trained': len(train_losses),
                    'early_stopped': len(train_losses) < config.get('num_epochs', 10),
                    # Add timing information
                    'timing': {
                        'total_method_time_seconds': method_duration,
                        'total_method_time_minutes': method_duration / 60,
                        'pure_training_time_seconds': training_duration,
                        'pure_training_time_minutes': training_duration / 60,
                        'time_per_epoch_seconds': training_duration / len(train_losses) if train_losses else 0,
                        'time_per_epoch_minutes': (training_duration / len(train_losses)) / 60 if train_losses else 0
                    }
                }
                
                # Save the best model for future inference and visualization
                try:
                    # Create model-specific and dataset-specific directory structure
                    models_dir = f'./results/{model_name}/{dataset_name}/layer_wise_probe/models'
                    os.makedirs(models_dir, exist_ok=True)
                    
                    if best_model_state is not None:
                        model_checkpoint = {
                            'layer': layer,
                            'method': context_method,
                            'model_state_dict': best_model_state,
                            'input_dim': input_dim,
                            'hidden_dim': hidden_dim,
                            'performance': {
                                'correlation': correlation,
                                'mae': mae,
                                'dev_loss': best_dev_loss,
                                'test_loss': avg_test_loss
                            },
                            'training_config': config,
                            'dataset_info': {
                                'dataset_size': len(dataset),
                                'train_size': train_size,
                                'dev_size': dev_size,
                                'test_size': test_size
                            },
                            'model_name': MODEL_NAME,
                            'dataset_name': DATASET_NAME,
                            'timestamp': time.time(),
                            'num_epochs_trained': len(train_losses)
                        }
                        
                        model_path = os.path.join(models_dir, f"layer_{layer}_{context_method}_best.pt")
                        torch.save(model_checkpoint, model_path)
                        print(f"💾 Saved best model: {model_path}")
                        
                except Exception as save_error:
                    print(f"⚠️  Error saving model: {save_error}")
                
            except Exception as e:
                print(f"Error with layer {layer}, context {context_method}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    # Calculate total training time across all experiments
    total_training_time = 0
    fastest_experiment = float('inf')
    slowest_experiment = 0
    
    for key, result in results.items():
        timing_info = result.get('timing', {})
        method_time = timing_info.get('total_method_time_minutes', 0)
        total_training_time += method_time
        
        if method_time > 0:
            fastest_experiment = min(fastest_experiment, method_time)
            slowest_experiment = max(slowest_experiment, method_time)
        
        print(f"{key} (input_dim={result['input_dim']}):")
        print(f"  Dev Loss: {result['dev_loss']:.6f}")
        print(f"  Test Loss: {result['test_loss']:.6f}")
        print(f"  Test MAE: {result['mae']:.6f}")
        print(f"  Test Correlation (Pearson): {result['correlation']:.6f}")
        print(f"  Test Correlation (Spearman): {result.get('spearman', 0.0):.6f}")
        if timing_info:
            print(f"  Training Time: {timing_info.get('total_method_time_minutes', 0):.2f} min")
        print()
    
    if total_training_time > 0:
        print(f"⏱️  Training Time Summary:")
        print(f"  Total time: {total_training_time:.2f} minutes ({total_training_time/60:.3f} hours)")
        print(f"  Average per experiment: {total_training_time/len(results):.2f} minutes")
        if fastest_experiment != float('inf'):
            print(f"  Fastest experiment: {fastest_experiment:.2f} minutes")
            print(f"  Slowest experiment: {slowest_experiment:.2f} minutes")
    
    # Save comprehensive training results to JSON (including GPU time info)
    gpu_summary = gpu_tracker.get_summary()
    save_comprehensive_results(results, layers_to_test, context_methods, config_key, gpu_summary, model_name, dataset_name, hidden_dim)
    
    # End GPU tracking and print summary
    gpu_tracker.end_session()
    gpu_tracker.get_summary()
    
    # Find best configuration based on dev loss
    if results:
        best_key = min(results.keys(), key=lambda k: results[k]['dev_loss'])
        print(f"Best configuration (by dev loss): {best_key}")
        print(f"Best dev loss: {results[best_key]['dev_loss']:.6f}")
        print(f"Corresponding test loss: {results[best_key]['test_loss']:.6f}")
        print(f"Corresponding test correlation: {results[best_key]['correlation']:.6f}")
        
        # Add performance per GPU hour analysis
        if gpu_summary['total_hours'] > 0:
            best_correlation = results[best_key]['correlation']
            performance_per_hour = best_correlation / gpu_summary['total_hours']
            print(f"Performance efficiency: {performance_per_hour:.3f} correlation per GPU hour")
    
    return results

def save_comprehensive_results(results: Dict, layers_tested: List[int], context_methods: List[str], config_key: str, gpu_summary: Dict = None, model_name: str = None, dataset_name: str = None, hidden_dim: int = None):
    """Save comprehensive training results to JSON for further analysis"""
    
    import json
    import numpy as np
    from datetime import datetime
    
    # Create results directory
    os.makedirs(f'./results/{model_name}/{dataset_name}/layer_wise_probe/json', exist_ok=True)
    
    # Prepare comprehensive summary
    comprehensive_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_used': config_key,
            'model_name': model_name or MODEL_NAME,
            'dataset_name': dataset_name or DATASET_NAME,
            'layers_tested': layers_tested,
            'context_methods': context_methods,
            'total_experiments': len(results),
            'hidden_dim': hidden_dim or 2048
        },
        'training_results': {},
        'summary_statistics': {},
        'best_configurations': {},
        'layer_analysis': {},
        'method_analysis': {},
        'gpu_usage': gpu_summary if gpu_summary else {}
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item'):  # Handle torch tensors and numpy scalars
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    # Store detailed training results
    comprehensive_results['training_results'] = convert_numpy(results)
    
    # Calculate summary statistics
    all_correlations = []
    all_maes = []
    all_test_losses = []
    all_dev_losses = []
    
    for key, result in results.items():
        if 'correlation' in result:
            all_correlations.append(result['correlation'])
            all_maes.append(result['mae'])
            all_test_losses.append(result['test_loss'])
            all_dev_losses.append(result['dev_loss'])
    
    if all_correlations:
        comprehensive_results['summary_statistics'] = {
            'correlation': {
                'mean': float(np.mean(all_correlations)),
                'std': float(np.std(all_correlations)),
                'min': float(np.min(all_correlations)),
                'max': float(np.max(all_correlations)),
                'median': float(np.median(all_correlations))
            },
            'mae': {
                'mean': float(np.mean(all_maes)),
                'std': float(np.std(all_maes)),
                'min': float(np.min(all_maes)),
                'max': float(np.max(all_maes)),
                'median': float(np.median(all_maes))
            },
            'test_loss': {
                'mean': float(np.mean(all_test_losses)),
                'std': float(np.std(all_test_losses)),
                'min': float(np.min(all_test_losses)),
                'max': float(np.max(all_test_losses)),
                'median': float(np.median(all_test_losses))
            },
            'dev_loss': {
                'mean': float(np.mean(all_dev_losses)),
                'std': float(np.std(all_dev_losses)),
                'min': float(np.min(all_dev_losses)),
                'max': float(np.max(all_dev_losses)),
                'median': float(np.median(all_dev_losses))
            }
        }
    
    # Find best configurations by different metrics
    if results:
        best_by_correlation = max(results.keys(), key=lambda k: results[k]['correlation'])
        best_by_mae = min(results.keys(), key=lambda k: results[k]['mae'])
        best_by_dev_loss = min(results.keys(), key=lambda k: results[k]['dev_loss'])
        best_by_test_loss = min(results.keys(), key=lambda k: results[k]['test_loss'])
        
        comprehensive_results['best_configurations'] = {
            'best_correlation': {
                'configuration': best_by_correlation,
                'value': results[best_by_correlation]['correlation'],
                'layer': results[best_by_correlation]['layer'],
                'method': results[best_by_correlation]['context_method']
            },
            'best_mae': {
                'configuration': best_by_mae,
                'value': results[best_by_mae]['mae'],
                'layer': results[best_by_mae]['layer'],
                'method': results[best_by_mae]['context_method']
            },
            'best_dev_loss': {
                'configuration': best_by_dev_loss,
                'value': results[best_by_dev_loss]['dev_loss'],
                'layer': results[best_by_dev_loss]['layer'],
                'method': results[best_by_dev_loss]['context_method']
            },
            'best_test_loss': {
                'configuration': best_by_test_loss,
                'value': results[best_by_test_loss]['test_loss'],
                'layer': results[best_by_test_loss]['layer'],
                'method': results[best_by_test_loss]['context_method']
            }
        }
    
    # Layer-wise analysis
    layer_analysis = {}
    for layer in layers_tested:
        layer_results = {k: v for k, v in results.items() if v.get('layer') == layer}
        
        if layer_results:
            layer_correlations = [r['correlation'] for r in layer_results.values()]
            layer_maes = [r['mae'] for r in layer_results.values()]
            
            best_layer_config = max(layer_results.keys(), key=lambda k: layer_results[k]['correlation'])
            worst_layer_config = min(layer_results.keys(), key=lambda k: layer_results[k]['correlation'])
            
            layer_analysis[str(layer)] = {
                'num_methods_tested': len(layer_results),
                'avg_correlation': float(np.mean(layer_correlations)),
                'std_correlation': float(np.std(layer_correlations)),
                'avg_mae': float(np.mean(layer_maes)),
                'std_mae': float(np.std(layer_maes)),
                'best_method': {
                    'configuration': best_layer_config,
                    'correlation': layer_results[best_layer_config]['correlation'],
                    'method': layer_results[best_layer_config]['context_method']
                },
                'worst_method': {
                    'configuration': worst_layer_config,
                    'correlation': layer_results[worst_layer_config]['correlation'],
                    'method': layer_results[worst_layer_config]['context_method']
                },
                'method_rankings': sorted(
                    [(k.split('_')[-1], v['correlation']) for k, v in layer_results.items()],
                    key=lambda x: x[1], reverse=True
                )
            }
    
    comprehensive_results['layer_analysis'] = layer_analysis
    
    # Method-wise analysis
    method_analysis = {}
    for method in context_methods:
        method_results = {k: v for k, v in results.items() if v.get('context_method') == method}
        
        if method_results:
            method_correlations = [r['correlation'] for r in method_results.values()]
            method_maes = [r['mae'] for r in method_results.values()]
            method_layers = [r['layer'] for r in method_results.values()]
            
            best_method_config = max(method_results.keys(), key=lambda k: method_results[k]['correlation'])
            worst_method_config = min(method_results.keys(), key=lambda k: method_results[k]['correlation'])
            
            # Calculate layer progression for this method
            layer_progression = []
            for layer in sorted(set(method_layers)):
                layer_key = f"layer_{layer}_{method}"
                if layer_key in results:
                    layer_progression.append({
                        'layer': layer,
                        'correlation': results[layer_key]['correlation'],
                        'mae': results[layer_key]['mae']
                    })
            
            method_analysis[method] = {
                'num_layers_tested': len(method_results),
                'avg_correlation': float(np.mean(method_correlations)),
                'std_correlation': float(np.std(method_correlations)),
                'avg_mae': float(np.mean(method_maes)),
                'std_mae': float(np.std(method_maes)),
                'best_layer': {
                    'configuration': best_method_config,
                    'correlation': method_results[best_method_config]['correlation'],
                    'layer': method_results[best_method_config]['layer']
                },
                'worst_layer': {
                    'configuration': worst_method_config,
                    'correlation': method_results[worst_method_config]['correlation'],
                    'layer': method_results[worst_method_config]['layer']
                },
                'layer_progression': layer_progression,
                'consistency_score': float(1.0 / (1.0 + np.std(method_correlations)))  # Higher is more consistent
            }
    
    comprehensive_results['method_analysis'] = method_analysis
    
    # Save to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace("/", "_") if model_name else "unknown_model"
    filename = f'./results/{model_name}/{dataset_name}/layer_wise_probe/json/layer_wise_training_results_{config_key}_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(convert_numpy(comprehensive_results), f, indent=2)
    
    # Also save as latest for easy access
    latest_filename = f'./results/{model_name}/{dataset_name}/layer_wise_probe/json/latest_layer_wise_results_{config_key}.json'
    with open(latest_filename, 'w') as f:
        json.dump(convert_numpy(comprehensive_results), f, indent=2)
    
    print(f"\n💾 Comprehensive results saved:")
    print(f"  - {filename}")
    print(f"  - {latest_filename}")
    print(f"  - Total experiments: {len(results)}")
    print(f"  - Best correlation: {comprehensive_results['best_configurations']['best_correlation']['value']:.4f}")
    print(f"  - Best configuration: {comprehensive_results['best_configurations']['best_correlation']['configuration']}")
    
    # Print GPU usage summary
    if gpu_summary:
        print(f"\n🔥 GPU Usage Summary:")
        print(f"  - Device: {gpu_summary.get('device', 'Unknown')}")
        print(f"  - Total GPU time: {gpu_summary.get('total_hours', 0):.3f} hours")
        print(f"  - Training sessions: {gpu_summary.get('num_sessions', 0)}")
        
        # Calculate efficiency metrics
        if gpu_summary.get('total_hours', 0) > 0:
            best_corr = comprehensive_results['best_configurations']['best_correlation']['value']
            experiments_per_hour = len(results) / gpu_summary['total_hours']
            performance_per_hour = best_corr / gpu_summary['total_hours']
            print(f"  - Experiments per hour: {experiments_per_hour:.2f}")
            print(f"  - Performance per hour: {performance_per_hour:.4f} correlation/hour")
    
    return filename


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train targeted word importance probe for multiple models")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='Model name to train probe for')
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, 
                        help='Dataset name')
    parser.add_argument('--loss_type', type=str, default='kl', choices=['kl', 'mse'],
                        help='Loss type: kl (distribution) or mse (regression)')
    
    # Parse known args to handle mode as positional argument
    args, remaining = parser.parse_known_args()
    
    # Get model-specific configuration
    model_config = get_model_config(args.model_name)
    training_config = get_config(args.model_name)  # Gets SMALL_TEST_CONFIG or FULL_TRAINING_CONFIG
    
    # Determine mode from remaining arguments or default
    if remaining:
        mode = remaining[0].lower()
    else:
        mode = "small"  # Default mode
    
    print(f"🤖 Model: {args.model_name}")
    print(f"📊 Dataset: {args.dataset_name}")
    print(f"🔧 Hidden Dimension: {model_config['hidden_dim']}")
    print(f"📐 Number of Layers: {model_config['num_layers']}")
    print(f"📉 Loss Type: {args.loss_type.upper()}")
    
    if mode == "small":
        print("🧪 Starting SMALL TEST with limited layers and samples...")
        print("This will train probes on 3 layers with ~100 samples")
        print("Expected runtime: ~5 minutes depending on hardware")
        print("This is the quick test option!")
        results = train_targeted_probe("small", args.model_name, args.dataset_name, args.loss_type)
    elif mode == "comprehensive":
        print("🚀 Starting COMPREHENSIVE TRAINING with ALL layers analysis...")
        print("This will train probes on ALL layers with samples based on config")
        print("Expected runtime: 45-90 minutes depending on hardware")
        print("This is the most thorough analysis option!")
        results = train_targeted_probe("comprehensive", args.model_name, args.dataset_name, args.loss_type)
    elif mode == "full":
        print("🚀 Starting FULL TRAINING with key layer analysis...")
        print("This will train probes on key layers with samples based on config")
        print("Expected runtime: 20-40 minutes depending on hardware")
        results = train_targeted_probe("full", args.model_name, args.dataset_name, args.loss_type)
    elif mode == "layerwise":
        print("🔍 Starting LAYERWISE TRAINING for individual layer analysis...")
        print("This will train probes on each transformer layer individually (skipping embedding layer if configured)")
        print("Expected runtime: depends on model size and config")
        results = train_targeted_probe("layerwise", args.model_name, args.dataset_name)
    else:
        print(f"Unknown mode: {mode}")
        print("Valid options: small, full, comprehensive, layerwise")
        sys.exit(1)
