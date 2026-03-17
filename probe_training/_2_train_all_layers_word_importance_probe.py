"""
Train Word Importance Probe with ALL LAYERS Hidden States

Methodology:
1. Input Data:
   - Loads hidden states from pre-extracted .pt files (one per article).
   - Loads ground truth word importance scores from JSON.
   - Features: Concatenates representations from ALL layers (including the embedding layer/Layer 0).
     - 'word_only': Concatenates word hidden states from all layers (dim: num_layers * hidden_dim).
     - 'word_and_context': Concatenates word + context hidden states from all layers (dim: 2 * num_layers * hidden_dim).

2. Model Architecture (EfficientMLPImportanceRegressor):
   - A deep MLP designed for high-dimensional inputs (e.g., ~32k - 65k dimensions).
   - Structure: Input -> Projection -> Hidden Layers -> Output.
   - Key components:
     - Progressive dimensionality reduction.
     - Batch Normalization and Dropout for regularization.
     - Residual connections to facilitate gradient flow.
     - Output: Scalar importance score (logits for KL loss, sigmoid for MSE).

3. Training Process:
   - Optimizer: AdamW with weight decay.
   - Scheduler: ReduceLROnPlateau based on validation loss.
   - Loss Functions: 
     - KL Divergence (via BCEWithLogitsLoss) for distribution matching.
     - MSE for direct score regression.
   - Optimization: Mixed precision training (AMP) for memory efficiency.
   - Early stopping based on validation performance.

4. Evaluation:
   - Metrics: MAE, Pearson/Spearman correlation, NDCG@10, KL/Rényi divergence.
   - Detailed analysis of zero-score word predictions.
   - Saves best models and comprehensive JSON results.
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
import argparse
from typing import List, Dict, Tuple, Optional
import gc
from scipy.stats import spearmanr
from config import MODEL_NAME, DATASET_NAME, get_model_config, get_config
from probe_utils import ndcg_at_k, renyi_divergence, kl_divergence, GPUTimeTracker


class AllLayersWordDataset(Dataset):
    """
    Dataset for word importance probing using ALL layers hidden states.
    Input: concatenated word+context representations from ALL layers.
    Memory efficient with batch loading.
    """
    def __init__(self, 
                 hidden_states_dir: str,
                 original_json_path: str,
                 context_method: str = "word_only",
                 selected_article_ids: set = None,
                 hidden_dim: int = 2048,
                 num_layers: int = 16,
                 normalize_scores: bool = False):
        """
        Args:
            hidden_states_dir: Directory with targeted hidden state files
            original_json_path: Path to original JSON with articles and word_importance
            context_method: "word_only" or "word_and_context"
            selected_article_ids: Set of article IDs to process (None = all articles)
            normalize_scores: Whether to normalize scores to probabilities (sum=1) per article
        """
        self.hidden_states_dir = hidden_states_dir
        self.context_method = context_method
        self.selected_article_ids = selected_article_ids
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        
        # Auto-detect number of layers from first file
        self.num_layers = None
        sample_files = [f for f in os.listdir(hidden_states_dir) if f.startswith("article_") and f.endswith(".pt")]
        
        if sample_files:
            sample_path = os.path.join(hidden_states_dir, sample_files[0])
            try:
                sample_data = torch.load(sample_path, map_location='cpu')
                self.num_layers = sample_data['num_layers']
                print(f"Detected {self.num_layers} layers in model")
            except Exception as e:
                print(f"Error detecting layers from {sample_path}: {e}")
                self.num_layers = 17  # Fallback
        else:
            self.num_layers = 17  # Fallback
        
        # Calculate input dimension
        # We use all layers including the embedding layer (layer 0)
        if context_method == "word_and_context":
            self.input_dim = 2 * self.hidden_dim * self.num_layers  # Word + context for all layers
        else:
            self.input_dim = self.hidden_dim * self.num_layers  # Word only for all layers
        
        print(f"Input dimension: {self.input_dim:,}")
        
        # Store article-level word lists for normalization
        self.article_words = {}  # article_id -> list of (word, score)
        
        # Build dataset pairs
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
        
        print(f"Loaded {len(self.pairs)} word pairs with available hidden states")
        if missing_count > 0:
            print(f"Warning: {missing_count} articles skipped due to missing/corrupted hidden states")
        print(f"Using context method: {self.context_method}")
        if self.normalize_scores:
            print(f"✓ Scores normalized to probabilities (sum=1) per article")

    def _load_raw_features(self, idx):
        """Load raw features for a single sample"""
        article_id, word, score = self.pairs[idx]
        
        # Load targeted hidden states
        hidden_path = os.path.join(self.hidden_states_dir, f"article_{article_id}.pt")
        
        data = torch.load(hidden_path, map_location='cpu')
        
        # Get word representations from all layers
        word_data = data['word_hidden_states'][word]
        word_reprs = word_data['hidden_states']  # List of tensors, one per layer
        
        # Get context representations from all layers
        context_reprs = data['context_hidden_states']  # List of tensors, one per layer
        
        # We use all layers including the embedding layer (Layer 0)
        # So we do NOT slice the lists (e.g., no [1:])

        # Combine representations based on context method
        if self.context_method == "word_and_context":
            # Concatenate word and context for each layer, then all layers
            layer_reprs = []
            for word_repr, context_repr in zip(word_reprs, context_reprs):
                layer_repr = torch.cat([word_repr, context_repr], dim=0)
                layer_reprs.append(layer_repr)
            combined_repr = torch.cat(layer_reprs, dim=0)  # [2*hidden_dim*num_transformer_layers]
        else:  # word_only
            # Concatenate word representations from all layers
            if isinstance(word_reprs, torch.Tensor):
                combined_repr = word_reprs.reshape(-1)
            else:
                combined_repr = torch.cat(word_reprs, dim=0)  # [hidden_dim*num_transformer_layers]
        
        return combined_repr.float(), torch.tensor(score, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            # Load features
            combined_repr, score = self._load_raw_features(idx)
            return combined_repr, score
            
        except Exception as e:
            article_id, word, score = self.pairs[idx]
            print(f"Error loading data for article {article_id}, word '{word}': {e}")
            
            # Return dummy data with correct dimensions
            dummy_repr = torch.zeros(self.input_dim, dtype=torch.float32)
            return dummy_repr, torch.tensor(0.0, dtype=torch.float32)

    def get_input_dim(self):
        """Get the input dimension"""
        return self.input_dim

class EfficientMLPImportanceRegressor(nn.Module):
    """
    Efficient MLP for importance score regression with large input dimensions.
    Uses techniques like dropout, batch normalization, and residual connections.
    """
    def __init__(self, input_dim: int, use_batch_norm: bool = True, use_residual: bool = True, use_sigmoid: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.use_residual = use_residual
        self.use_sigmoid = use_sigmoid
        
        # Calculate hidden sizes based on input dimension
        # Use progressive reduction to handle large input dimensions efficiently
        hidden1 = min(2048, max(512, input_dim // 4))
        hidden2 = min(1024, max(256, hidden1 // 2))
        hidden3 = min(512, max(128, hidden2 // 2))
        
        print(f"MLP Architecture: {input_dim:,} -> {hidden1} -> {hidden2} -> {hidden3} -> 1")
        
        # Input projection layer (handles very large inputs)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),  # Normalize input features
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Hidden layers with optional residual connections
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.output = nn.Linear(hidden3, 1)
        
        # Optional sigmoid activation
        if use_sigmoid:
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        
        # Residual projection layers (if dimensions don't match)
        if use_residual:
            self.residual_proj1 = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
            self.residual_proj2 = nn.Linear(hidden2, hidden3) if hidden2 != hidden3 else nn.Identity()
    
    def forward(self, x):
        # Input projection
        x1 = self.input_projection(x)
        
        # Hidden layer 1 with optional residual
        x2 = self.hidden1(x1)
        if self.use_residual and hasattr(self, 'residual_proj1'):
            x2 = x2 + self.residual_proj1(x1)
        
        # Hidden layer 2 with optional residual
        x3 = self.hidden2(x2)
        if self.use_residual and hasattr(self, 'residual_proj2'):
            x3 = x3 + self.residual_proj2(x2)
        
        # Output
        logits = self.output(x3).squeeze(-1)
        return self.output_activation(logits)

def train_all_layers_probe(config_key: str = "small", 
                          max_samples: int = None, 
                          verbose: bool = True,
                          model_name: str = None,
                          dataset_name: str = None,
                          loss_type: str = 'kl'):
    """
    Train word importance probe using ALL layers hidden states simultaneously.
    
    Args:
        config_key: Configuration to use ("small", "full", "comprehensive")
        max_samples: Maximum number of samples to use
        verbose: Enable verbose output
        model_name: Model name to use
        dataset_name: Dataset name to use
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
    num_layers = model_config["num_layers"]
    
    print(f"🤖 Model: {model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Hidden Dimension: {hidden_dim}")
    print(f"📐 Number of Layers: {num_layers}")
    
    # Mixed precision setting (can be made configurable later)
    use_mixed_precision = False  # Disabled to prevent numerical instability
    
    # Initialize GPU time tracker
    gpu_tracker = GPUTimeTracker()
    gpu_tracker.start_session(f"all_layers_probe_training_{config_key}")
    
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
    
    # Set up comprehensive random seeding for reproducibility
    random_seed = config.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    print(f"Using {config_key} configuration")
    print(f"Enabled: {config['enabled']}")
    print(f"Mixed precision: {use_mixed_precision}")
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
    hidden_states_dir = f"saved_features/hidden_states/{model_name}/{dataset_name}/article_with_zeros"
    original_json_path = f"../data/{model_name}/{dataset_name}/generated_summaries_with_word_importance_deduplicated.json"
    
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
    from config import CONTEXT_METHODS_CONFIG
    if CONTEXT_METHODS_CONFIG['use_only_best']:
        context_methods = CONTEXT_METHODS_CONFIG['default_methods']  # ["word_only"] - best performing
        print(f"🎯 Using optimized method: {context_methods}")
        print("   Running only word_only method for efficiency (best performing)")
    else:
        context_methods = CONTEXT_METHODS_CONFIG['all_methods']  # ["word_only", "word_and_context"]
        print(f"📊 Using comprehensive comparison: {context_methods}")
        print("   Running both methods for comparison")
    
    results = {}
    
    for context_method in context_methods:
        print(f"\n{'='*80}")
        print(f"Training ALL-LAYERS probe with context method: {context_method}")
        print(f"{'='*80}")
        
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
            print("Creating dataset...")
            # Normalize scores if using KL loss
            normalize_scores = (loss_type == 'kl')
            
            dataset = AllLayersWordDataset(
                hidden_states_dir=hidden_states_dir,
                original_json_path=original_json_path,
                context_method=context_method,
                selected_article_ids=selected_article_ids,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                normalize_scores=normalize_scores
            )
            
            if len(dataset) == 0:
                print("No valid samples found!")
                continue
            
            print(f"📊 Final dataset contains {len(dataset)} word pairs")
            
            # Get input dimension
            input_dim = dataset.get_input_dim()
            print(f"Final input dimension: {input_dim:,}")
            
            # Memory estimation
            batch_size = config.get('batch_size', 8)
            estimated_memory_gb = (input_dim * batch_size * 4) / (1024**3)  # 4 bytes per float32
            print(f"Estimated memory per batch: {estimated_memory_gb:.2f} GB")
            
            # Adjust batch size if necessary for large inputs
            if estimated_memory_gb > 2.0:  # If estimated memory > 2GB per batch
                new_batch_size = max(1, int(batch_size * 2.0 / estimated_memory_gb))
                print(f"Adjusting batch size from {batch_size} to {new_batch_size} for memory efficiency")
                batch_size = new_batch_size
            
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
            
            print(f"Dataset split - Train: {train_size}, Dev: {dev_size}, Test: {test_size}")
            print(f"Using random seed {random_seed} for reproducible train/dev/test splits")
            
            # Create dataloaders with memory-efficient settings
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=2,  # Parallel data loading
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True
            )
            dev_loader = DataLoader(
                dev_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True
            )
            
            # Create model with efficient architecture
            use_sigmoid = (loss_type == 'mse')  # Sigmoid for MSE, raw logits for KL
            model = EfficientMLPImportanceRegressor(
                input_dim=input_dim,
                use_batch_norm=True,
                use_residual=True,
                use_sigmoid=use_sigmoid
            ).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Loss and optimizer - dual loss system
            if loss_type == 'kl':
                # For KL loss, use BCE with logits
                criterion = nn.BCEWithLogitsLoss()
            elif loss_type == 'mse':
                criterion = nn.MSELoss()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}. Use 'kl' or 'mse'")
                
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=config.get('learning_rate', 1e-4),
                                  weight_decay=1e-5)  # L2 regularization
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=False
            )
            
            # Mixed precision scaler
            scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and torch.cuda.is_available() else None
            
            # Training loop
            model.train()
            train_losses = []
            dev_losses = []
            dev_maes = []
            dev_correlations = []
            dev_spearmans = []
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
                for batch_repr, batch_scores in train_pbar:
                    batch_repr = batch_repr.to(device, non_blocking=True)
                    batch_scores = batch_scores.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    if scaler is not None:  # Mixed precision training
                        with torch.cuda.amp.autocast():
                            predictions = model(batch_repr)
                            loss = criterion(predictions, batch_scores)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:  # Standard training
                        predictions = model(batch_repr)
                        loss = criterion(predictions, batch_scores)
                        loss.backward()
                        optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    num_train_batches += 1
                    
                    # Update progress bar
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'avg_loss': f'{epoch_train_loss/num_train_batches:.6f}'
                    })
                    
                    # Memory cleanup
                    if num_train_batches % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                avg_train_loss = epoch_train_loss / num_train_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                epoch_dev_loss = 0.0
                num_dev_batches = 0
                dev_predictions = []
                dev_targets = []
                
                with torch.no_grad():
                    for batch_repr, batch_scores in tqdm(dev_loader, desc=f"Epoch {epoch+1} - Dev"):
                        batch_repr = batch_repr.to(device, non_blocking=True)
                        batch_scores = batch_scores.to(device, non_blocking=True)
                        
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                predictions = model(batch_repr)
                                loss = criterion(predictions, batch_scores)
                        else:
                            predictions = model(batch_repr)
                            loss = criterion(predictions, batch_scores)
                        
                        epoch_dev_loss += loss.item()
                        num_dev_batches += 1
                        
                        # Collect predictions and targets for metrics calculation
                        # Convert logits to probabilities for metrics if using KL loss
                        preds_for_metrics = predictions
                        if loss_type == 'kl':
                            preds_for_metrics = torch.sigmoid(predictions)
                            
                        dev_predictions.extend(preds_for_metrics.cpu().numpy().flatten())
                        dev_targets.extend(batch_scores.cpu().numpy().flatten())
                
                avg_dev_loss = epoch_dev_loss / num_dev_batches
                dev_losses.append(avg_dev_loss)
                
                # Calculate validation metrics for this epoch
                dev_predictions_np = np.array(dev_predictions)
                dev_targets_np = np.array(dev_targets)
                
                dev_mae = np.mean(np.abs(dev_predictions_np - dev_targets_np))
                dev_correlation = np.corrcoef(dev_predictions_np, dev_targets_np)[0, 1] if len(dev_predictions_np) > 1 else 0.0
                dev_spearman, _ = spearmanr(dev_predictions_np, dev_targets_np) if len(dev_predictions_np) > 1 else (0.0, 0.0)

                # Store epoch metrics
                dev_maes.append(dev_mae)
                dev_correlations.append(dev_correlation)
                dev_spearmans.append(dev_spearman)
                
                # Learning rate scheduling
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_dev_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Manual learning rate change logging
                if current_lr < old_lr:
                    print(f"  → Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                      f"Dev Loss = {avg_dev_loss:.6f}, Dev MAE = {dev_mae:.6f}, "
                      f"Dev Corr (P) = {dev_correlation:.6f}, Dev Corr (S) = {dev_spearman:.6f}, LR = {current_lr:.2e}")
                
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
                
                # Memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load best model for final evaluation
            if best_model_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                print(f"Loaded best model with dev loss: {best_dev_loss:.6f}")
            
            # Save the best model
            model_save_dir = f'./results/{model_name}/{dataset_name}/multi_layer_probe/models'
            os.makedirs(model_save_dir, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f'{model_save_dir}/best_model_{context_method}_{config_key}_{timestamp}.pt'
            latest_model_filename = f'{model_save_dir}/latest_best_model_{context_method}_{config_key}.pt'
            
            # Save model state dict, architecture info, and training metadata
            model_save_dict = {
                'model_state_dict': model.state_dict(),
                'model_architecture': {
                    'input_dim': input_dim,
                    'model_class': 'EfficientMLPImportanceRegressor',
                    'use_batch_norm': True,
                    'use_residual': True
                },
                'training_metadata': {
                    'context_method': context_method,
                    'config_key': config_key,
                    'best_dev_loss': best_dev_loss,
                    'num_epochs_trained': len(train_losses),
                    'dataset_size': len(dataset),
                    'model_parameters': total_params,
                    'random_seed': random_seed,
                    'timestamp': timestamp
                },
                'performance_metrics': {
                    'dev_loss': best_dev_loss,
                    'train_losses': train_losses,
                    'dev_losses': dev_losses,
                    'dev_maes': dev_maes,
                    'dev_correlations': dev_correlations,
                    'dev_spearmans': dev_spearmans
                }
            }
            
            torch.save(model_save_dict, model_filename)
            torch.save(model_save_dict, latest_model_filename)
            
            print(f"💾 Best model saved:")
            print(f"  - {model_filename}")
            print(f"  - {latest_model_filename}")
            
            # Final evaluation on test set
            print("Evaluating on test set...")
            model.eval()
            test_loss = 0.0
            num_test_batches = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_repr, batch_scores in tqdm(test_loader, desc="Test Evaluation"):
                    batch_repr = batch_repr.to(device, non_blocking=True)
                    batch_scores = batch_scores.to(device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            predictions = model(batch_repr)
                            loss = criterion(predictions, batch_scores)
                    else:
                        predictions = model(batch_repr)
                        loss = criterion(predictions, batch_scores)
                    
                    test_loss += loss.item()
                    num_test_batches += 1
                    
                    # Store predictions and targets
                    # Convert logits to probabilities for metrics if using KL loss
                    preds_for_metrics = predictions
                    if loss_type == 'kl':
                        preds_for_metrics = torch.sigmoid(predictions)
                        
                    batch_preds = preds_for_metrics.cpu().numpy()
                    batch_targets = batch_scores.cpu().numpy()
                    all_predictions.extend(batch_preds)
                    all_targets.extend(batch_targets)
            
            avg_test_loss = test_loss / num_test_batches
            
            # Per-sample metrics

            # --- Standardized Per-sample Metrics Calculation ---
            # Group all predictions and targets by article_id (per sample)
            article_pred_targets = {}
            # Use all_predictions and all_targets, which are in the same order as test_dataset
            for idx, dataset_idx in enumerate(test_dataset.indices):
                article_id, word, true_score = dataset.pairs[dataset_idx]
                if article_id not in article_pred_targets:
                    article_pred_targets[article_id] = {'preds': [], 'targets': []}
                article_pred_targets[article_id]['preds'].append(all_predictions[idx])
                article_pred_targets[article_id]['targets'].append(all_targets[idx])

            spearman_scores = []
            ndcg_scores = []
            mae_scores = []
            for aid, vals in article_pred_targets.items():
                preds = np.array(vals['preds'])
                targets = np.array(vals['targets'])
                if len(preds) > 1:
                    sp, _ = spearmanr(preds, targets)
                    spearman_scores.append(sp)
                # NDCG@10 (assume articles have >=10 tokens)
                ndcg_scores.append(ndcg_at_k(preds, targets, k=10))
                mae_scores.append(np.mean(np.abs(preds - targets)))

            avg_spearman = np.mean([s for s in spearman_scores if s is not None]) if spearman_scores else 0.0
            avg_ndcg_10 = np.mean(ndcg_scores) if ndcg_scores else 0.0
            avg_mae = np.mean(mae_scores) if mae_scores else 0.0

            # Global Pearson correlation (for reference)
            predictions_np = np.array(all_predictions)
            targets_np = np.array(all_targets)
            correlation = np.corrcoef(predictions_np, targets_np)[0, 1] if len(predictions_np) > 1 and not np.isnan(np.corrcoef(predictions_np, targets_np)[0, 1]) else 0.0

            # --- Per-sample KL and Rényi divergence ---
            kl_divs = []
            renyi_2s = []
            EPS = 1e-12
            skipped_kl_renyi = 0
            for aid, vals in article_pred_targets.items():
                preds = np.array(vals['preds'])
                targets = np.array(vals['targets'])
                # Only compute if all values are non-negative and sum > 0
                if np.all(preds >= 0) and np.all(targets >= 0) and preds.sum() > 0 and targets.sum() > 0:
                    pred_probs = preds / (preds.sum() + EPS)
                    target_probs = targets / (targets.sum() + EPS)
                    kl = kl_divergence(target_probs, pred_probs)
                    renyi = renyi_divergence(target_probs, pred_probs, alpha=2.0)
                    if kl is not None and not np.isnan(kl):
                        kl_divs.append(kl)
                    if renyi is not None and not np.isnan(renyi):
                        renyi_2s.append(renyi)
                else:
                    skipped_kl_renyi += 1
            if skipped_kl_renyi > 0:
                print(f"[Warning] Skipped {skipped_kl_renyi} articles for KL/Rényi due to negative or zero-sum values.")
            kl_div = np.mean(kl_divs) if kl_divs else 0.0
            renyi_2 = np.mean(renyi_2s) if renyi_2s else 0.0
            
            # Calculate timing
            method_end_time = time.time()
            method_duration = method_end_time - method_start_time
            training_duration = method_end_time - training_start_time
            
            print(f"\n🎯 Results for {context_method}:")
            print(f"Final Test Loss: {avg_test_loss:.6f}")
            print(f"Best Dev Loss: {best_dev_loss:.6f}")
            print(f"Test MAE (avg per sample): {avg_mae:.6f}")
            print(f"Test Correlation (Pearson, global): {correlation:.6f}")
            print(f"Test Correlation (Spearman, avg per sample): {avg_spearman:.6f}")
            print(f"Test NDCG@10 (avg per sample): {avg_ndcg_10:.6f}")
            print(f"Test KL Divergence: {kl_div:.6f}")
            print(f"Test Rényi Divergence (α=2.0): {renyi_2:.6f}")
            print(f"Training time: {training_duration/60:.2f} minutes")
            print(f"Total method time: {method_duration/60:.2f} minutes")
            
            # Store results
            results[context_method] = {
                'test_loss': avg_test_loss,
                'dev_loss': best_dev_loss,
                'mae': avg_mae,
                'correlation': correlation,
                'spearman': avg_spearman,
                'ndcg_10': avg_ndcg_10,
                'kl_div': kl_div,
                'renyi_2': renyi_2,
                'train_losses': train_losses,
                'dev_losses': dev_losses,
                'input_dim': input_dim,
                'context_method': context_method,
                'loss_type': loss_type,
                'dataset_size': len(dataset),
                'train_size': train_size,
                'dev_size': dev_size,
                'test_size': test_size,
                'num_epochs_trained': len(train_losses),
                'early_stopped': len(train_losses) < config.get('num_epochs', 10),
                'model_parameters': total_params,
                'batch_size_used': batch_size,
                'mixed_precision': use_mixed_precision,
                'saved_model_path': model_filename,
                'latest_model_path': latest_model_filename,
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
            print(f"Error with context method {context_method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("ALL-LAYERS PROBE TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Input dimension: {result['input_dim']:,}")
        print(f"  Model parameters: {result['model_parameters']:,}")
        print(f"  Dev Loss: {result['dev_loss']:.6f}")
        print(f"  Test Loss: {result['test_loss']:.6f}")
        print(f"  Test MAE: {result['mae']:.6f}")
        print(f"  Test Correlation: {result['correlation']:.6f}")
        print(f"  Training time: {result['timing']['pure_training_time_minutes']:.2f} min")
    
    # Save results
    save_all_layers_results(results, config_key, gpu_tracker.get_summary(), model_name, dataset_name, hidden_dim)
    
    # End GPU tracking
    gpu_tracker.end_session()
    gpu_tracker.get_summary()
    
    # Find best method
    if results:
        best_method = max(results.keys(), key=lambda k: results[k]['correlation'])
        print(f"\n🏆 Best method: {best_method}")
        print(f"Best correlation: {results[best_method]['correlation']:.6f}")
        print(f"Input dimension: {results[best_method]['input_dim']:,}")
    
    return results

def save_all_layers_results(results: Dict, config_key: str, gpu_summary: Dict, model_name: str = None, dataset_name: str = None, hidden_dim: int = None):
    """Save comprehensive results for all-layers probe training"""
    from datetime import datetime
    
    os.makedirs(f'./results/{model_name}/{dataset_name}/multi_layer_probe/json', exist_ok=True)
    
    # Prepare results
    comprehensive_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config_used': config_key,
            'model_name': model_name,
            'dataset_name': dataset_name or DATASET_NAME,
            'training_type': 'multi_layer_probe',
            'total_experiments': len(results),
            'hidden_dim': hidden_dim or 2048
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
    model_folder = model_name or MODEL_NAME
    dataset_folder = dataset_name or DATASET_NAME
    filename = f'./results/{model_folder}/{dataset_folder}/multi_layer_probe/json/multi_layer_training_results_{config_key}_{timestamp}.json'
    latest_filename = f'./results/{model_folder}/{dataset_folder}/multi_layer_probe/json/latest_multi_layer_results_{config_key}.json'
    
    with open(filename, 'w') as f:
        json.dump(convert_numpy(comprehensive_results), f, indent=2)
    
    with open(latest_filename, 'w') as f:
        json.dump(convert_numpy(comprehensive_results), f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"  - {filename}")
    print(f"  - {latest_filename}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train all-layers word importance probe for multiple models")
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
        print("🧪 Starting SMALL TEST for ALL-LAYERS probe...")
        print("This will train probes using ALL layers with limited samples")
        print("Expected runtime: ~5-10 minutes")
        results = train_all_layers_probe("small", model_name=args.model_name, dataset_name=args.dataset_name, loss_type=args.loss_type)
    elif mode == "comprehensive":
        print("🚀 Starting COMPREHENSIVE ALL-LAYERS probe training...")
        print("This will train probes using ALL layers with ALL available samples")
        print("Expected runtime: 30-60 minutes")
        results = train_all_layers_probe("comprehensive", model_name=args.model_name, dataset_name=args.dataset_name, loss_type=args.loss_type)
    elif mode == "full":
        print("🚀 Starting FULL ALL-LAYERS probe training...")
        print("This will train probes using ALL layers with standard samples")
        print("Expected runtime: 15-30 minutes")
        results = train_all_layers_probe("full", model_name=args.model_name, dataset_name=args.dataset_name, loss_type=args.loss_type)
    else:
        print(f"Unknown mode: {mode}")
        print("Valid options: small, full, comprehensive")
        print("Usage: python script.py --loss_type [kl|mse] <mode>")
        print("Example: python script.py --loss_type kl comprehensive")
        sys.exit(1)
