"""
Configuration for probing experiments

TO USE ONLY WORD_ONLY METHOD (RECOMMENDED):
- Keep CONTEXT_METHODS_CONFIG['use_only_best'] = True

TO COMPARE BOTH METHODS:
- Set CONTEXT_METHODS_CONFIG['use_only_best'] = False
"""

# Model and data configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Updated default to match current experiments
DATASET_NAME = "cnn_dailymail"

# Model-specific configurations
MODEL_CONFIGS = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "hidden_dim": 2048,
        "num_layers": 16,
        "vocab_size": 128256
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "vocab_size": 128256
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "hidden_dim": 1536,
        "num_layers": 28,
        "vocab_size": 151936
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "hidden_dim": 2048,
        "num_layers": 36,
        "vocab_size": 151936
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "hidden_dim": 3584,
        "num_layers": 28, 
        "vocab_size": 152064
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "hidden_dim": 5120,
        "num_layers": 48,
        "vocab_size": 152064
    }
}

# Get current model config
def get_model_config(model_name=None):
    if model_name is None:
        model_name = MODEL_NAME
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        print(f"⚠️  Warning: Model '{model_name}' not found in MODEL_CONFIGS. Using default Qwen/Qwen2.5-1.5B-Instruct.")
        config = MODEL_CONFIGS["Qwen/Qwen2.5-1.5B-Instruct"]
    return config

# Legacy constants for backward compatibility
HIDDEN_DIM = get_model_config()["hidden_dim"]
VOCAB_SIZE = get_model_config()["vocab_size"]

# Context methods configuration
# Based on experimental results, word_only consistently outperforms word_and_context
CONTEXT_METHODS_CONFIG = {
    'default_methods': ["word_only"],  # Use only word_only (best performing)
    'all_methods': ["word_only", "word_and_context"],  # For comprehensive comparison
    'use_only_best': True  # If True, uses only word_only method for efficiency
}

# Small test configuration - OPTIMIZED FOR RAPID TESTING
def get_small_test_config(model_name=None):
    model_config = get_model_config(model_name)
    num_layers = model_config["num_layers"]
    
    # Select early, middle, late layers (3 layers total)
    early_layer = max(0, num_layers // 4)
    middle_layer = num_layers // 2
    late_layer = num_layers - 1
    
    return {
        'enabled': True,  # Set to False for full training
        'layers_to_test': [early_layer, middle_layer, late_layer],  # Early, middle, late layers (auto-adapted to model)
        'max_samples': 10,  # VERY SMALL for rapid testing (was 5)
        'random_seed': 42,     # Fixed seed for reproducible results
        'batch_size': 128,     # INCREASED for better GPU utilization (was 4)
        'num_epochs': 5,       # Quick test epochs
        'test_ratio': 0.2,  # 20% test set
        'dev_ratio': 0.2,   # 20% dev set
        'learning_rate': 2e-4,  # Higher LR for faster convergence (was 1e-4)
        'early_stopping': 2,  # Very short patience for rapid testing (was 3)
        'top_k_display': 10,  # Fewer tokens to display
        'num_sample_predictions': 3  # Fewer sample predictions
    }

# Full training configuration
def get_full_training_config(model_name=None):
    model_config = get_model_config(model_name)
    num_layers = model_config["num_layers"]
    
    # Select representative layers (approximately 9 layers)
    if num_layers <= 16:
        # For smaller models, use every 2nd layer
        layers_to_test = list(range(0, num_layers, 2))
    else:
        # For larger models, select 9 representative layers
        step = max(1, num_layers // 9)
        layers_to_test = list(range(0, num_layers, step))[:9]
        # Always include the last layer
        if (num_layers - 1) not in layers_to_test:
            layers_to_test.append(num_layers - 1)
    
    return {
        'enabled': True,  # Enable full training
        'layers_to_test': layers_to_test,  # Representative layer sampling (auto-adapted to model)
        'max_samples': 25,  # REDUCED for faster training (was 100)
        'random_seed': 42,     # Fixed seed for reproducible results
        'batch_size': 128,      # INCREASED for better GPU utilization (was 32)
        'num_epochs': 10,      # Reduced for efficiency (was 15)
        'test_ratio': 0.2,  # 20% test set
        'dev_ratio': 0.2,   # 20% dev set
        'learning_rate': 1e-4,  # Balanced learning rate
        'early_stopping': 3,  # Reduced patience for faster experiments (was 5)
        'top_k_display': 20,
        'num_sample_predictions': 5
    }

# Comprehensive training configuration (all layers) - FOR GENERAL PROBE TRAINING
def get_comprehensive_training_config(model_name=None, dataset_name=None):
    model_config = get_model_config(model_name)
    num_layers = model_config["num_layers"]
    
    # Determine max_samples based on dataset
    max_samples = None
    if dataset_name == "cnn_dailymail":
        max_samples = 300
    elif dataset_name == "samsum":
        max_samples = 819
    elif dataset_name == "decoda":
        max_samples = 100
    
    # Base defaults
    cfg = {
        'enabled': True,
        'layers_to_test': list(range(num_layers + 1)),  # All layers (auto-adapted to model)
        'max_samples': max_samples,  # Dataset-specific sample size
        'random_seed': 42,     # Fixed seed for reproducible results
        'batch_size': 128,      # OPTIMIZED for better GPU utilization (was 32)
        'num_epochs': 20,       # Keep sufficient epochs for proper convergence
        'test_ratio': 0.2,
        'dev_ratio': 0.2,
        'learning_rate': 1e-4,
        'early_stopping': 3,  # Efficient but allows proper convergence
        'top_k_display': 15,
        'num_sample_predictions': 5
    }

    # Decoda-specific training adjustments
    if dataset_name == "decoda":
        cfg['num_epochs'] = 30
        cfg['early_stopping'] = 5

    return cfg

# Layer-wise probe training configuration - OPTIMIZED FOR LAYER-WISE ANALYSIS
def get_layerwise_training_config(model_name=None):
    model_config = get_model_config(model_name)
    num_layers = model_config["num_layers"]
    
    return {
        'enabled': True,
        'layers_to_test': list(range(0, num_layers + 1)),  # includes embedding layer (index 0) and all transformer layers
        #'max_samples': 300,  # OPTIMAL: Based on experimental validation (~9K word pairs, excellent performance)
        'max_samples': None,  # Use all available samples for SAMSum
        'random_seed': 42,     # Fixed seed for reproducible results
        'batch_size': 256,      # OPTIMIZED for better GPU utilization
        'num_epochs': 20,       # Sufficient epochs for proper convergence
        'test_ratio': 0.2,
        'dev_ratio': 0.2,
        'learning_rate': 1e-4,
        'early_stopping': 3,  # Efficient but allows proper convergence
        'top_k_display': 15,
        'num_sample_predictions': 5
    }

# Legacy static configs for backward compatibility
SMALL_TEST_CONFIG = get_small_test_config()
FULL_TRAINING_CONFIG = get_full_training_config()
COMPREHENSIVE_TRAINING_CONFIG = get_comprehensive_training_config()
LAYERWISE_TRAINING_CONFIG = get_layerwise_training_config()

def get_config(model_name=None):
    """Return the active configuration based on SMALL_TEST_CONFIG['enabled']"""
    small_config = get_small_test_config(model_name)
    if small_config['enabled']:
        return small_config
    else:
        return get_full_training_config(model_name)
