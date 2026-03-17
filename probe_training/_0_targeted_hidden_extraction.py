"""
Enhanced Targeted Hidden State Extraction for Word Importance Probing
"""

import os
import json
import torch
import gc
import random
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- Model selection and dynamic config ---
import argparse
from config import MODEL_NAME as DEFAULT_MODEL_NAME, DATASET_NAME as DEFAULT_DATASET_NAME

parser = argparse.ArgumentParser(description="Targeted Hidden State Extraction for Word Importance Probing")
parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help='Model name or path')
parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME, help='Dataset name')
parser.add_argument('--max_zero_score_words', type=int, default=50, help='Maximum zero-score words per article (use 0 to extract all zero-score words)')
parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
parser.add_argument('--randomize_weights', action='store_true', help='Randomize model weights (control experiment to test if learned representations matter)')
args = parser.parse_args()

MODEL_NAME = args.model_name
DATASET_NAME = args.dataset_name

# Determine max_samples based on dataset if not provided
if args.max_samples is None:
    if DATASET_NAME == "cnn_dailymail":
        MAX_SAMPLES = 300
    elif DATASET_NAME == "samsum":
        MAX_SAMPLES = 819
    elif DATASET_NAME == "decoda":
        MAX_SAMPLES = 100
    else:
        MAX_SAMPLES = None
else:
    MAX_SAMPLES = args.max_samples

DATASET_CONFIGS = {
    "cnn_dailymail": {
        "text_field": "article",
        "summary_field": "summary", # ref:highlights, generated:summary
        "id_field": "id"
    },
    "samsum": {
        "text_field": "dialogue",
        "summary_field": "generated_summary", # ref: summary, generated: generated_summary
        "id_field": "id"
    },
    "decoda": {
        "text_field": "dialogue",
        "summary_field": "generated_summary", # ref:synopsis, generated: generated_summary
        "id_field": "id"
    }
}

# Handle max_zero_score_words: 0 means None (extract all)
MAX_ZERO_SCORE_WORDS = None if args.max_zero_score_words == 0 else args.max_zero_score_words

def randomize_model_weights(model, seed=42):
    """
    Randomize the model's weights IN-PLACE using the architecture's default
    initializer while preserving existing device placement (device_map="auto" etc.).

    This avoids device mismatches between inputs (on GPU) and modules offloaded
    across devices by Accelerate, which can happen if we replace the model object.

    Args:
        model: A loaded HuggingFace `PreTrainedModel` (possibly with device_map)
        seed: Random seed for reproducibility

    Returns:
        The SAME model instance with randomized weights.
    """
    print(f"🎲 Reinitializing model IN-PLACE with architecture defaults (seed={seed})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build a fresh CPU model from the existing config using HF's native initializer
    try:
        fresh = AutoModelForCausalLM.from_config(model.config)
    except Exception as e:
        raise RuntimeError("Failed to reinitialize model from config.") from e

    # Ensure tied weights in the fresh model
    if hasattr(fresh, 'tie_weights'):
        try:
            fresh.tie_weights()
        except Exception:
            pass

    # Match parameter dtypes to the existing model to avoid dtype mismatch on load
    try:
        ref_param = next(model.parameters())
        target_dtype = ref_param.dtype
        fresh = fresh.to(target_dtype)
    except Exception:
        pass

    # Load randomized weights into the existing (possibly sharded) model
    fresh_sd = fresh.state_dict()
    missing, unexpected = model.load_state_dict(fresh_sd, strict=False)
    del fresh, fresh_sd
    try:
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if missing or unexpected:
        print(f"[WARN] Randomization state load: missing={len(missing)}, unexpected={len(unexpected)}")
    print("✅ Model randomized IN-PLACE using architecture defaults (preserved device map)")
    return model

def find_article_end_position(tokenizer, prompt, article):
    """
    Find the exact position where the article content ends in the tokenized prompt.
    This is more precise than estimation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Full prompt with chat template
        article: Original article text
    
    Returns:
        int: Token position where article content ends
    """
    try:
        # Method 1: Find article content boundaries in the prompt text
        article_start_in_prompt = prompt.find(article)
        if article_start_in_prompt >= 0:
            article_end_in_prompt = article_start_in_prompt + len(article)
            
            # Tokenize text up to the end of article
            text_up_to_article_end = prompt[:article_end_in_prompt]
            tokens_up_to_article_end = tokenizer.encode(text_up_to_article_end, add_special_tokens=False)
            
            return len(tokens_up_to_article_end) - 1  # Last token of article content
        
        # Method 2: If article not found exactly, tokenize parts separately
        # This handles cases where chat template might modify the text slightly
        parts = prompt.split(article)
        if len(parts) >= 2:
            # Article is in the prompt, find where it ends
            prefix_tokens = tokenizer.encode(parts[0], add_special_tokens=False)
            article_tokens = tokenizer.encode(article, add_special_tokens=False)
            
            return len(prefix_tokens) + len(article_tokens) - 1
        
        # Method 3: Fallback - look for chat template patterns
        # Find the last content before assistant response
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        # For most chat templates, the user content ends before assistant tag
        # This is a reasonable position for "context"
        return len(prompt_tokens) - 1
        
    except Exception as e:
        print(f"Error finding article end position: {e}")
        # Final fallback
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return len(prompt_tokens) - 1

def find_word_variants_in_text(target_word, text, case_sensitive=False):
    """
    Enhanced word finding with fuzzy matching and better tokenization handling.
    
    Args:
        target_word: Base word to find (e.g., "pacquiao", "anti-american")
        text: Text to search in
        case_sensitive: Whether to match case exactly
    
    Returns:
        list: List of (start_char, end_char, matched_word) tuples
    """
    if not case_sensitive:
        target_lower = target_word.lower()
        search_text = text.lower()
    else:
        target_lower = target_word
        search_text = text
    
    matches = []
    
    # STRATEGY 1: Exact word matching with variants
    base_pattern = re.escape(target_lower)
    
    # Enhanced variant patterns for better coverage
    variant_patterns = [
        base_pattern,                   # exact match
        base_pattern + r"s",            # plural: word -> words
        base_pattern + r"'s",           # possessive: word's
        base_pattern + r"ed",           # past tense: work -> worked
        base_pattern + r"ing",          # progressive: work -> working
        base_pattern + r"er",           # comparative: fast -> faster
        base_pattern + r"est",          # superlative: fast -> fastest
        base_pattern + r"ly",           # adverb: quick -> quickly
    ]
    
    # Try each variant pattern
    for pattern in variant_patterns:
        full_pattern = r'\b' + pattern + r'\b'
        for match in re.finditer(full_pattern, search_text):
            start_char = match.start()
            end_char = match.end()
            matched_word = text[start_char:end_char]
            matches.append((start_char, end_char, matched_word))
    
    # STRATEGY 2: Handle compound words and hyphenation
    if '-' in target_lower or '_' in target_lower:
        # Try with different separators
        variants = [
            target_lower.replace('-', ' '),     # anti-american -> anti american
            target_lower.replace('-', ''),      # anti-american -> antiamerican  
            target_lower.replace('_', ' '),     # word_word -> word word
            target_lower.replace('_', ''),      # word_word -> wordword
            target_lower.replace('_', '-'),     # word_word -> word-word
        ]
        
        for variant in variants:
            if variant != target_lower:  # Avoid duplicates
                variant_pattern = r'\b' + re.escape(variant) + r'\b'
                for match in re.finditer(variant_pattern, search_text):
                    start_char = match.start()
                    end_char = match.end()
                    matched_word = text[start_char:end_char]
                    matches.append((start_char, end_char, matched_word))
    
    # STRATEGY 3: Handle words that might have different punctuation
    if any(c in target_lower for c in ["'", '"', '.', ',']):
        # Remove punctuation and try again
        clean_target = re.sub(r'[^\w\s-]', '', target_lower)
        if clean_target != target_lower and clean_target.strip():
            clean_pattern = r'\b' + re.escape(clean_target) + r'\b'
            for match in re.finditer(clean_pattern, search_text):
                start_char = match.start()
                end_char = match.end()
                matched_word = text[start_char:end_char]
                matches.append((start_char, end_char, matched_word))
    
    # STRATEGY 4: Fuzzy matching for partial words (be conservative)
    if len(target_lower) >= 5:  # Only for longer words to avoid false positives
        # Try to find the word as part of larger words
        partial_pattern = re.escape(target_lower)
        for match in re.finditer(partial_pattern, search_text):
            start_char = match.start()
            end_char = match.end()
            
            # Check if this is inside a larger word by expanding to word boundaries
            word_start = start_char
            word_end = end_char
            
            # Expand backwards to word boundary
            while word_start > 0 and search_text[word_start - 1].isalnum():
                word_start -= 1
            
            # Expand forwards to word boundary  
            while word_end < len(search_text) and search_text[word_end].isalnum():
                word_end += 1
            
            # Only accept if the target word forms a significant part (>= 70%) of the found word
            found_word = search_text[word_start:word_end]
            if len(target_lower) / max(len(found_word), 1) >= 0.7:
                matched_word = text[word_start:word_end]
                matches.append((word_start, word_end, matched_word))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in matches:
        match_key = (match[0], match[1])  # start, end position
        if match_key not in seen:
            seen.add(match_key)
            unique_matches.append(match)
    
    return unique_matches

def convert_char_positions_to_tokens(tokenizer, text, char_positions, tokens):
    """
    Convert character positions to token positions with proper alignment.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Original text
        char_positions: List of (start_char, end_char, word) tuples
        tokens: Pre-tokenized token IDs
    
    Returns:
        list: List of (start_token, num_tokens, word) tuples
    """
    if not char_positions:
        return []
    
    # Build character-to-token mapping
    token_char_spans = []
    current_char = 0
    
    for token_idx, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        # Handle special tokens that might not have text representation
        if not token_text.strip():
            token_text = " "  # Assume whitespace
        
        token_start = current_char
        token_end = current_char + len(token_text)
        token_char_spans.append((token_start, token_end, token_idx))
        current_char = token_end
        
        # Stop if we've covered the text
        if current_char >= len(text):
            break
    
    # Convert each character position to token position
    token_positions = []
    
    for start_char, end_char, word in char_positions:
        start_token_idx = None
        end_token_idx = None
        
        # Find tokens that overlap with the word span
        for token_start, token_end, token_idx in token_char_spans:
            # Token contains word start
            if token_start <= start_char < token_end:
                start_token_idx = token_idx
            # Token contains word end
            if token_start < end_char <= token_end:
                end_token_idx = token_idx
        
        if start_token_idx is not None:
            if end_token_idx is not None:
                num_tokens = end_token_idx - start_token_idx + 1
            else:
                num_tokens = 1
            
            # Validate the token span actually contains the word
            if start_token_idx + num_tokens <= len(tokens):
                candidate_tokens = tokens[start_token_idx:start_token_idx + num_tokens]
                decoded_text = tokenizer.decode(candidate_tokens).strip()
                
                # Check if the decoded tokens contain the word (case-insensitive)
                if word.lower() in decoded_text.lower():
                    token_positions.append((start_token_idx, num_tokens, word))
    
    return token_positions

def find_word_positions_in_article(tokenizer, word, article, article_tokens, strategy="first"):
    """
    Find where a word appears in the article token sequence using improved word-level matching.
    
    Args:
        tokenizer: HuggingFace tokenizer
        word: Target word to find
        article: Article text (used for word-level matching)
        article_tokens: Pre-tokenized article token IDs
        strategy: "first", "last", "all" - which occurrence(s) to return
    
    Returns:
        For "first"/"last": tuple (start_position, num_tokens) or (-1, 0) if not found
        For "all": list of tuples [(start_position, num_tokens), ...] 
    """
    try:
        # Use improved word-level matching
        char_positions = find_word_variants_in_text(word, article, case_sensitive=False)
        
        if not char_positions:
            return [] if strategy == "all" else (-1, 0)
        
        # Convert character positions to token positions
        token_positions = convert_char_positions_to_tokens(tokenizer, article, char_positions, article_tokens)
        
        if not token_positions:
            return [] if strategy == "all" else (-1, 0)
        
        # Convert to expected format (remove matched_word for backward compatibility)
        positions = [(start_token, num_tokens) for start_token, num_tokens, matched_word in token_positions]
        
        # Apply strategy
        if strategy == "all":
            return positions
        elif strategy == "first":
            return positions[0] if positions else (-1, 0)
        elif strategy == "last":
            return positions[-1] if positions else (-1, 0)
        else:
            return positions[0] if positions else (-1, 0)
            
    except Exception as e:
        print(f"Error finding word '{word}' in article: {e}")
        return [] if strategy == "all" else (-1, 0)

def process_article_word_occurrences(tokenizer, word, score, article, article_tokens, hidden_states, 
                                   full_input_ids, seq_len, multi_occurrence_handling):
    """
    Enhanced process to find word occurrences with improved tokenization handling and fuzzy matching.
    
    IMPROVED APPROACH: Multi-strategy word finding with better tokenization awareness.
    
    Args:
        tokenizer: HuggingFace tokenizer
        word: Target word to find
        score: Importance score for this word
        article: Article text (for word finding)
        article_tokens: Pre-tokenized article tokens (legacy, not used)
        hidden_states: Model hidden states [num_layers, seq_len, hidden_dim]
        full_input_ids: The actual tokenized input sequence tensor [1, seq_len]
        seq_len: Sequence length
        multi_occurrence_handling: "individual", "aggregate_avg", or "aggregate_concat"
    
    Returns:
        dict: Dictionary of word representations ready to add to word_hidden_states
    """
    
    token_positions = []
    target_word_lower = word.lower()
    
    # STRATEGY 1: Enhanced single-token matching with fuzzy variants
    for pos in range(seq_len):
        token_ids = full_input_ids[0][pos:pos+1].tolist()
        token_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip().lower()
        
        if token_text and len(token_text) >= 1:
            # Enhanced matching conditions
            if (target_word_lower == token_text or 
                target_word_lower == token_text.strip() or
                token_text == target_word_lower + 's' or      # plural
                token_text == target_word_lower + "'s" or     # possessive
                token_text == target_word_lower + 'ed' or     # past tense
                token_text == target_word_lower + 'ing' or    # progressive
                token_text == target_word_lower + 'ly' or     # adverb
                # Handle compound word fragments
                (len(target_word_lower) >= 4 and target_word_lower in token_text) or
                (len(token_text) >= 4 and token_text in target_word_lower)):
                
                token_positions.append((pos, 1, token_text, token_text))
    
    # STRATEGY 2: Multi-token combinations (2-4 tokens for compound words)
    if not token_positions:
        max_tokens = min(4, seq_len)  # Search up to 4 tokens for compound words
        
        for num_tokens in range(2, max_tokens + 1):
            for pos in range(seq_len - num_tokens + 1):
                token_ids = full_input_ids[0][pos:pos+num_tokens].tolist()
                token_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip().lower()
                
                if token_text and len(token_text) >= 3:
                    # Enhanced multi-token matching
                    if (target_word_lower == token_text or
                        target_word_lower.replace('-', ' ') == token_text or  # anti-american -> anti american
                        target_word_lower.replace('-', '') == token_text or   # anti-american -> antiamerican
                        target_word_lower.replace('_', ' ') == token_text or  # word_word -> word word
                        target_word_lower.replace('_', '') == token_text or   # word_word -> wordword
                        # Remove punctuation and compare
                        re.sub(r'[^\w\s]', '', target_word_lower) == re.sub(r'[^\w\s]', '', token_text)):
                        
                        token_positions.append((pos, num_tokens, token_text, token_text))
                        break  # Found with this token count, stop searching longer sequences
            
            if token_positions:  # Found with this token count, no need to try longer
                break
    
    # STRATEGY 3: Flexible matching for very short words or when no exact matches found
    if not token_positions and len(target_word_lower) <= 4:
        for pos in range(seq_len):
            token_ids = full_input_ids[0][pos:pos+1].tolist()
            token_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip().lower()
            
            # More flexible matching for short words
            if (token_text and 
                target_word_lower == token_text.strip('.,!?;:"()[]{}') or  # Remove common punctuation
                (len(target_word_lower) >= 2 and len(token_text) >= 2 and 
                 abs(len(target_word_lower) - len(token_text)) <= 1 and
                 target_word_lower[:2] == token_text[:2])):  # Same first 2 chars, similar length
                
                token_positions.append((pos, 1, token_text, token_text))
    
    # STRATEGY 4: Character-level fuzzy search as last resort for important words
    if not token_positions and score >= 0.3:  # Only for moderately important words
        # Try to find the word by examining character-level token alignment
        full_sequence_text = tokenizer.decode(full_input_ids[0].tolist(), skip_special_tokens=True).lower()
        
        # Find character positions of the word in the decoded sequence
        char_matches = find_word_variants_in_text(target_word_lower, full_sequence_text, case_sensitive=False)
        
        if char_matches:
            # Convert character positions back to token positions (approximate)
            for char_start, char_end, matched_text in char_matches[:2]:  # Max 2 matches to avoid noise
                # Find the token that contains this character position
                current_char = 0
                for pos in range(seq_len):
                    token_ids = full_input_ids[0][pos:pos+1].tolist()
                    token_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    token_len = len(token_text)
                    
                    # Check if the character position falls within this token
                    if current_char <= char_start < current_char + token_len:
                        # Estimate how many tokens this word spans
                        word_char_len = char_end - char_start
                        estimated_tokens = 1
                        
                        # Check if the word extends into next tokens
                        remaining_chars = word_char_len - (current_char + token_len - char_start)
                        check_pos = pos + 1
                        
                        while remaining_chars > 0 and check_pos < seq_len:
                            next_token_ids = full_input_ids[0][check_pos:check_pos+1].tolist()
                            next_token_text = tokenizer.decode(next_token_ids, skip_special_tokens=True)
                            remaining_chars -= len(next_token_text)
                            estimated_tokens += 1
                            check_pos += 1
                            
                            if estimated_tokens >= 4:  # Reasonable limit
                                break
                        
                        # Validate that the extracted tokens actually contain the word
                        if pos + estimated_tokens <= seq_len:
                            validation_tokens = full_input_ids[0][pos:pos+estimated_tokens].tolist()
                            validation_text = tokenizer.decode(validation_tokens, skip_special_tokens=True).strip().lower()
                            
                            # Check if our target word is reasonably contained in the validation text
                            if (target_word_lower in validation_text or 
                                validation_text in target_word_lower or
                                # Fuzzy match: significant overlap
                                len(set(target_word_lower.split()) & set(validation_text.split())) > 0):
                                
                                token_positions.append((pos, estimated_tokens, matched_text, validation_text))
                        break
                    
                    current_char += token_len
    
    if not token_positions:
        return {}
    
    # Extract representations for all valid positions (same as before)
    article_representations = []
    valid_positions = []
    
    for token_pos, num_tokens, matched_word, actual_text in token_positions:
        try:
            # Extract hidden states
            if num_tokens == 1:
                word_repr = hidden_states[:, token_pos, :]  # [num_layers, hidden_dim]
            else:
                word_repr = hidden_states[:, token_pos:token_pos + num_tokens, :].mean(dim=1)  # [num_layers, hidden_dim]
            
            article_representations.append(word_repr)
            valid_positions.append((token_pos, num_tokens, matched_word, actual_text))
            
        except Exception as e:
            # Skip if hidden state extraction fails
            continue
    
    if not article_representations:
        return {}
    
    result = {}
    
    if multi_occurrence_handling == "individual":
        # Each occurrence is a separate training example with same target score
        for i, (word_repr, (token_pos, num_tokens, matched_word, actual_text)) in enumerate(zip(article_representations, valid_positions)):
            occurrence_key = f"{word}_occ{i}" if len(article_representations) > 1 else word
            
            result[occurrence_key] = {
                'hidden_states': word_repr.cpu().half(),
                'score': score,
                'position': token_pos,
                'num_tokens': num_tokens,
                'occurrence_index': i,
                'total_occurrences': len(article_representations),
                'word_original': word,
                'multi_occurrence_strategy': 'individual',
                'token_validation': {
                    'actual_text': actual_text,
                    'matched_word': matched_word,
                    'target_word': word.lower(),
                    'validation_passed': True,
                    'extraction_method': 'enhanced_fuzzy_matching',
                    'matching_strategy': 'multi_strategy_enhanced'
                }
            }
    
    elif multi_occurrence_handling == "aggregate_avg":
        # Average all occurrences, single training example per word
        final_representation = torch.stack(article_representations).mean(dim=0)
        first_position = valid_positions[0]  # Use first position as reference
        
        result[word] = {
            'hidden_states': final_representation.cpu().half(),
            'score': score,
            'position': first_position[0],
            'num_tokens': first_position[1],
            'num_occurrences': len(token_positions),
            'aggregation_strategy': 'average',
            'word_original': word,
            'multi_occurrence_strategy': 'aggregate_avg',
            'token_validation': {
                'actual_text': first_position[3],
                'matched_word': first_position[2],
                'target_word': word.lower(),
                'validation_passed': True,
                'extraction_method': 'enhanced_fuzzy_matching',
                'matching_strategy': 'multi_strategy_enhanced'
            }
        }
    
    elif multi_occurrence_handling == "aggregate_concat":
        # Concatenate all occurrences along hidden dimension
        final_representation = torch.cat(article_representations, dim=-1)  # [num_layers, hidden_dim * num_occurrences]
        first_position = valid_positions[0]  # Use first position as reference
        
        result[word] = {
            'hidden_states': final_representation.cpu().half(),
            'score': score,
            'position': first_position[0],
            'num_tokens': first_position[1],
            'num_occurrences': len(token_positions),
            'aggregation_strategy': 'concatenate',
            'word_original': word,
            'concatenated_dim': final_representation.shape[-1],
            'multi_occurrence_strategy': 'aggregate_concat',
            'token_validation': {
                'actual_text': first_position[3],
                'matched_word': first_position[2],
                'target_word': word.lower(),
                'validation_passed': True,
                'extraction_method': 'enhanced_fuzzy_matching',
                'matching_strategy': 'multi_strategy_enhanced'
            }
        }
    
    return result

def extract_additional_article_words(tokenizer, article, article_tokens, word_importance_dict, max_additional_words=50):
    """
    Extract meaningful words from the article that are not in word_importance dictionary.
    These words will be assigned a score of 0 for training negative examples.
    
    MINIMAL FILTERING APPROACH:
    ✅ KEEPS: Almost everything - stop words, numbers, abbreviations, proper nouns, technical terms
    ❌ FILTERS: Only empty strings, pure punctuation, words already in annotations
    
    RATIONALE: Stop words and common words can be valuable negative examples for probe training.
    The probe should learn to distinguish truly important words from common/unimportant ones.
    
    Args:
        tokenizer: HuggingFace tokenizer
        article: Article text
        article_tokens: Pre-tokenized article tokens (not used but kept for compatibility)
        word_importance_dict: Dictionary/list of words that already have importance scores
        max_additional_words: Maximum number of additional words to extract (None = extract all)
    
    Returns:
        list: List of (word, 0.0) tuples for words present in article but not in word_importance
    """
    import re
    import string
    
    # Create set of words that already have importance scores
    existing_words = set()
    if isinstance(word_importance_dict, dict):
        existing_words = set(word.lower() for word in word_importance_dict.keys())
    elif isinstance(word_importance_dict, list) and len(word_importance_dict) > 0:
        first_entry = word_importance_dict[0]
        if isinstance(first_entry, (list, tuple)) and len(first_entry) == 2:
            existing_words = set(word.lower() for word, score in word_importance_dict)
        elif isinstance(first_entry, dict):
            existing_words = set(entry["word"].lower() for entry in word_importance_dict)
    
    # Extract meaningful words from article
    # Split into words while preserving important punctuation patterns
    words_in_article = article.split()
    
    # MINIMAL FILTERING: Only remove truly meaningless entries
    additional_words = []
    word_candidates = set()
    
    for word in words_in_article:
        # Clean word but preserve important characters
        word_clean = re.sub(r'^[^\w]+|[^\w]+$', '', word)  # Remove leading/trailing punctuation only
        word_lower = word_clean.lower()
        
        # MINIMAL FILTERING: Keep almost everything including stop words, numbers, abbreviations
        if (word_lower not in existing_words and 
            len(word_lower) >= 1 and  # Keep even single characters (A, I, etc.)
            word_lower.strip() and   # Skip empty strings
            not re.match(r'^[^\w]*$', word_clean)):  # Skip pure punctuation/whitespace
            
            word_candidates.add(word_lower)
    
    # Convert to list and limit with randomization to avoid bias
    word_list = list(word_candidates)
    if max_additional_words is not None and len(word_list) > max_additional_words:
        # Randomize selection to avoid position bias
        random.shuffle(word_list)
        word_list = word_list[:max_additional_words]
    
    additional_words = [(word, 0.0) for word in word_list]
    
    return additional_words

def extract_targeted_hidden_states(test_mode=False, max_samples=None, multi_occurrence_handling="aggregate_avg", sort_by_length=False, include_zero_score_words=True, max_zero_score_words=30):
    """
    Extract hidden states for training word importance probes with balanced positive/negative examples.
    
    Words are extracted from the article portion of the input sequence.
    Success rates:
    - Zero-score words: ~88% success rate (excellent negative examples)
    - Annotated words: ~29% success rate (positive examples)
    
    Args:
        test_mode: If True, process limited samples for testing
        max_samples: Maximum number of samples to process
        multi_occurrence_handling: How to handle word repetitions
            - "aggregate_avg": Average occurrences (RECOMMENDED - robust)
            - "aggregate_concat": Concatenate occurrences (preserves all info)
            - "individual": Each occurrence as separate example (more data)
        sort_by_length: If True, process shortest articles first
        include_zero_score_words: If True, add zero-score words as negative examples (RECOMMENDED)
        max_zero_score_words: Maximum zero-score words per article (30 recommended, None = extract all)
    
    Enhanced Key Features:
        ✅ Multi-strategy fuzzy word matching (finds: pacquiao, Pacquiao, pacquiaos, Pacquiao's, anti-american, antiamerican)
        ✅ Enhanced tokenization handling (compound words, hyphenation, punctuation variants)
        ✅ Multi-token sequence matching (up to 4 tokens for complex words)
        ✅ Character-level fallback for important words (score >= 0.3)
        ✅ Multi-occurrence = multiple word instances (not subword tokens)
        ✅ Balanced training data (positive + negative examples)
        ✅ Memory efficient (~40-200x less storage than full hidden states)
        ✅ Consistent handling for annotated and zero-score words
    
    Output Structure:
        - h1 (word representations): Hidden states for target words
        - h2 (context representations): Article-end context hidden states
        - Metadata: positions, scores, extraction details
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Multi-occurrence handling: {multi_occurrence_handling}")
    if include_zero_score_words:
        print(f"Include zero-score words: {include_zero_score_words} (max: {max_zero_score_words})")
    if sort_by_length:
        print(f"Sort by length: Processing shortest articles first")
    
    # Set random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    print(f"Set random seed: {random_seed}")
    
    if test_mode:
        print("=== TEST MODE: Processing limited samples ===")
        max_samples = max_samples or 10
    
    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0
    
    # Randomize weights if requested (control experiment)
    if args.randomize_weights:
        model = randomize_model_weights(model, seed=42)
    
    # Get dataset-specific field names
    dataset_config = DATASET_CONFIGS.get(DATASET_NAME, DATASET_CONFIGS["cnn_dailymail"])
    text_field = dataset_config["text_field"]
    summary_field = dataset_config["summary_field"]
    id_field = dataset_config["id_field"]

    # Data paths
    data_file = f"../data/{MODEL_NAME}/{DATASET_NAME}/generated_summaries_with_word_importance_deduplicated.json"

    # Create descriptive save directory based on extraction settings
    mode_suffix = "article"
    if multi_occurrence_handling != "aggregate_avg":
        mode_suffix += f"_{multi_occurrence_handling}"
    if include_zero_score_words:
        mode_suffix += "_with_zeros"
    
    # Add randomized suffix if using randomized weights
    if args.randomize_weights:
        mode_suffix += "_RANDOMIZED"

    save_dir = f"saved_features/hidden_states/{MODEL_NAME}/{DATASET_NAME}/{mode_suffix}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Save directory: {save_dir}")

    # Load data
    print(f"Loading data from: {data_file}")
    with open(data_file, "r") as f:
        data = json.load(f)

    # Filter only articles/dialogues with word_importance data
    filtered_data = []
    for i, item in enumerate(data):
        if "word_importance" in item and item["word_importance"]:
            # Debug: check the format of word_importance for the first item
            if i == 0:
                print(f"Debug: word_importance format for first item:")
                print(f"Type: {type(item['word_importance'])}")
                if isinstance(item['word_importance'], dict):
                    sample_items = list(item['word_importance'].items())[:3]
                    print(f"First few entries: {sample_items}")
                else:
                    print(f"First few entries: {item['word_importance'][:3] if len(item['word_importance']) > 0 else 'empty'}")
            
            # Check if word_importance has valid scores
            # Handle different possible formats
            try:
                if isinstance(item["word_importance"], dict):
                    # Format: {"word": score, "word2": score2, ...}
                    valid_scores = [score for word, score in item["word_importance"].items() if score >= 0.1]
                elif isinstance(item["word_importance"], list) and len(item["word_importance"]) > 0:
                    first_entry = item["word_importance"][0]
                    
                    if isinstance(first_entry, (list, tuple)) and len(first_entry) == 2:
                        # Format: [(word, score), (word, score), ...]
                        valid_scores = [score for word, score in item["word_importance"] if score >= 0.1]
                    elif isinstance(first_entry, dict):
                        # Format: [{"word": word, "score": score}, ...]
                        valid_scores = [entry["score"] for entry in item["word_importance"] if entry.get("score", 0) >= 0.1]
                    else:
                        # Unknown format, skip
                        print(f"Warning: Unknown word_importance format in item {item.get(id_field, i)}: {type(first_entry)}")
                        continue
                else:
                    continue
                    
                if valid_scores:
                    filtered_data.append(item)
                    
            except Exception as e:
                print(f"Error processing word_importance for item {item.get(id_field, i)}: {e}")
                continue

    print(f"Found {len(filtered_data)} items with valid word_importance data out of {len(data)} total")

    # Sort by text length if requested
    if sort_by_length:
        print("Sorting items by length (shortest first)...")
        filtered_data = sorted(filtered_data, key=lambda x: len(x[text_field]))
        print(f"Shortest item: {len(filtered_data[0][text_field])} characters")
        print(f"Longest item: {len(filtered_data[-1][text_field])} characters")
        if len(filtered_data) >= 10:
            print(f"Median item: {len(filtered_data[len(filtered_data)//2][text_field])} characters")

    # Limit samples if max_samples is provided (regardless of test_mode)
    if max_samples:
        original_count = len(filtered_data)
        filtered_data = filtered_data[:max_samples]
        if sort_by_length:
            print(f"Limited to {len(filtered_data)} SHORTEST samples (from {original_count} total)")
        else:
            print(f"Limited to {len(filtered_data)} samples (from {original_count} total)")
        
        # Show detailed info about selected items
    # Extract indices that already exist
    existing_files = set()
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            if filename.startswith("article_") and filename.endswith(".pt"):
                idx = filename.replace("article_", "").replace(".pt", "")
                existing_files.add(idx)
    
    print(f"Found {len(existing_files)} existing hidden state files")
    
    # Process each article
    processed_count = 0
    skipped_count = 0
    total_words_found = 0
    total_words_searched = 0
    total_zero_score_words_found = 0
    total_zero_score_words_searched = 0
    
    for item in tqdm(filtered_data, desc="Extracting targeted hidden states"):
        idx = item[id_field]
        
        # Skip if already processed
        if str(idx) in existing_files:
            skipped_count += 1
            continue
        
        article = item[text_field]
        summary = item[summary_field]
        word_importance = item["word_importance"]
        
        # Create list of annotated words with their scores
        target_words = []
        try:
            if isinstance(word_importance, dict):
                # Format: {"word": score, "word2": score2, ...}
                target_words = [(word, score) for word, score in word_importance.items()]
            elif isinstance(word_importance, list) and len(word_importance) > 0:
                first_entry = word_importance[0]
                
                if isinstance(first_entry, (list, tuple)) and len(first_entry) == 2:
                    # Format: [(word, score), (word, score), ...]
                    target_words = [(word, score) for word, score in word_importance]
                elif isinstance(first_entry, dict):
                    # Format: [{"word": word, "score": score}, ...]
                    target_words = [(entry["word"], entry["score"]) for entry in word_importance]
        except Exception as e:
            print(f"Error processing word_importance for article {idx}: {e}")
            continue
        
        # ADD ZERO-SCORE WORDS: Extract additional words from article with score=0.0
        if include_zero_score_words:
            additional_words = extract_additional_article_words(
                tokenizer, article, None, word_importance, max_zero_score_words
            )
            if additional_words:
                target_words.extend(additional_words)
        
        # Prepare the FULL TEXT that was actually processed
        # This should be: prompt + summary (the complete sequence that was processed)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following news article:\n\n{article}"},
            {"role": "assistant", "content": summary}  # Add the actual summary
        ]
        
        # Apply chat template to get the complete sequence
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Tokenize the complete sequence (increased from 2048 to 4096)
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Track sequence length statistics
        seq_len = inputs['input_ids'].shape[1]
        if seq_len >= 4096:
            print(f"Warning: Article {idx} truncated at max_length (4096 tokens)")
        
        # Tokenize parts separately to find positions
        prompt_only = tokenizer.apply_chat_template(messages[:-1], tokenize=False)  # Without assistant response
        
        try:
            # Extract hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get all layer hidden states: [num_layers, batch_size, seq_len, hidden_dim]
                hidden_states = torch.stack(outputs.hidden_states)  # Shape: [num_layers, 1, seq_len, hidden_dim]
                hidden_states = hidden_states.squeeze(1)  # Remove batch dimension: [num_layers, seq_len, hidden_dim]
                
                seq_len = hidden_states.shape[1]
                
                # Extract targeted hidden states for each word
                word_hidden_states = {}
                
                article_tokens = tokenizer.encode(article, add_special_tokens=False)
                
                for word, score in target_words:
                    total_words_searched += 1
                    
                    # Track if this is a zero-score word
                    is_zero_score_word = (score == 0.0)
                    if is_zero_score_word:
                        total_zero_score_words_searched += 1
                    
                    word_found = False
                    
                    # Extract from article with multi-occurrence handling
                    article_results = process_article_word_occurrences(
                        tokenizer, word, score, article, article_tokens, hidden_states,
                        inputs['input_ids'], seq_len, multi_occurrence_handling
                    )
                    if article_results:
                        word_hidden_states.update(article_results)
                        word_found = True
                    
                    # Debug: For first article, show details about word finding
                    if word_found:
                        total_words_found += 1
                        if is_zero_score_word:
                            total_zero_score_words_found += 1
                
                # Find the EXACT end of the article content (more precise than estimation)
                article_end_pos = find_article_end_position(tokenizer, prompt_only, article)
                context_hidden_states = hidden_states[:, article_end_pos, :].cpu().half()  # [num_layers, hidden_dim]
                
                # Only save if we found at least one target word
                if word_hidden_states:
                    # Save targeted data
                    save_data = {
                        'article_id': idx,
                        'word_hidden_states': word_hidden_states,  # Dict: word -> {'hidden_states': tensor, 'score': float}
                        'context_hidden_states': context_hidden_states,  # [num_layers, hidden_dim]
                        'article_end_pos': article_end_pos,
                        'seq_len': seq_len,
                        'num_layers': hidden_states.shape[0],
                        'hidden_dim': hidden_states.shape[2],
                        'num_target_words': len(word_hidden_states)
                    }
                    
                    save_path = os.path.join(save_dir, f"article_{idx}.pt")
                    torch.save(save_data, save_path)
                    processed_count += 1
                
                # Clear memory
                del hidden_states, outputs
                if 'save_data' in locals():
                    del save_data
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            continue
    
    print(f"\n📊 TARGETED EXTRACTION COMPLETE!")
    print(f"📈 Processing Summary:")
    print(f"  • Processed: {processed_count} new articles")
    print(f"  • Skipped: {skipped_count} existing articles")
    print(f"  • Total candidate articles: {len(filtered_data)}")
    
    if test_mode:
        print(f"  • Test mode: Processing was limited to {len(filtered_data)} shortest articles")
        if processed_count == 0 and skipped_count > 0:
            print(f"  ⚠️  All selected articles were already processed (files exist)")
            print(f"     To force reprocessing, delete files in: {save_dir}")
    
    # Statistics breakdown
    annotated_words_searched = total_words_searched - total_zero_score_words_searched
    annotated_words_found = total_words_found - total_zero_score_words_found
    
    print(f"\n🎯 Word Extraction Statistics:")
    print(f"  • Total words found: {total_words_found}/{total_words_searched} ({100*total_words_found/max(1,total_words_searched):.1f}%)")
    if include_zero_score_words and total_zero_score_words_searched > 0:
        print(f"    - Annotated words: {annotated_words_found}/{annotated_words_searched} ({100*annotated_words_found/max(1,annotated_words_searched):.1f}%)")
        print(f"    - Zero-score words: {total_zero_score_words_found}/{total_zero_score_words_searched} ({100*total_zero_score_words_found/max(1,total_zero_score_words_searched):.1f}%)")
    
    print(f"  ✅ Improved extraction: Direct sequence search (no position mapping issues)")
    
    print(f"\n💾 Output:")
    print(f"  • Save directory: {save_dir}")
    
    # Show what files exist
    if os.path.exists(save_dir):
        existing_file_count = len([f for f in os.listdir(save_dir) if f.startswith("article_") and f.endswith(".pt")])
        print(f"  • Total files in directory: {existing_file_count}")
    
    # Dynamic note based on extraction mode
    if multi_occurrence_handling == "individual":
        note = f"  • Content: Each word occurrence from article as separate training example + context hidden states"
    else:
        note = f"  • Content: Aggregated word hidden states from article ({multi_occurrence_handling}) + context hidden states"
    if include_zero_score_words:
        note += f"\n    + Zero-score words for balanced negative examples (max {max_zero_score_words} per article)"
    print(note)
    
    print(f"  • Memory efficiency: ~40-200x less storage than full hidden states")
    

if __name__ == "__main__":
    # RECOMMENDED: Article mode with zero-score words for balanced training data
    # This approach has shown much better success rates:
    # - Annotated words: ~29% success rate
    # - Zero-score words: ~88% success rate
    # The zero-score words provide excellent negative examples for probe training
    
    if MAX_ZERO_SCORE_WORDS is None:
        print("🔥 EXTRACTING ALL ZERO-SCORE WORDS (no limit) - Best for unbiased training")
    else:
        print(f"📏 Limiting zero-score words to {MAX_ZERO_SCORE_WORDS} per article")
    
    if MAX_SAMPLES:
        print(f"🔢 Limiting to {MAX_SAMPLES} samples for {DATASET_NAME}")
    else:
        print(f"🔢 Processing ALL samples for {DATASET_NAME}")
        
    print("="*60)
    
    # Production-ready extraction
    extract_targeted_hidden_states(
        test_mode=False,                        # Set to False for full dataset
        max_samples=MAX_SAMPLES,
        multi_occurrence_handling="aggregate_avg",  # Recommended strategy
        include_zero_score_words=True,         # ESSENTIAL for balanced data
        max_zero_score_words=MAX_ZERO_SCORE_WORDS,  # From command line argument
        sort_by_length=False                  # Random sampling for better representation
    )
    
    print("\n📁 OUTPUT DIRECTORIES:")
    print("  • With zeros: .../article_with_zeros/")
