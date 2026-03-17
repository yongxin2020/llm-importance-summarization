"""
Objective: calculate importance scores for words that appear in the generated summaries
1. Collects all unique words (not tokens) from all generated summaries (tokenization is model-specific)
2. Calculates how frequently each word appears across summaries (scores between 0 and 1)
"""

import json
import re
from collections import defaultdict
import sys
import os

# Ensure we can import dataset_configs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_configs import get_dataset_config


def clean_summary(summary_text):
    # Remove common prefix patterns
    patterns_to_remove = [
        r"^Here is the \d+-word summary:\s*",
        r"^\(\d+ words\)\s*",
        r"^\d+-word summary:\s*",
        r"^Summary \(\d+ words\):\s*"
    ]

    for pattern in patterns_to_remove:
        summary_text = re.sub(pattern, "", summary_text, flags=re.IGNORECASE)

    # Remove any leading/trailing whitespace or quotes
    summary_text = summary_text.strip().strip('"').strip("'")

    return summary_text


def calculate_word_importance(summaries, generated_field):
    """Calculate word importance scores based on word frequency across summaries.
    
    Args:
        summaries: List of summary records
        generated_field: Field name containing the generated summary text
        
    Returns:
        Dictionary of word -> count mappings
    """
    word_counts = defaultdict(int)
    for summary in summaries:
        if isinstance(summary, dict):
            text = summary[generated_field]
        else:
            text = str(summary)
        cleaned_text = clean_summary(text).lower()
        # Find all the words, "Python 3.9" → ['Python', '3.9'], I'm → ["I'm"], "state-of-the-art" → ['state-of-the-art']
        words = re.findall(r"\b[\w'-]+(?:\.\d+)?\b", cleaned_text)
        for word in set(words):  # Use set to count each word once per summary
            word_counts[word] += 1
    return word_counts


parser = argparse.ArgumentParser(description="Preprocess data for token/word importance calculation.")
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name (default: meta-llama/Llama-3.2-1B-Instruct)')
parser.add_argument('--dataset_name', type=str, default="cnn_dailymail", help='Dataset name (default: cnn_dailymail)')
args = parser.parse_args()

model_name = args.model_name
dataset_name = args.dataset_name

# Validate inputs
if not model_name or not dataset_name:
    parser.error("Model name and dataset name are required")

# Get configuration for the dataset
try:
    config = get_dataset_config(dataset_name)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Construct paths
data_dir = f"../data/{model_name}/{dataset_name}"
input_file = os.path.join(data_dir, "predictions.json")
output_file_with_importance = os.path.join(data_dir, "generated_summaries_with_word_importance.json")
output_file_deduplicated = os.path.join(data_dir, "generated_summaries_with_word_importance_deduplicated.json")

# Validate input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file not found at {input_file}")
    print(f"Please ensure the predictions.json file exists in {data_dir}")
    sys.exit(1)

# Any new dataset without this key will default to "generated_summary".
generated_field = config.get("generated_field", "generated_summary")

# Load the generated summaries from the JSON file
try:
    with open(input_file, "r") as f:
        summaries = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON from {input_file}: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: Failed to load file {input_file}: {e}")
    sys.exit(1)

# Validate summaries is not empty
if not summaries:
    print(f"Error: No summaries found in {input_file}")
    sys.exit(1)

# Validate all summaries have required fields
for i, summary in enumerate(summaries):
    if not isinstance(summary, dict):
        print(f"Error: Summary at index {i} is not a dictionary")
        sys.exit(1)
    if "id" not in summary:
        print(f"Error: Summary at index {i} missing 'id' field")
        sys.exit(1)
    if generated_field not in summary:
        print(f"Error: Summary at index {i} missing '{generated_field}' field")
        sys.exit(1)

# Group summaries by article ID
article_to_summaries = defaultdict(list)
for summary in summaries:
    article_to_summaries[summary["id"]].append(summary)

# Calculate word-level importance scores for each article's summaries
for article_id, article_summaries in article_to_summaries.items():
    word_counts = calculate_word_importance(article_summaries, generated_field)
    total_summaries = len(article_summaries)
    importance_scores = {word: count / total_summaries for word, count in word_counts.items()}

    for summary in article_summaries:
        summary["word_importance"] = importance_scores

with open(output_file_with_importance, "w") as f:
    json.dump(summaries, f, indent=4)

print(f"Saved summaries with word importance to {output_file_with_importance}")


samples = summaries
# Deduplicate by id - keep first occurrence of each id
# This is useful for datasets with multiple annotations per article,
# where we only need one representative sample
unique_samples = {}
for item in samples:
    if item["id"] not in unique_samples:
        unique_samples[item["id"]] = item
deduplicated_samples = list(unique_samples.values())

print(f"Original data size: {len(samples)}")
print(f"After deduplication: {len(deduplicated_samples)}")

with open(output_file_deduplicated, "w") as f:
    json.dump(deduplicated_samples, f, indent=4)

print(f"Saved deduplicated summaries to {output_file_deduplicated}")
print("\n✅ Preprocessing complete!")
