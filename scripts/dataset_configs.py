"""
Dataset configuration mapping for different summarization datasets.
This allows the main data generation script to handle multiple datasets uniformly.
"""

SAMSUM_CONFIG = {
    "text_field": "dialogue",
    "summary_field": "summary",
    "id_field": "id",
    "prompt_placeholder": "dialogue",
    "prompt_prefix": "Summarize the following dialogue",
    "length_constraint": "in exactly {word_count} words",
    "system_message": "You are a helpful assistant that produces dialogue summaries. Provide ONLY the summary text.",
    "description": "Conversation/dialogue summarization",
    "generated_field": "generated_summary"
}

DATASET_CONFIGS = {
    "cnn_dailymail": {
        "text_field": "article",
        "summary_field": "highlights", 
        "id_field": "id",
        "prompt_placeholder": "text",
        "prompt_prefix": "Summarize the following text",
        "length_constraint": "in exactly {word_count} words",
        "system_message": "You are a helpful assistant that produces article summaries. Provide ONLY the summary text.",
        "description": "News article summarization",
        "generated_field": "summary"
    },
    "knkarthick/samsum": SAMSUM_CONFIG,
    "samsum": SAMSUM_CONFIG.copy(),  # Alias for knkarthick/samsum to support local folder naming
    "decoda": {
        "text_field": "dialogue",
        "summary_field": "synopsis",
        "id_field": "id",
        "prompt_placeholder": "dialogue",
        "prompt_prefix": "Résumez le dialogue suivant",
        "length_constraint": "en exactement {word_count} mots",
        "system_message": "Vous êtes un assistant utile qui produit des résumés de dialogue. Fournissez UNIQUEMENT le texte du résumé.",
        "description": "Conversation/dialogue summarization (French)",
        "generated_field": "generated_summary"
    }
}

def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    config = DATASET_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_CONFIGS.keys())}")
    return config

def create_prompt_template(dataset_name: str, word_count: int) -> str:
    """Create a prompt template for a specific dataset and word count."""
    config = get_dataset_config(dataset_name)
    length_constraint = config.get("length_constraint", "in exactly {word_count} words")
    return f"{config['prompt_prefix']} {length_constraint.format(word_count=word_count)}: {{{config['prompt_placeholder']}}}"

def get_system_message(dataset_name: str) -> str:
    """Get the system message for a specific dataset."""
    config = get_dataset_config(dataset_name)
    return config["system_message"]
