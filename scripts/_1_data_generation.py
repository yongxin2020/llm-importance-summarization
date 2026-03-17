from datasets import load_dataset
import openai
import json
import time
import argparse
from typing import Tuple, Optional, List, Dict
from model_utils import UnifiedLLMGenerator
from dataset_configs import get_dataset_config, create_prompt_template, get_system_message
import os
from pathlib import Path
import logging
from datetime import datetime
import asyncio
import concurrent.futures
from threading import Lock

class SummaryGenerator:
    def __init__(self, api_key: str, default_model: str = "deepseek-reasoner", dataset_name: str = "cnn_dailymail"):
        self.llm_generator = UnifiedLLMGenerator(api_key=api_key, default_model=default_model)
        self.save_lock = Lock()  # Thread-safe saving
        self.dataset_config = get_dataset_config(dataset_name)
        self.dataset_name = dataset_name
        
    def generate_summary(
        self,
        text: str,
        prompt_template: str,
        model_name: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Tuple[str, int, int, int]:
        """
        Generate a summary using the specified model with unified error handling.
        
        Args:
            text: The input text to summarize
            prompt_template: Template for the prompt (should contain {text} placeholder)
            model_name: Optional model name override
            max_tokens: Maximum tokens for the response
            temperature: Creativity parameter
            
        Returns:
            Tuple containing:
            - summary text (or error message)
            - input tokens
            - output tokens
            - total tokens
            
        Scenario Behavior:
            Successful generation: Returns (summary, input_tokens, output_tokens, total_tokens)
            Content restriction: Returns ("[Content restricted]", input_token_estimate, 0, input_token_estimate)
            Rate limit: Retries up to 3 times before failing
            Other errors: Returns ("[Error: ...]", estimated_tokens, 0, estimated_tokens)
        """
        estimated_tokens = len(text.split())
        messages = [
            {"role": "system", "content": get_system_message(self.dataset_name)},
            {"role": "user", "content": prompt_template.format(**{self.dataset_config["prompt_placeholder"]: text})}
        ]
        
        try:
            # Initialize the model
            self.llm_generator.initialize_model(model_name)
            
            # Get the provider type for special handling
            provider = self.llm_generator._identify_provider(model_name or self.llm_generator.default_model)
            
            if provider in ["openai", "deepseek"]:
                # For API-based models, we can get token usage
                response = self.llm_generator._client.chat.completions.create(
                    model=model_name or self.llm_generator.default_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                summary = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
            elif provider == "huggingface":
                # For HF models, we need to estimate token usage
                input_ids = self.llm_generator._tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt"
                )
                input_tokens = input_ids.shape[-1]
                
                summary = self.llm_generator._generate_hf_response(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Estimate output tokens
                output_tokens = len(self.llm_generator._tokenizer.encode(summary))
                total_tokens = input_tokens + output_tokens
                
            return summary, input_tokens, output_tokens, total_tokens
            
        except Exception as e:
            if "Content Exists Risk" in str(e) or "content policy" in str(e).lower():
                print(f"Skipping restricted content (input tokens: ~{estimated_tokens})")
                return (
                    "[Content restricted]",
                    estimated_tokens,  # input tokens (estimated)
                    0,  # output tokens
                    estimated_tokens  # total tokens (input only)
                )
            elif "rate limit" in str(e).lower():
                print("Rate limit hit, retrying...")
                raise  # Will trigger retry
            else:
                print(f"❌ Unexpected error: {str(e)}")
                return (
                    f"[Error: {str(e)}]",
                    estimated_tokens,  # input tokens (estimated)
                    0,  # output tokens
                    estimated_tokens  # total tokens (input only)
                )

    def generate_batch_summaries(
        self,
        batch_requests: List[Dict],
        model_name: Optional[str] = None,
        max_workers: Optional[int] = None
    ) -> List[Tuple[str, int, int, int]]:
        """
        Generate multiple summaries in batch using ThreadPoolExecutor.
        
        Args:
            batch_requests: List of dicts with keys 'text', 'prompt_template'
            model_name: Optional model name override
            max_workers: Number of concurrent threads (uses instance default if None)
            
        Returns:
            List of tuples (summary, input_tokens, output_tokens, total_tokens)
        """
        if max_workers is None:
            max_workers = getattr(self, 'max_workers', 3)
        
        # For local models (Hugging Face), disable concurrency to avoid conflicts
        if model_name and any(provider in model_name.lower() for provider in ['meta-llama', 'llama', 'mistral', 'qwen']):
            print(f"Detected local model {model_name}, processing sequentially to avoid conflicts")
            max_workers = 1
            
        def process_single_request(request_with_index):
            index, request = request_with_index
            try:
                # Use thread-safe generation with lock
                with self.save_lock:
                    result = self.generate_summary(
                        text=request['text'],
                        prompt_template=request['prompt_template'],
                        model_name=model_name
                    )
                return index, result
            except Exception as e:
                # Handle individual request failures
                estimated_tokens = 100  # fallback estimate
                error_result = (f"[Batch Error: {str(e)}]", estimated_tokens, 0, estimated_tokens)
                return index, error_result
        
        # Add indices to preserve order
        indexed_requests = [(i, req) for i, req in enumerate(batch_requests)]
        
        if max_workers == 1:
            # Sequential processing for local models
            indexed_results = []
            for req in indexed_requests:
                result = process_single_request(req)
                indexed_results.append(result)
        else:
            # Concurrent processing for API models
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all requests with indices
                futures = [executor.submit(process_single_request, req) for req in indexed_requests]
                
                # Collect results and sort by original index
                indexed_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        index, result = future.result()
                        indexed_results.append((index, result))
                    except Exception as e:
                        # This shouldn't happen since we handle errors in process_single_request
                        print(f"Unexpected executor error: {e}")
        
        # Sort by index and extract results
        indexed_results.sort(key=lambda x: x[0])
        results = [result for index, result in indexed_results]
        
        return results

    def process_item_batch(
        self,
        items_batch: List[Dict],
        prompts: Dict,
        model_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a batch of data items (articles/dialogues) with all their prompts.
        
        Args:
            items_batch: List of data item dicts
            prompts: Dictionary of prompt categories and templates
            model_name: Optional model name override
            
        Returns:
            List of summary entries for all items and prompts
        """
        all_requests = []
        request_metadata = []
        
        # Get dataset-specific field names
        text_field = self.dataset_config["text_field"]
        summary_field = self.dataset_config["summary_field"]
        id_field = self.dataset_config["id_field"]
        
        # Prepare all requests
        for item in items_batch:
            item_text = item[text_field]
            item_id = item[id_field]
            reference_summary = item[summary_field]
            
            for category, subcategories in prompts.items():
                for subcategory, prompt_template in subcategories.items():
                    current_prompt = f"{category}_{subcategory}"
                    
                    all_requests.append({
                        'text': item_text,
                        'prompt_template': prompt_template
                    })
                    
                    request_metadata.append({
                        'item_id': item_id,
                        'item_text': item_text,
                        'reference_summary': reference_summary,
                        'criteria': current_prompt
                    })
        
        # Process all requests in batch
        batch_results = self.generate_batch_summaries(all_requests, model_name)
        
        # Verify we got the expected number of results
        if len(batch_results) != len(request_metadata):
            print(f"WARNING: Expected {len(request_metadata)} results, got {len(batch_results)}")
        
        # Combine results with metadata
        summary_entries = []
        text_field = self.dataset_config["text_field"]
        summary_field = self.dataset_config["summary_field"]
        generated_field = self.dataset_config.get("generated_field", "generated_summary")
        
        for i, ((summary, input_tokens, output_tokens, total_tokens), metadata) in enumerate(zip(batch_results, request_metadata)):
            summary_entry = {
                text_field: metadata['item_text'],  # "article" or "dialogue"
                summary_field: metadata['reference_summary'],  # "highlights" or "summary"
                "id": metadata['item_id'],
                generated_field: summary,
                "criteria": metadata['criteria'],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            summary_entries.append(summary_entry)
        
        return summary_entries


def save_predictions_checkpoint(summaries: List[Dict], save_fp: str, backup: bool = True) -> None:
    """Save predictions to file with backup option."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    
    # Create backup if file exists and backup is requested
    if backup and os.path.exists(save_fp):
        backup_fp = save_fp.replace('.json', f'_backup_{int(time.time())}.json')
        os.rename(save_fp, backup_fp)
        print(f"Created backup: {backup_fp}")
    
    # Save current results
    with open(save_fp, "w") as f:
        json.dump(summaries, f, indent=4)
    print(f"Saved {len(summaries)} predictions to {save_fp}")


def load_existing_predictions(save_fp: str) -> List[Dict]:
    """Load existing predictions if file exists."""
    if os.path.exists(save_fp):
        try:
            with open(save_fp, "r") as f:
                existing = json.load(f)
            print(f"Loaded {len(existing)} existing predictions from {save_fp}")
            return existing
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Could not load existing predictions from {save_fp}, starting fresh")
            return []
    return []


def get_processed_article_ids(summaries: List[Dict], total_prompts: int = 10) -> set:
    """Get set of article IDs that have been fully processed (all prompts completed).
    
    Args:
        summaries: List of summary records
        total_prompts: Expected number of prompts per article (default: 10 for length_num 10-100)
        
    Returns:
        Set of article IDs with all prompts completed
    """
    article_counts = {}
    for summary in summaries:
        article_id = summary['id']
        article_counts[article_id] = article_counts.get(article_id, 0) + 1
    
    completed_articles = {aid for aid, count in article_counts.items() 
                         if count >= total_prompts}
    
    return completed_articles


def setup_logging(save_fp: str) -> logging.Logger:
    """Setup logging to both file and console."""
    log_fp = save_fp.replace('.json', '.log')
    os.makedirs(os.path.dirname(log_fp), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_fp),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct', help="gpt-3.5-turbo, gpt-4, deepseek-reasoner")
    parser.add_argument('--save_fp', default='../data/meta-llama/Llama-3.2-1B-Instruct/cnn_dailymail/predictions.json', help="place to save the prediction file")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="../config.json", help="Path to config.json file with API keys")
    parser.add_argument("--checkpoint_freq", type=int, default=50, help="Save checkpoint every N articles")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions.json")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of articles to process in each batch")
    parser.add_argument("--max_workers", type=int, default=3, help="Number of concurrent API requests")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing/subset runs)")
    
    args = parser.parse_args()
    model = args.model
    save_fp = args.save_fp
    checkpoint_freq = args.checkpoint_freq
    resume = args.resume
    batch_size = args.batch_size
    max_workers = args.max_workers
    max_samples = args.max_samples
    config_path = args.config_path

    # Setup logging
    logger = setup_logging(save_fp)
    logger.info(f"Starting data generation with model: {model}")
    logger.info(f"Save path: {save_fp}")
    logger.info(f"Checkpoint frequency: {checkpoint_freq} articles")
    logger.info(f"Batch processing: {batch_size} articles, {max_workers} concurrent workers")

    # Load dataset with or without config
    if args.dataset == "decoda":
        # Special handling for local decoda dataset
        possible_paths = ["decoda/test.json", "../decoda/test.json"]
        data_path = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if data_path:
            dataset = load_dataset("json", data_files={"test": data_path})
            # Add empty train/validation splits if missing to avoid errors
            if "train" not in dataset:
                from datasets import Dataset
                dataset["train"] = Dataset.from_dict({})
            if "validation" not in dataset:
                from datasets import Dataset
                dataset["validation"] = Dataset.from_dict({})
        else:
            raise FileNotFoundError(f"Could not find decoda/test.json in {possible_paths}")
            
    elif args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset)
        
    train_data = dataset["train"]
    logger.info(f"Number of samples in training data: {len(train_data)}")
    validation_data = dataset["validation"]
    logger.info(f"Number of samples in validation data: {len(validation_data)}")
    test_data = dataset["test"]
    logger.info(f"Number of samples in test data: {len(test_data)}")

    # Generate dataset-specific prompts dynamically (needed for get_processed_article_ids)
    word_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    prompts = {
        "length_num": {
            count: create_prompt_template(args.dataset, count) for count in word_counts
        }
    }

    # Load existing predictions if resuming
    summaries = []
    processed_articles = set()
    if resume:
        summaries = load_existing_predictions(save_fp)
        # Calculate total prompts from the prompt structure
        total_prompts_count = sum(len(subcats) for subcats in prompts.values())
        processed_articles = get_processed_article_ids(summaries, total_prompts_count)
        logger.info(f"Resuming: Found {len(processed_articles)} completed articles")

    # Load the API key from the config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    if "deepseek" in model:
        api_key = config.get("deepseek_api_key")
    else:
        api_key = config.get("HF_access_token")

    # Initialize generator with dataset-specific configuration
    generator = SummaryGenerator(api_key=api_key, dataset_name=args.dataset)
    # Update the generator to use specified max_workers
    generator.max_workers = max_workers

    # Apply sample limit if specified
    if max_samples is not None:
        # Use original order (no shuffle) to match previous DeepSeek run
        test_data = test_data.select(range(min(max_samples, len(test_data))))
        logger.info(f"Limited to first {len(test_data)} samples in original order (max_samples={max_samples})")
    else:
        # For production, keep original order for consistency
        logger.info(f"Processing full dataset in original order: {len(test_data)} samples")

    # Deduplicate items based on ID to handle datasets with multiple annotations per sample (e.g. decoda)
    seen_ids = set()
    unique_items = []
    for sample in test_data:
        if sample["id"] not in seen_ids:
            seen_ids.add(sample["id"])
            unique_items.append(sample)
    
    if len(unique_items) < len(test_data):
        logger.info(f"Deduplication: Reduced from {len(test_data)} raw samples to {len(unique_items)} unique items")

    total_articles = len(unique_items)
    total_prompts = sum(len(subcats) for subcats in prompts.values())
    articles_to_process = total_articles - len(processed_articles)
    # Get dataset-specific configuration
    dataset_config = get_dataset_config(args.dataset)
    dataset_description = dataset_config["description"]
    
    logger.info(f"Total unique items: {total_articles}")
    logger.info(f"Items to process: {articles_to_process}")
    logger.info(f"Prompts per item: {total_prompts}")
    logger.info(f"Dataset type: {dataset_description}")
    logger.info("="*60)

    start_time = time.time()
    items_processed = 0
    last_checkpoint_time = time.time()

    # Process items in batches
    items_to_process_list = [sample for sample in unique_items if sample["id"] not in processed_articles]
    
    for batch_start in range(0, len(items_to_process_list), batch_size):
        batch_end = min(batch_start + batch_size, len(items_to_process_list))
        current_batch = items_to_process_list[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: items {batch_start + 1}-{batch_end}")
        batch_start_time = time.time()
        
        # Process entire batch
        try:
            batch_summaries = generator.process_item_batch(
                current_batch, 
                prompts, 
                model
            )
            summaries.extend(batch_summaries)
            
            # Log batch results
            batch_time = time.time() - batch_start_time
            items_in_batch = len(current_batch)
            items_processed += items_in_batch
            
            # Calculate progress
            elapsed_time = time.time() - start_time
            avg_time_per_article = elapsed_time / items_processed if items_processed > 0 else 0
            estimated_remaining = avg_time_per_article * (articles_to_process - items_processed)
            
            # Log batch completion
            generated_field = dataset_config.get("generated_field", "generated_summary")
            completed_in_batch = len([s for s in batch_summaries if not s[generated_field].startswith("[")])
            restricted_in_batch = len([s for s in batch_summaries if "[Content restricted]" in s[generated_field]])
            failed_in_batch = len([s for s in batch_summaries if "[Generation Error" in s[generated_field] or "[Error:" in s[generated_field] or "[Batch Error:" in s[generated_field]])
            
            logger.info(f"Batch completed in {batch_time:.1f}s ({batch_time/items_in_batch:.1f}s per item)")
            logger.info(f"Batch results: {completed_in_batch} successful, {restricted_in_batch} restricted, {failed_in_batch} failed")
            
            # Log some failed examples for debugging
            if failed_in_batch > 0:
                failed_examples = [s[generated_field] for s in batch_summaries if s[generated_field].startswith("[")][:3]
                logger.warning(f"Failed examples: {failed_examples}")
            
            logger.info(f"Overall progress: {items_processed}/{articles_to_process} articles")
            logger.info(f"ETA: {estimated_remaining/3600:.1f} hours")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {str(e)}")
            # Fall back to individual processing for this batch
            logger.info("Falling back to individual article processing...")
            
            for sample in current_batch:
                article_id = sample[dataset_config["id_field"]]
                article_text = sample[dataset_config["text_field"]]
                highlights = sample[dataset_config["summary_field"]]
                
                logger.info(f"Processing article {article_id} individually...")
                article_summaries = []
                
                # Process each prompt for this article
                generated_field = dataset_config.get("generated_field", "generated_summary")
                for category, subcategories in prompts.items():
                    for subcategory, prompt_template in subcategories.items():
                        current_prompt = f"{category}_{subcategory}"
                        
                        try:
                            summary, input_tokens, output_tokens, total_tokens = generator.generate_summary(
                                text=article_text,
                                prompt_template=prompt_template,
                                model_name=model
                            )
                            
                            summary_entry = {
                                dataset_config["text_field"]: article_text,
                                dataset_config["summary_field"]: highlights,
                                dataset_config["id_field"]: article_id,
                                generated_field: summary,
                                "criteria": current_prompt,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens,
                            }
                            article_summaries.append(summary_entry)
                            
                        except Exception as inner_e:
                            logger.error(f"❌ Failed individual generation for {current_prompt}: {str(inner_e)}")
                            summary_entry = {
                                dataset_config["text_field"]: article_text,
                                dataset_config["summary_field"]: highlights,
                                dataset_config["id_field"]: article_id,
                                generated_field: f"[Generation Error: {str(inner_e)}]",
                                "criteria": current_prompt,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                            }
                            article_summaries.append(summary_entry)
                
                summaries.extend(article_summaries)
                items_processed += 1

        # Save checkpoint periodically
        if items_processed % checkpoint_freq == 0 or items_processed >= articles_to_process:
            checkpoint_time = time.time() - last_checkpoint_time
            logger.info(f"Saving checkpoint after {items_processed} articles ({checkpoint_time:.1f}s since last save)")
            save_predictions_checkpoint(summaries, save_fp, backup=False)
            last_checkpoint_time = time.time()

    # Final summary and save
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    
    generated_field = dataset_config.get("generated_field", "generated_summary")
    completed = len([s for s in summaries if not s[generated_field].startswith("[")])
    restricted = len([s for s in summaries if "[Content restricted]" in s[generated_field]])
    failed = len([s for s in summaries if "[Generation Error" in s[generated_field] or "[Error:" in s[generated_field]])

    logger.info(f"Summary Generation Complete!")
    logger.info(f"• Total time: {elapsed_time/3600:.2f} hours")
    logger.info(f"• Articles processed: {items_processed}")
    logger.info(f"• Successfully generated: {completed}/{items_processed*total_prompts}")
    logger.info(f"• Content restrictions: {restricted}")
    logger.info(f"• Generation failures: {failed}")

    # Final save with backup
    save_predictions_checkpoint(summaries, save_fp, backup=True)
    logger.info(f"Final results saved to {save_fp}")


if __name__ == "__main__":
    main()