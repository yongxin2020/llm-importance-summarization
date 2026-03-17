from typing import Optional, Union, List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

class UnifiedLLMGenerator:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.default_model = default_model
        self._current_model = None
        self._initialized = False
        self._model = None
        self._tokenizer = None
        self._client = None
        self._init_lock = threading.Lock()  # Thread safety for model initialization

    def initialize_model(self, model_name: Optional[str] = None) -> None:
        """Initialize the model and necessary components based on provider."""
        model_name = model_name or self.default_model
        
        # Use lock to prevent concurrent initialization
        with self._init_lock:
            if self._initialized and model_name == self._current_model:
                return
                
            print(f"Initializing model: {model_name}")  # Debug output

            provider = self._identify_provider(model_name)

            try:
                if provider == "openai":
                    self._initialize_openai(model_name)
                elif provider == "deepseek":
                    self._initialize_deepseek(model_name)
                elif provider == "huggingface":
                    self._initialize_huggingface(model_name)
                else:
                    raise ValueError(f"Unsupported model provider for: {model_name}")

                self._current_model = model_name
                self._initialized = True
                print(f"✅ Successfully initialized {model_name}")

            except Exception as e:
                self._initialized = False
                self._current_model = None
                print(f"❌ Model initialization failed for {model_name}: {str(e)}")
                raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _initialize_openai(self, model_name: str) -> None:
        """Initialize OpenAI client for API v1.0+ compatibility.
        
        Note: OpenAI models were not used in the paper. This method is included
        for code completeness and future extensibility.
        
        Args:
            model_name: Model name to use with OpenAI
            
        Raises:
            RuntimeError: If OpenAI package is not installed
        """
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI package not installed. Install with: pip install openai")
        
        try:
            # Use OpenAI client initialization for v1.0+ compatibility
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            print(f"✅ OpenAI client initialized for {model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI client: {e}")
            raise

    def _initialize_deepseek(self, model_name: str) -> None:
        """Initialize Deepseek client."""
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI package not installed. Install with: pip install openai")
        
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=30.0
            )
            print(f"✅ DeepSeek client initialized for {model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize DeepSeek client: {e}")
            raise

    def _initialize_huggingface(self, model_name: str) -> None:
        """Initialize HuggingFace model and tokenizer."""
        try:
            print(f"Loading HuggingFace model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loading model weights...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"  # Automatically handle GPU placement
            )
            
            # Set pad token if not available
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"✅ HuggingFace model {model_name} loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load HuggingFace model {model_name}: {str(e)}")
            # Clean up any partially loaded components
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _generate_hf_response(
            self,
            messages: List[Dict[str, str]],
            max_tokens: int,
            temperature: float,
            **kwargs
    ) -> str:
        """Unified generation for HuggingFace models."""
        try:
            # Prepare input
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self._model.device)

            attention_mask = input_ids.ne(self._tokenizer.pad_token_id)

            # Generate response
            with torch.no_grad():  # Save memory
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=temperature,
                    eos_token_id=[self._tokenizer.eos_token_id],
                    pad_token_id=self._tokenizer.pad_token_id,
                    **kwargs
                )

            # Decode response
            response = self._tokenizer.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Clean up GPU memory
            del input_ids, attention_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return response.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA out of memory during generation: {e}")
            # Emergency cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"CUDA out of memory: {e}. Try reducing batch_size or max_tokens.")
            
        except Exception as e:
            print(f"❌ HuggingFace generation failed: {e}")
            # Cleanup on any error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _identify_provider(self, model_name: str) -> str:
        """Identify the provider based on model name patterns."""
        if model_name.startswith("gpt-"):
            return "openai"
        elif model_name.startswith("deepseek"):
            return "deepseek"
        elif "/" in model_name:  # HuggingFace format
            return "huggingface"
        return "openai"  # Default fallback

    def cleanup(self) -> None:
        """Clean up resources and free GPU memory."""
        try:
            if self._model is not None:
                del self._model
            if self._tokenizer is not None:
                del self._tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._model = None
            self._tokenizer = None
            self._client = None
            self._initialized = False
            self._current_model = None
            
            print("🧹 Resources cleaned up successfully")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
