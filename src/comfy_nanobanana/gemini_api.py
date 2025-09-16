import os
import time
import asyncio
from typing import Optional, List, Union, Dict, Any, Callable, Tuple
from google import genai
from google.genai import types
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
RETRYABLE_ERRORS = ['rate limit', 'timeout', '429', '503', '500', 'connection']
CRITICAL_ERRORS = ["API_KEY_INVALID", "API key not valid", "UNAUTHENTICATED"]
QUOTA_ERRORS = ["RESOURCE_EXHAUSTED", "quota"]


class GeminiAPIClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: float = DEFAULT_RETRY_DELAY, http_options = None):
        """Initialize Gemini client with API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY env variable not set")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        try:
            self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """Retry a function call on failure with exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                # Check if it's a retryable error
                error_str = str(e).lower()
                if any(x in error_str for x in RETRYABLE_ERRORS):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"API request failed (attempt {attempt + 1}/{self.max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                # For non-retryable errors or final attempt, raise immediately
                raise e
        
        # If all retries failed, raise the last error
        if last_error:
            raise last_error
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
        
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if np_image.shape[-1] == 1:
            np_image = np_image.squeeze(-1)
            return Image.fromarray(np_image, mode='L')
        elif np_image.shape[-1] == 3:
            return Image.fromarray(np_image, mode='RGB')
        elif np_image.shape[-1] == 4:
            # Convert RGBA to RGB for sending to API
            return Image.fromarray(np_image[:, :, :3], mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {np_image.shape}")
    
    def bytes_to_tensor(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to torch tensor."""
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGBA for ComfyUI
        if image.mode != 'RGBA':
            if image.mode == 'RGB':
                # Add alpha channel
                image = image.convert('RGBA')
            else:
                image = image.convert('RGBA')
        
        np_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_array)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        top_p: float = 0.95,
        max_output_tokens: int = 8192
    ) -> str:
        """Generate text using Gemini API."""
        contents = []
        
        # Combine system prompt and main prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Add text prompt
        contents.append(full_prompt)
        
        # Add images if provided
        if images:
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                contents.append(pil_image)
        
        # Configure generation
        config = types.GenerateContentConfig(
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        
        if seed is not None:
            config.seed = seed
        
        # Generate content with retry logic
        response = self._retry_on_failure(
            self.client.models.generate_content,
            model=model,
            contents=contents,
            config=config
        )
        
        # Extract text from response
        return response.text if response.text else ""
    
    
    def generate_image(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash-image-preview",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> tuple[List[torch.Tensor], str]:
        """Generate images using Gemini API with image generation model."""
        
        # For image generation with gemini-2.5-flash-image-preview, use chat interface
        chat = self.client.chats.create(model=model)
        
        # Prepare message contents
        message_contents = []
        
        # Combine system prompt and main prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        message_contents.append(full_prompt)
        
        # Add input images if provided
        if images:
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                message_contents.append(pil_image)
        
        # Send message to generate image with retry logic
        response = self._retry_on_failure(
            chat.send_message,
            message_contents
        )
        
        output_images = []
        text_output = ""
        
        # Process response
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle text parts
                    if part.text is not None:
                        text_output += part.text
                    # Handle image parts
                    elif part.inline_data is not None:
                        try:
                            # Get image data - inline_data.data is already bytes
                            if hasattr(part.inline_data, 'data'):
                                image_bytes = part.inline_data.data
                            else:
                                # If it's a dict
                                image_bytes = part.inline_data.get('data', b'')
                            
                            if image_bytes:
                                # Convert bytes to tensor
                                tensor = self.bytes_to_tensor(image_bytes)
                                output_images.append(tensor)
                        except Exception as e:
                            print(f"Error processing image from response: {e}")
                            import traceback
                            traceback.print_exc()
        
        # Return empty tensor if no images generated
        if not output_images:
            dummy_tensor = torch.zeros((1, 64, 64, 4))
            output_images = [dummy_tensor]
        
        return output_images, text_output
    
    async def generate_text_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        top_p: float = 0.95,
        max_output_tokens: int = 8192
    ) -> str:
        """Generate text using Gemini API with async client."""
        contents = []
        
        # Combine system prompt and main prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Add text prompt
        contents.append(full_prompt)
        
        # Add images if provided
        if images:
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                contents.append(pil_image)
        
        # Configure generation
        config = types.GenerateContentConfig(
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        
        if seed is not None:
            config.seed = seed
        
        # Generate content with async client
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        # Extract text from response
        return response.text if response.text else ""
    
    async def generate_image_async(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash-image-preview",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> tuple[List[torch.Tensor], str]:
        """Generate images using async Gemini API."""
        
        # For image generation, we still need to use chat interface
        # Note: This might not be fully async, but we wrap it for consistency
        chat = self.client.aio.chats.create(model=model)
        
        # Prepare message contents
        message_contents = []
        
        # Combine system prompt and main prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        message_contents.append(full_prompt)
        
        # Add input images if provided
        if images:
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                message_contents.append(pil_image)
        
        # Send message to generate image
        response = await chat.send_message(message_contents)
        
        output_images = []
        text_output = ""
        
        # Process response
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle text parts
                    if part.text is not None:
                        text_output += part.text
                    # Handle image parts
                    elif part.inline_data is not None:
                        try:
                            # Get image data
                            if hasattr(part.inline_data, 'data'):
                                image_bytes = part.inline_data.data
                            else:
                                image_bytes = part.inline_data.get('data', b'')
                            
                            if image_bytes:
                                # Convert bytes to tensor
                                tensor = self.bytes_to_tensor(image_bytes)
                                output_images.append(tensor)
                        except Exception as e:
                            print(f"Error processing image from response: {e}")
        
        # Return empty tensor if no images generated
        if not output_images:
            dummy_tensor = torch.zeros((1, 64, 64, 4))
            output_images = [dummy_tensor]
        
        return output_images, text_output
    
    async def generate_batch_concurrent_async(
        self,
        prompts_configs: List[Dict[str, Any]],
        model: str,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Union[tuple, str]]:
        """Generate multiple requests concurrently using async client."""
        
        async def process_single_request(index: int, config: Dict[str, Any]):
            """Process a single request and handle errors."""
            try:
                if "image" in model.lower():
                    # Generate image
                    result = await self.generate_image_async(
                        prompt=config.get('prompt'),
                        model=model,
                        images=config.get('images'),
                        seed=config.get('seed'),
                        system_prompt=config.get('system_prompt')
                    )
                else:
                    # Generate text
                    result = await self.generate_text_async(
                        prompt=config.get('prompt'),
                        system_prompt=config.get('system_prompt'),
                        model=model,
                        images=config.get('images'),
                        seed=config.get('seed'),
                        top_p=config.get('top_p', 0.95),
                        max_output_tokens=config.get('max_output_tokens', 8192)
                    )
                
                print(f"[NanoBanana Gemini] Completed batch {index + 1}/{len(prompts_configs)}")
                if progress_callback:
                    progress_callback(index)
                return result
                
            except Exception as e:
                error_str = str(e)
                print(f"[NanoBanana Gemini] Error in batch {index + 1}: {error_str}")
                if progress_callback:
                    progress_callback(index)
                
                # Check for critical errors that should stop all processing
                if any(x in error_str for x in CRITICAL_ERRORS):
                    # API key error - stop all batch processing
                    raise ValueError(f"Invalid API key. Please check your Gemini API key.")
                elif any(x in error_str.lower() for x in QUOTA_ERRORS):
                    # Quota exceeded - stop all batch processing
                    raise ValueError(f"API quota exceeded. Please check your Gemini API usage limits.")
                elif "INVALID_ARGUMENT" in error_str and "model" in error_str.lower():
                    # Invalid model - stop all batch processing
                    raise ValueError(f"Invalid model specified. Please check the model name.")
                
                # For other errors, continue with error placeholder but log them clearly
                print(f"[NanoBanana Gemini] Non-critical error, continuing with placeholder for batch {index + 1}")
                
                # Return error placeholder
                if "image" in model.lower():
                    return ([torch.zeros((1, 64, 64, 4))], f"Error in batch {index + 1}: {error_str}")
                else:
                    return f"Error in batch {index + 1}: {error_str}"
        
        # Create tasks for all configs
        tasks = [
            process_single_request(i, config) 
            for i, config in enumerate(prompts_configs)
        ]
        
        # Run all tasks concurrently using asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results
    
    def generate_batch_concurrent(
        self,
        prompts_configs: List[Dict[str, Any]],
        model: str,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[Union[tuple, str]]:
        """Wrapper to run async batch generation in sync context."""
        
        import threading
        import concurrent.futures
        
        def run_async_in_thread():
            """Run async code in a separate thread with its own event loop."""
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.generate_batch_concurrent_async(
                        prompts_configs=prompts_configs,
                        model=model,
                        progress_callback=progress_callback
                    )
                )
            finally:
                loop.close()
        
        # Run the async code in a separate thread to avoid event loop conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async_in_thread)
            return future.result()
