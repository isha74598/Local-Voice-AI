"""
Vision processor for describing camera images using Ollama vision models.
"""
import requests
import base64
import logging
import asyncio
from typing import Optional

logger = logging.getLogger("local-agent.vision")


def _call_vision_api(image_base64: str, question: str, model: str, base_url: str) -> Optional[str]:
    """
    Synchronous function to call Ollama vision API.
    This will be run in an executor to avoid blocking.
    """
    try:
        # Remove data URI prefix if present
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]
        
        # Ollama vision API endpoint
        url = f"{base_url}/api/generate"
        
        # Prepare the request
        # Ollama expects base64 string directly (without data URI prefix)
        payload = {
            "model": model,
            "prompt": f"{question} Describe what you see in this image in detail. Be specific about objects, colors, layout, and any notable features.",
            "images": [image_base64],  # Ollama expects base64 string
            "stream": False
        }
        
        # Make the request
        response = requests.post(url, json=payload, timeout=60)  # Longer timeout for vision processing
        response.raise_for_status()
        
        result = response.json()
        description = result.get("response", "").strip()
        
        if description:
            logger.info(f"Vision model description received: {len(description)} chars")
            return description
        else:
            logger.warning("Vision model returned empty response")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling vision model API: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing vision: {e}")
        return None


async def describe_image_with_vision_model(
    image_base64: str,
    question: str,
    model: str = "qwen2.5vl:7b",  # Use available vision model
    base_url: str = "http://localhost:11434"
) -> Optional[str]:
    """
    Send image to Ollama vision model and get description.
    Runs the API call in an executor to avoid blocking.
    
    Args:
        image_base64: Base64 encoded image (with or without data URI prefix)
        question: The user's question about the image
        model: Vision model name (default: qwen2.5vl:7b)
        base_url: Ollama API base URL
    
    Returns:
        Description of the image, or None if failed
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _call_vision_api,
        image_base64,
        question,
        model,
        base_url
    )
