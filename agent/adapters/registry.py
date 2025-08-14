"""
Adapter registry for mapping suite configurations to adapter implementations.
"""

from typing import Dict, Type, Any
from .base import Adapter
from .vlm_example import HuggingFaceVLMAdapter

# Registry mapping adapter types to their implementations
ADAPTER_REGISTRY: Dict[str, Type[Adapter]] = {
    "vlm_example": HuggingFaceVLMAdapter,  # Default fallback
    "huggingface": HuggingFaceVLMAdapter,
    "huggingface_vlm": HuggingFaceVLMAdapter,
    "hf": HuggingFaceVLMAdapter,
    "default": HuggingFaceVLMAdapter,
}

def get_adapter(config: Dict) -> Adapter:
    """
    Create and return an adapter based on the configuration.
    
    Args:
        config: Suite configuration containing adapter_config
        
    Returns:
        Configured adapter instance
    """
    adapter_config = config.get("adapter_config", {})
    
    # Determine adapter type from config
    adapter_type = None
    
    # Check for explicit adapter type
    if "adapter_type" in adapter_config:
        adapter_type = adapter_config["adapter_type"]
    elif "model_type" in adapter_config:
        adapter_type = adapter_config["model_type"]
    elif "model_name" in adapter_config:
        # If model_name is present, assume HuggingFace
        adapter_type = "huggingface"
    else:
        # Default fallback
        adapter_type = "default"
    
    # Get adapter class from registry
    adapter_class = ADAPTER_REGISTRY.get(adapter_type, ADAPTER_REGISTRY["default"])
    
    # Create adapter instance
    adapter = adapter_class()
    
    # Load the adapter with configuration
    try:
        adapter.load(adapter_config)
        print(f"✅ Loaded adapter: {adapter_class.__name__} with config: {adapter_config}")
    except Exception as e:
        print(f"❌ Failed to load adapter {adapter_class.__name__}: {e}")
        # Continue anyway - some adapters might work without explicit loading
    
    return adapter
