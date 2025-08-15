"""
Adapter registry for mapping suite configurations to adapter implementations.
"""

from typing import Dict, Type, Any
from .base import Adapter
from .vlm_example import HuggingFaceVLMAdapter


ADAPTER_REGISTRY: Dict[str, Type[Adapter]] = {
    "vlm_example": HuggingFaceVLMAdapter,  
    "huggingface": HuggingFaceVLMAdapter,
    "huggingface_vlm": HuggingFaceVLMAdapter,
    "hf": HuggingFaceVLMAdapter,
    "default": HuggingFaceVLMAdapter,
}

def get_adapter(config: Dict) -> Adapter:
    adapter_config = config.get("adapter_config", {})
    
    adapter_type = None
    
    if "adapter_type" in adapter_config:
        adapter_type = adapter_config["adapter_type"]
    elif "model_type" in adapter_config:
        adapter_type = adapter_config["model_type"]
    elif "model_name" in adapter_config:
        adapter_type = "huggingface"
    else:
        adapter_type = "default"
    
    adapter_class = ADAPTER_REGISTRY.get(adapter_type, ADAPTER_REGISTRY["default"])
    
    adapter = adapter_class()

    adapter.load(adapter_config)
    print(f"Loaded adapter: {adapter_class.__name__} with config: {adapter_config}")

    return adapter
