import time
import uuid
from agent.adapters.base import Adapter, TokenCB
from typing import Dict, Any, Optional

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoProcessor = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

import torch

# class VLMExampleAdapter(Adapter):
    # name = "VLMExample"
    # modalities = ["image", "text"]
    
    # def load(self, config):
    #     """Load the model (simulated for this example)"""
    #     self.config = config
    #     print(f"Loaded VLM adapter with config: {config}")
    
    # def warmup(self):
    #     """Warmup the model"""
    #     print("Warming up VLM model...")
    #     time.sleep(0.1)  # Simulate warmup
    
    # def teardown(self):
    #     """Cleanup resources"""
    #     print("Cleaning up VLM adapter")
    
    # def vlm_generate(self, prompt: str, media: dict, stream: bool = True, 
    #                  on_token: TokenCB = None, **kw) -> dict:
    #     """Generate text from image + prompt"""
    #     t_start = time.time()
        
    #     # Simulate VLM inference
    #     tokens = ["A", " cat", " sitting", " on", " a", " mat", "."]
    #     first_t = None
    #     output_tokens = []
        
    #     for i, token in enumerate(tokens):
    #         time.sleep(0.05)  # Simulate token generation time
    #         ts = time.time()
            
    #         if first_t is None:
    #             first_t = ts  # First token timestamp
            
    #         if on_token:
    #             on_token(token, ts)
            
    #         output_tokens.append(token)
        
    #     t_end = time.time()
        
    #     return {
    #         "trial_id": str(uuid.uuid4()),
    #         "output_text": "".join(output_tokens),
    #         "timestamps": {
    #             "t_start": t_start,
    #             "t_first": first_t,
    #             "t_end": t_end
    #         }
    #     }
    
    # # Implement other required methods
    # def infer_image(self, batch, **kw): return {"result": "image_inference"}
    # def infer_video(self, frames, on_first_frame=None, **kw): return {"result": "video_inference"}
    # def infer_audio(self, samples, on_first_audio=None, **kw): return {"result": "audio_inference"}
    # def encode_image(self, img): return {"code": "encoded"}
    # def decode_image(self, code): return "decoded_image"
    # def encode_video_frame(self, frame): return {"code": "encoded_frame"}
    # def decode_video_frame(self, code): return "decoded_frame"
    # def latent_entropy(self, codes): return 0.5

class HuggingFaceVLMAdapter(Adapter):
    """Real VLM adapter using Hugging Face models"""
    
    name = "HuggingFaceVLM"
    modalities = ["image", "text"]
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEVICE: {self.device}")
    
    def load(self, config: Dict[str, Any]):
        """Load a Hugging Face VLM model"""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library not available. Install with: pip install transformers")
        
        if not config.get("model_name"):
            raise ValueError("model_name is required")
        
        model_name = config.get("model_name")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config.get("precision") == "fp16" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)
        

    
    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the model"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # For real models, you'd process the image here
        # For now, we'll use the prompt directly
        return {
            "prompt": sample.get("prompt", ""),
            "image_path": sample.get("image", ""),
            "processed": True
        }
    
    def infer(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference with the model"""
        if not self.model:
            raise ValueError("Model not loaded")
    
        return self._real_inference(prepared)

    
    def _real_inference(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Real model inference"""
        prompt = prepared["prompt"]
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to(self.device)  

        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=prepared.get("max_new_tokens", 50),
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        end_time = time.time()
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "input_tokens": inputs.shape[1],
            "output_tokens": outputs.shape[1],
            "inference_time": end_time - start_time
        }

    
    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the raw model output"""
        return {
            "trial_id": str(uuid.uuid4()),
            "output_text": raw_output["generated_text"],
            "timing": {
                "t_start": time.time() - raw_output["inference_time"],
                "t_first": time.time() - raw_output["inference_time"] + 0.01,
                "t_end": time.time()
            },
            "pred": {
                "text": raw_output["generated_text"],
                "tokens": raw_output["generated_text"].split()
            },
            "tokens": raw_output["generated_text"].split(),  # Keep for backward compatibility
            "stats": {
                "input_tokens": raw_output["input_tokens"],
                "output_tokens": raw_output["output_tokens"],
                "inference_time_s": raw_output["inference_time"]
            }
        }

    
    def teardown(self):
        """Clean up resources"""
        if self.model:
            del self.model
            del self.processor
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 Model resources cleaned up")
    

    def infer_image(self, batch, **kw): return {"result": "image_inference"}
    def infer_video(self, frames, on_first_frame=None, **kw): return {"result": "video_inference"}
    def infer_audio(self, samples, on_first_audio=None, **kw): return {"result": "audio_inference"}
    def vlm_generate(self, prompt, media, stream=True, on_token=None, **kw):
        # Use the main inference pipeline
        sample = {"prompt": prompt, "image": media.get("image", "")}
        prepared = self.prepare(sample)
        raw_output = self.infer(prepared)
        return self.postprocess(raw_output)
    
    def encode_image(self, img): return {"code": "encoded"}
    def decode_image(self, code): return "decoded_image"
    def encode_video_frame(self, frame): return {"code": "encoded_frame"}
    def decode_video_frame(self, code): return "decoded_frame"
    def latent_entropy(self, codes): return 0.5


AdapterImpl = HuggingFaceVLMAdapter
