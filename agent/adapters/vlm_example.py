# agent/adapters/vlm_example.py
# Generic Hugging Face VLM adapter with:
# - config-first discovery
# - optional preloading of vendor modules to register custom classes
# - processor-first input building
# - streaming generation for TTFT
# No model-specific branches.

import os
import time
import uuid
import threading
import importlib
from typing import Dict, Any, Optional, List

import torch

try:
    from packaging import version as _pkg_version
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
        TextIteratorStreamer,
        __version__ as _tf_version,
    )
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False
    _tf_version = "0.0.0"
    AutoConfig = AutoModelForCausalLM = AutoModel = AutoProcessor = AutoTokenizer = TextIteratorStreamer = None

try:
    from PIL import Image
    HAS_PIL = True
except Exception:  # pragma: no cover
    HAS_PIL = False
    Image = None

# Project adapter interface
from agent.adapters.base import Adapter, TokenCB  # type: ignore


def _now() -> float:
    return time.time()

import inspect, importlib, pkgutil
from typing import Iterable
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

def _issubclass_safe(obj, base):
    try:
        return inspect.isclass(obj) and issubclass(obj, base)
    except Exception:
        return False

def _walk_submodules(root_mod) -> Iterable[object]:
    """Yield the root module and all importable submodules (best-effort)."""
    yield root_mod
    if hasattr(root_mod, "__path__"):
        prefix = root_mod.__name__ + "."
        for m in pkgutil.walk_packages(root_mod.__path__, prefix):
            name = m.name
            try:
                yield importlib.import_module(name)
            except Exception:
                # Ignore submodules that fail to import
                continue

def _auto_register_transformers_types(preloaded_module_names: Iterable[str]) -> None:
    """
    Generic: find any PretrainedConfig / PreTrainedModel in the given modules
    and register them with AutoConfig/AutoModel/AutoModelForCausalLM.
    This avoids hardcoding model names or class symbols.
    """
    for mod_name in preloaded_module_names or []:
        mod = importlib.import_module(mod_name)
        for sub in _walk_submodules(mod):
            for _, obj in inspect.getmembers(sub, inspect.isclass):
                # Register config by model_type
                if _issubclass_safe(obj, PretrainedConfig):
                    model_type = getattr(obj, "model_type", None)
                    if model_type:
                        try:
                            AutoConfig.register(model_type, obj)
                        except Exception:
                            pass
                # Register model heads by config_class
                if _issubclass_safe(obj, PreTrainedModel):
                    cfg_cls = getattr(obj, "config_class", None)
                    if cfg_cls:
                        try:
                            AutoModel.register(cfg_cls, obj)
                        except Exception:
                            pass
                        try:
                            AutoModelForCausalLM.register(cfg_cls, obj)
                        except Exception:
                            pass

class HuggingFaceVLMAdapter(Adapter):
    """
    Adaptive, model-agnostic VLM adapter.

    Key behavior (generic):
      - Optionally import vendor modules listed in config['preload_modules'] BEFORE touching HF APIs.
        This lets vendor packages register custom config/model classes with transformers.
      - Discover capabilities from AutoConfig (no weights first).
      - Load AutoModel(ForCausalLM)/AutoProcessor/AutoTokenizer with trust_remote_code=True.
      - Build inputs via Processor where available (multimodal-safe).
      - Stream tokens to measure TTFT.
    """

    name = "HuggingFaceVLM"
    modalities = ["image", "text"]

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_info: Dict[str, Any] = {}
        self.inference_method: Optional[str] = None
        self._min_tf = "4.54.0"

    # ---------- Public Adapter API ----------

    def load(self, config: Dict[str, Any]):
        """
        Load any HF multimodal checkpoint that either:
          - publishes auto_map entries, OR
          - has a vendor package that registers the custom classes when imported.

        Config keys:
          - model_name (str): HF repo or local path
          - precision (str): "fp32" | "fp16" | "bf16"
          - preload_modules (List[str], optional): modules to import before discovery
          - device_map (str|None, optional): override ("auto" to let HF place weights)
        """
        self._ensure_deps()

        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("config.model_name is required")

        # 0) Optionally import vendor modules up front (generic mechanism)
        for mod in config.get("preload_modules", []) or []:
            importlib.import_module(mod)
        
        _auto_register_transformers_types(config.get("preload_modules", []))
        precision = str(config.get("precision", "fp32")).lower()
        torch_dtype = (
            torch.float16 if precision == "fp16"
            else torch.bfloat16 if precision in ("bf16", "bfloat16")
            else torch.float32
        )

        device_map = config.get("device_map", ("auto" if self.device == "cuda" else None))

        self.model_info = self._discover_model_interface(model_name)

        # 2) load model
        load_kwargs: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch_dtype)
        if device_map:
            load_kwargs["device_map"] = device_map


        if self.model_info["has_auto_causal_lm"]:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)

        # 3) load processor & tokenizer (processor-first when present)
        if self.model_info["has_auto_processor"]:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        if self.processor is not None:
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4) determine strategy from runtime attributes
        interface = {
            "has_generate": hasattr(self.model, "generate"),
            "has_language_model": hasattr(self.model, "language_model"),
            "has_prepare_inputs_embeds": hasattr(self.model, "prepare_inputs_embeds"),
            "attributes": [a for a in dir(self.model) if not a.startswith("_")],
        }
        self.inference_method = self._determine_inference_strategy(interface)

        # respect device if no device_map
        if self.device == "cuda" and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)

    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Adapter not loaded")
        prompt = sample.get("prompt") or sample.get("question") or ""
        img_path = sample.get("image") or sample.get("image_path")
        max_new = int(sample.get("max_new_tokens", 128))
        return {"prompt": prompt, "image_path": img_path, "max_new_tokens": max_new}

    def infer(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Adapter not loaded")
        if self.inference_method == "custom_vlm":
            return self._custom_vlm_inference(prepared, on_token=None)
        if self.inference_method == "standard_generate":
            return self._standard_generate_inference(prepared, on_token=None)
        if self.inference_method == "forward_pass":
            return self._forward_inference(prepared)
        raise RuntimeError(f"Unknown inference method: {self.inference_method}")

    def postprocess(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        t_end = _now()
        t_start = t_end - float(raw_output["inference_time"])
        t_first = float(raw_output.get("t_first", t_start))
        text = raw_output.get("generated_text", "")
        toks = text.split()
        return {
            "trial_id": str(uuid.uuid4()),
            "output_text": text,
            "timing": {"t_start": t_start, "t_first": t_first, "t_end": t_end},
            "pred": {"text": text, "tokens": toks},
            "tokens": toks,
            "stats": {
                "input_tokens": int(raw_output.get("input_tokens", 0)),
                "output_tokens": int(raw_output.get("output_tokens", 0)),
                "inference_time": float(raw_output["inference_time"]),
                "inference_method": raw_output.get("method", "unknown"),
            },
        }

    def teardown(self):
        if self.model is not None:
            try:
                del self.model
                del self.processor
                del self.tokenizer
            except Exception:
                pass
            self.model = self.processor = self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def vlm_generate(self, prompt: str, media: Dict[str, Any], stream: bool = True,
                     on_token: Optional[TokenCB] = None, **kw) -> Dict[str, Any]:
        sample = {"prompt": prompt, "image": media.get("image", None), **kw}
        pre = self.prepare(sample)
        if self.inference_method == "custom_vlm":
            raw = self._custom_vlm_inference(pre, on_token=on_token if stream else None)
        else:
            raw = self._standard_generate_inference(pre, on_token=on_token if stream else None)
        return self.postprocess(raw)


    ## MODEL ADAPTATION LOGIC

    def _ensure_deps(self):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not available. Install with: pip install transformers")
        if _pkg_version.parse(_tf_version) < _pkg_version.parse(self._min_tf):
            raise RuntimeError(
                f"Transformers>={self._min_tf} required for dynamic multimodal checkpoints; found {_tf_version}"
            )

    def _discover_model_interface(self, model_name: str) -> Dict[str, Any]:
        """
        Cheap, robust discovery: only read config (no weights).
        """
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        auto_map = getattr(cfg, "auto_map", {}) or {}
        info = {
            "model_type": getattr(cfg, "model_type", None),
            "raw_auto_map": dict(auto_map),
            "has_auto_causal_lm": "AutoModelForCausalLM" in auto_map,
            "has_auto_model": "AutoModel" in auto_map,
            "has_auto_processor": "AutoProcessor" in auto_map,
            "has_vision": hasattr(cfg, "vision_config"),
            "has_audio": hasattr(cfg, "audio_config"),
        }
        return info

    def _determine_inference_strategy(self, interface: Dict[str, Any]) -> str:
        if interface.get("has_language_model") and interface.get("has_prepare_inputs_embeds"):
            return "custom_vlm"            # VLM split into encoder + language_model
        if interface.get("has_generate"):
            return "standard_generate"     # vanilla .generate path
        if "forward" in interface.get("attributes", []):
            return "forward_pass"          # last resort
        raise RuntimeError(f"Model has no usable inference methods. Available: {interface['attributes']}")

    ## INFERENCE LOGIC

    def _custom_vlm_inference(self, prepared: Dict[str, Any],
                              on_token: Optional[TokenCB]) -> Dict[str, Any]:
        """
        Processor-first path for models that expose:
          - processor(...) to pack conversations + images
          - model.prepare_inputs_embeds(...)
          - model.language_model.generate(...)
        """
        prompt = prepared.get("prompt", "")
        image_path = prepared.get("image_path", None)
        max_new = int(prepared.get("max_new_tokens", 128))

        pil_images: List[Any] = []
        if image_path:
            if not HAS_PIL:
                raise RuntimeError("Pillow not installed; pip install pillow")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_images = [Image.open(image_path).convert("RGB")]

        if self.processor is None:
            raise RuntimeError("NO PROCESSOR")

        conversation = [
            {"role": "User", "content": "<image_placeholder>" + prompt,
             "images": [image_path] if image_path else []},
            {"role": "Assistant", "content": ""},
        ]

        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        )

        if hasattr(prepare_inputs, "to"):
            prepare_inputs = prepare_inputs.to(self.model.device)
        elif isinstance(prepare_inputs, dict):
            prepare_inputs = {k: (v.to(self.model.device) if torch.is_tensor(v) else v)
                              for k, v in prepare_inputs.items()}

        t0 = _now()
        t_first: Optional[float] = None

        if hasattr(self.model, "prepare_inputs_embeds") and hasattr(self.model, "language_model"):
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=getattr(prepare_inputs, "attention_mask", None) or prepare_inputs.get("attention_mask"),
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
                streamer=streamer,
            )

            out_holder: Dict[str, Any] = {}

            def _gen():
                out = self.model.language_model.generate(**gen_kwargs)
                out_holder["out"] = out

            th = threading.Thread(target=_gen, daemon=True)
            th.start()

            chunks: List[str] = []
            for chunk in streamer:
                if t_first is None:
                    t_first = _now()
                chunks.append(chunk)
                if on_token:
                    try:
                        on_token(chunk)
                    except Exception:
                        pass
            th.join()

            text = "".join(chunks)
            t1 = _now()
            out = out_holder.get("out")
            n_out = int(out.shape[1]) if isinstance(out, torch.Tensor) else 0
            in_ids = getattr(prepare_inputs, "input_ids", None) or prepare_inputs.get("input_ids")
            n_in = int(in_ids.shape[1]) if torch.is_tensor(in_ids) else 0

            return {
                "generated_text": text,
                "input_tokens": n_in,
                "output_tokens": n_out,
                "inference_time": t1 - t0,
                "method": "custom_vlm",
                "t_first": t_first or t0,
            }

        if hasattr(self.model, "generate"):
            return self._generate_with_streamer(
                self.model, prepare_inputs, max_new, on_token=on_token, method_tag="standard_generate_vlm"
            )

        raise RuntimeError("Model exposes neither VLM-specific nor standard generation methods.")

    def _standard_generate_inference(self, prepared: Dict[str, Any],
                                     on_token: Optional[TokenCB]) -> Dict[str, Any]:
        prompt = prepared.get("prompt", "")
        image_path = prepared.get("image_path", None)
        max_new = int(prepared.get("max_new_tokens", 128))

        inputs = None

        if self.processor is not None:
            pil_images: Optional[List[Any]] = None
            if image_path:
                if not HAS_PIL:
                    raise RuntimeError("Pillow not installed; pip install pillow")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                pil_images = [Image.open(image_path).convert("RGB")]
            try:
                inputs = self.processor(text=prompt, images=pil_images or None, return_tensors="pt")
            except TypeError:
                pass

        if inputs is None:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        elif isinstance(inputs, dict):
            inputs = {k: (v.to(self.model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        return self._generate_with_streamer(
            self.model, inputs, max_new, on_token=on_token, method_tag="standard_generate"
        )

    def _forward_inference(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        prompt = prepared.get("prompt", "")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        t0 = _now()
        with torch.no_grad():
            outputs = self.model(**inputs)
        t1 = _now()

        logits = getattr(outputs, "logits", None)
        if logits is None:
            last = getattr(outputs, "last_hidden_state", None)
            if last is None:
                raise RuntimeError("Model outputs lack 'logits' and 'last_hidden_state'.")
            logits = last
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        return {
            "generated_text": text,
            "input_tokens": int(inputs.get("input_ids", torch.empty(1, 0)).shape[1]),
            "output_tokens": int(pred_ids.shape[1]),
            "inference_time": t1 - t0,
            "method": "forward",
            "t_first": t0,
        }

    # ---------- Utilities ----------

    def _generate_with_streamer(self, model, inputs, max_new_tokens: int,
                                on_token: Optional[TokenCB], method_tag: str) -> Dict[str, Any]:
        if not hasattr(model, "generate"):
            raise RuntimeError("Model has no .generate()")

        t0 = _now()
        t_first: Optional[float] = None

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **(inputs if isinstance(inputs, dict) else {}),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
        )

        out_holder: Dict[str, Any] = {}

        def _gen():
            out = model.generate(**gen_kwargs)
            out_holder["out"] = out

        th = threading.Thread(target=_gen, daemon=True)
        th.start()

        chunks: List[str] = []
        for chunk in streamer:
            if t_first is None:
                t_first = _now()
            chunks.append(chunk)
            if on_token:
                try:
                    on_token(chunk)
                except Exception:
                    pass
        th.join()

        text = "".join(chunks)
        t1 = _now()

        out = out_holder.get("out")
        n_out = int(out.shape[1]) if isinstance(out, torch.Tensor) else 0

        in_ids = None
        if hasattr(inputs, "input_ids"):
            in_ids = inputs.input_ids
        elif isinstance(inputs, dict):
            in_ids = inputs.get("input_ids")
        n_in = int(in_ids.shape[1]) if torch.is_tensor(in_ids) else 0

        return {
            "generated_text": text,
            "input_tokens": n_in,
            "output_tokens": n_out,
            "inference_time": t1 - t0,
            "method": method_tag,
            "t_first": t_first or t0,
        }

    def infer_image(self, batch, **kw): 
        raise NotImplementedError("Image inference not implemented in this adapter")
    
    def infer_video(self, frames, on_first_frame=None, **kw): 
        raise NotImplementedError("Video inference not implemented in this adapter")
    
    def infer_audio(self, samples, on_first_audio=None, **kw): 
        raise NotImplementedError("Audio inference not implemented in this adapter")
    
    def vlm_generate(self, prompt, media, stream=True, on_token=None, **kw):
        # Use the main inference pipeline
        sample = {"prompt": prompt, "image": media.get("image", "")}
        prepared = self.prepare(sample)
        raw_output = self.infer(prepared)
        return self.postprocess(raw_output)
    
    def encode_image(self, img): 
        raise NotImplementedError("Image encoding not implemented in this adapter")
    
    def decode_image(self, code): 
        raise NotImplementedError("Image decoding not implemented in this adapter")
    
    def encode_video_frame(self, frame): 
        raise NotImplementedError("Video frame encoding not implemented in this adapter")
    
    def decode_video_frame(self, code): 
        raise NotImplementedError("Video frame decoding not implemented in this adapter")
    
    def latent_entropy(self, codes): 
        raise NotImplementedError("Latent entropy not implemented in this adapter")

AdapterImpl = HuggingFaceVLMAdapter
