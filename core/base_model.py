import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
import gc
from contextlib import contextmanager

class BaseMultimodalModel(ABC, nn.Module):
    """Base class for all multimodal AI models"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.is_loaded = False
        self._model = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory"""
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate output from input"""
        pass
    
    @contextmanager
    def model_context(self):
        """Context manager for automatic model loading/unloading"""
        try:
            if not self.is_loaded:
                self.load_model()
            yield self
        finally:
            if hasattr(self, 'auto_unload') and self.auto_unload:
                self.unload_model()
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device"""
        return tensor.to(self.device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
            }
        return {"cpu_memory": "N/A"}

class ModelManager:
    """Manages multiple models with memory optimization"""
    
    def __init__(self, max_loaded_models: int = 2):
        self.models: Dict[str, BaseMultimodalModel] = {}
        self.loaded_models: List[str] = []
        self.max_loaded_models = max_loaded_models
        self.logger = logging.getLogger("ModelManager")
    
    def register_model(self, name: str, model: BaseMultimodalModel):
        """Register a model"""
        self.models[name] = model
        self.logger.info(f"Registered model: {name}")
    
    def load_model(self, name: str) -> BaseMultimodalModel:
        """Load a specific model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        
        model = self.models[name]
        
        if not model.is_loaded:
            # Unload oldest model if at capacity
            if len(self.loaded_models) >= self.max_loaded_models:
                oldest_model = self.loaded_models.pop(0)
                self.models[oldest_model].unload_model()
                self.logger.info(f"Unloaded model: {oldest_model}")
            
            model.load_model()
            self.loaded_models.append(name)
            self.logger.info(f"Loaded model: {name}")
        
        return model
    
    def unload_model(self, name: str):
        """Unload a specific model"""
        if name in self.models and self.models[name].is_loaded:
            self.models[name].unload_model()
            if name in self.loaded_models:
                self.loaded_models.remove(name)
            self.logger.info(f"Unloaded model: {name}")
    
    def unload_all(self):
        """Unload all models"""
        for name in list(self.loaded_models):
            self.unload_model(name)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return self.loaded_models.copy()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for all models"""
        stats = {}
        for name, model in self.models.items():
            if model.is_loaded:
                stats[name] = model.get_memory_usage()
        return stats