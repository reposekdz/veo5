import torch
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from .base_model import ModelManager
from .text_to_image import AdvancedTextToImageModel
from .text_to_video import AdvancedTextToVideoModel
from .image_to_video import AdvancedImageToVideoModel
from .enhancement import AdvancedEnhancementModel
from .research_ai import AdvancedResearchAI
from .conversational_ai import AdvancedConversationalAI
from .tera_scale_ai import TeraScaleMultimodalAI
from .universal_processor import UniversalFileProcessor
from .knowledge_base import AdvancedKnowledgeBase
from config import config

class MultimodalAI:
    """Main multimodal AI system orchestrating all models"""
    
    def __init__(self, device: str = "cuda", max_loaded_models: int = 3):
        self.device = device
        self.model_manager = ModelManager(max_loaded_models)
        self.logger = logging.getLogger("MultimodalAI")
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "total_time": 0.0,
            "model_usage": {}
        }
    
    def _initialize_models(self):
        """Initialize all available models"""
        
        # Text-to-Image models
        for model_name in config.text_to_image_models.keys():
            model = AdvancedTextToImageModel(model_name, self.device)
            self.model_manager.register_model(f"t2i_{model_name}", model)
        
        # Text-to-Video models
        for model_name in config.text_to_video_models.keys():
            model = AdvancedTextToVideoModel(model_name, self.device)
            self.model_manager.register_model(f"t2v_{model_name}", model)
        
        # Image-to-Video models
        for model_name in config.image_to_video_models.keys():
            model = AdvancedImageToVideoModel(model_name, self.device)
            self.model_manager.register_model(f"i2v_{model_name}", model)
        
        # Enhancement models
        for model_name in config.image_enhancement_models.keys():
            model = AdvancedEnhancementModel(model_name, self.device)
            self.model_manager.register_model(f"enh_{model_name}", model)
        
        # Research AI
        research_ai = AdvancedResearchAI(self.device)
        self.model_manager.register_model("research_ai", research_ai)
        
        # Conversational AI
        conversational_ai = AdvancedConversationalAI(self.device)
        self.model_manager.register_model("conversational_ai", conversational_ai)
        
        # Tera-scale AI
        tera_scale_ai = TeraScaleMultimodalAI(self.device)
        self.model_manager.register_model("tera_scale_ai", tera_scale_ai)
        
        # Universal Processor
        universal_processor = UniversalFileProcessor(self.device)
        self.model_manager.register_model("universal_processor", universal_processor)
        
        # Knowledge Base
        knowledge_base = AdvancedKnowledgeBase(self.device)
        self.model_manager.register_model("knowledge_base", knowledge_base)
        
        self.logger.info(f"Initialized {len(self.model_manager.models)} models")
    
    def text_to_image(
        self,
        prompt: str,
        model: str = "sdxl",
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        enhance: bool = False,
        enhancement_model: str = "realesrgan",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate images from text prompt"""
        
        start_time = time.time()
        
        try:
            # Load and use text-to-image model
            t2i_model = self.model_manager.load_model(f"t2i_{model}")
            
            images = t2i_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                **kwargs
            )
            
            # Enhance images if requested
            if enhance:
                enh_model = self.model_manager.load_model(f"enh_{enhancement_model}")
                enhanced_images = []
                
                for img in images:
                    enhanced, _ = enh_model.enhance_image(img)
                    enhanced_images.append(Image.fromarray(enhanced))
                
                images = enhanced_images
            
            # Update stats
            generation_time = time.time() - start_time
            self._update_stats("text_to_image", generation_time, model)
            
            return {
                "images": images,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model": model,
                    "dimensions": (width, height),
                    "num_images": len(images),
                    "generation_time": generation_time,
                    "enhanced": enhance,
                    "seed": seed
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text-to-image generation failed: {e}")
            raise
    
    def text_to_video(
        self,
        prompt: str,
        model: str = "zeroscope",
        negative_prompt: Optional[str] = None,
        width: int = 576,
        height: int = 320,
        num_frames: int = 24,
        fps: int = 8,
        guidance_scale: float = 9.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        enhance: bool = False,
        enhancement_model: str = "basicvsr",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from text prompt"""
        
        start_time = time.time()
        
        try:
            # Load and use text-to-video model
            t2v_model = self.model_manager.load_model(f"t2v_{model}")
            
            frames, metadata = t2v_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                **kwargs
            )
            
            # Enhance video if requested
            if enhance:
                enh_model = self.model_manager.load_model(f"enh_{enhancement_model}")
                enhanced_frames, enh_metadata = enh_model.enhance_video(frames)
                frames = enhanced_frames
                metadata.update(enh_metadata)
            
            # Update stats
            generation_time = time.time() - start_time
            self._update_stats("text_to_video", generation_time, model)
            
            metadata.update({
                "generation_time": generation_time,
                "enhanced": enhance
            })
            
            return {
                "frames": frames,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Text-to-video generation failed: {e}")
            raise
    
    def image_to_video(
        self,
        image: Union[Image.Image, np.ndarray, str],
        model: str = "stable_video",
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 576,
        num_frames: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        seed: Optional[int] = None,
        enhance: bool = False,
        enhancement_model: str = "basicvsr",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from input image"""
        
        start_time = time.time()
        
        try:
            # Load and use image-to-video model
            i2v_model = self.model_manager.load_model(f"i2v_{model}")
            
            frames, metadata = i2v_model.generate(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                seed=seed,
                **kwargs
            )
            
            # Enhance video if requested
            if enhance:
                enh_model = self.model_manager.load_model(f"enh_{enhancement_model}")
                enhanced_frames, enh_metadata = enh_model.enhance_video(frames)
                frames = enhanced_frames
                metadata.update(enh_metadata)
            
            # Update stats
            generation_time = time.time() - start_time
            self._update_stats("image_to_video", generation_time, model)
            
            metadata.update({
                "generation_time": generation_time,
                "enhanced": enhance
            })
            
            return {
                "frames": frames,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Image-to-video generation failed: {e}")
            raise
    
    def enhance_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        model: str = "realesrgan",
        scale: int = 4,
        face_enhance: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Enhance image quality"""
        
        start_time = time.time()
        
        try:
            # Load and use enhancement model
            enh_model = self.model_manager.load_model(f"enh_{model}")
            
            enhanced, metadata = enh_model.enhance_image(
                image=image,
                scale=scale,
                face_enhance=face_enhance,
                **kwargs
            )
            
            # Update stats
            generation_time = time.time() - start_time
            self._update_stats("enhance_image", generation_time, model)
            
            metadata.update({
                "generation_time": generation_time
            })
            
            return {
                "image": Image.fromarray(enhanced),
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Image enhancement failed: {e}")
            raise
    
    def enhance_video(
        self,
        video_frames: List[Union[Image.Image, np.ndarray]],
        model: str = "basicvsr",
        scale: int = 4,
        temporal_consistency: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Enhance video quality"""
        
        start_time = time.time()
        
        try:
            # Load and use enhancement model
            enh_model = self.model_manager.load_model(f"enh_{model}")
            
            enhanced_frames, metadata = enh_model.enhance_video(
                video_frames=video_frames,
                scale=scale,
                temporal_consistency=temporal_consistency,
                **kwargs
            )
            
            # Update stats
            generation_time = time.time() - start_time
            self._update_stats("enhance_video", generation_time, model)
            
            metadata.update({
                "generation_time": generation_time
            })
            
            return {
                "frames": enhanced_frames,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Video enhancement failed: {e}")
            raise
    
    async def async_text_to_image(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version of text_to_image"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.text_to_image, *args, **kwargs)
    
    async def async_text_to_video(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version of text_to_video"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.text_to_video, *args, **kwargs)
    
    async def async_image_to_video(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version of image_to_video"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.image_to_video, *args, **kwargs)
    
    def batch_process(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        """Process multiple tasks in parallel"""
        
        results = []
        
        def process_task(task):
            task_type = task.get("type")
            task_args = task.get("args", {})
            
            if task_type == "text_to_image":
                return self.text_to_image(**task_args)
            elif task_type == "text_to_video":
                return self.text_to_video(**task_args)
            elif task_type == "image_to_video":
                return self.image_to_video(**task_args)
            elif task_type == "enhance_image":
                return self.enhance_image(**task_args)
            elif task_type == "enhance_video":
                return self.enhance_video(**task_args)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        
        # Process tasks in batches
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(process_task, task) for task in tasks]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch task failed: {e}")
                    results.append({"error": str(e)})
        
        return results
    
    def create_workflow(
        self,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a multi-step workflow"""
        
        workflow_results = []
        intermediate_data = {}
        
        for i, step in enumerate(steps):
            step_type = step.get("type")
            step_args = step.get("args", {})
            input_from = step.get("input_from")  # Reference to previous step output
            
            # Get input from previous step if specified
            if input_from is not None and input_from < len(workflow_results):
                prev_result = workflow_results[input_from]
                
                if step_type == "image_to_video" and "images" in prev_result:
                    step_args["image"] = prev_result["images"][0]
                elif step_type == "enhance_image" and "images" in prev_result:
                    step_args["image"] = prev_result["images"][0]
                elif step_type == "enhance_video" and "frames" in prev_result:
                    step_args["video_frames"] = prev_result["frames"]
            
            # Execute step
            try:
                if step_type == "text_to_image":
                    result = self.text_to_image(**step_args)
                elif step_type == "text_to_video":
                    result = self.text_to_video(**step_args)
                elif step_type == "image_to_video":
                    result = self.image_to_video(**step_args)
                elif step_type == "enhance_image":
                    result = self.enhance_image(**step_args)
                elif step_type == "enhance_video":
                    result = self.enhance_video(**step_args)
                else:
                    raise ValueError(f"Unknown step type: {step_type}")
                
                workflow_results.append(result)
                self.logger.info(f"Completed workflow step {i + 1}/{len(steps)}: {step_type}")
                
            except Exception as e:
                self.logger.error(f"Workflow step {i + 1} failed: {e}")
                workflow_results.append({"error": str(e)})
                break
        
        return {
            "steps": workflow_results,
            "success": all("error" not in result for result in workflow_results)
        }
    
    def _update_stats(self, operation: str, time_taken: float, model: str):
        """Update performance statistics"""
        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_time"] += time_taken
        
        if model not in self.generation_stats["model_usage"]:
            self.generation_stats["model_usage"][model] = {
                "count": 0,
                "total_time": 0.0
            }
        
        self.generation_stats["model_usage"][model]["count"] += 1
        self.generation_stats["model_usage"][model]["total_time"] += time_taken
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.generation_stats.copy()
        
        if stats["total_generations"] > 0:
            stats["average_time"] = stats["total_time"] / stats["total_generations"]
        
        # Add model efficiency stats
        for model, data in stats["model_usage"].items():
            if data["count"] > 0:
                data["average_time"] = data["total_time"] / data["count"]
        
        # Add memory stats
        stats["memory"] = self.model_manager.get_memory_stats()
        stats["loaded_models"] = self.model_manager.get_loaded_models()
        
        return stats
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by category"""
        return {
            "text_to_image": list(config.text_to_image_models.keys()),
            "text_to_video": list(config.text_to_video_models.keys()),
            "image_to_video": list(config.image_to_video_models.keys()),
            "image_enhancement": list(config.image_enhancement_models.keys()),
            "video_enhancement": list(config.video_enhancement_models.keys())
        }
    
    def research(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform research query with paper analysis and web search"""
        start_time = time.time()
        
        try:
            research_model = self.model_manager.load_model("research_ai")
            result = research_model.chat(query, **kwargs)
            
            generation_time = time.time() - start_time
            self._update_stats("research", generation_time, "research_ai")
            
            result["metadata"] = {
                "query": query,
                "generation_time": generation_time,
                "model": "research_ai"
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Research query failed: {e}")
            raise
    
    def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """Advanced conversational AI with memory and personality"""
        start_time = time.time()
        
        try:
            chat_model = self.model_manager.load_model("conversational_ai")
            result = chat_model.chat(message, **kwargs)
            
            generation_time = time.time() - start_time
            self._update_stats("chat", generation_time, "conversational_ai")
            
            result["metadata"] = {
                "message": message,
                "generation_time": generation_time,
                "model": "conversational_ai"
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            raise
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search academic papers"""
        try:
            research_model = self.model_manager.load_model("research_ai")
            return research_model.search_papers(query, max_results)
        except Exception as e:
            self.logger.error(f"Paper search failed: {e}")
            raise
    
    def analyze_paper(self, paper_id: str) -> Dict[str, Any]:
        """Analyze specific academic paper"""
        try:
            research_model = self.model_manager.load_model("research_ai")
            return research_model.analyze_paper(paper_id)
        except Exception as e:
            self.logger.error(f"Paper analysis failed: {e}")
            raise
    
    def web_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform web search"""
        try:
            research_model = self.model_manager.load_model("research_ai")
            return research_model.web_search(query, max_results)
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            raise
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            chat_model = self.model_manager.load_model("conversational_ai")
            return chat_model.get_conversation_history(conversation_id)
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def set_personality(self, traits: Dict[str, float]):
        """Set conversational AI personality"""
        try:
            chat_model = self.model_manager.load_model("conversational_ai")
            chat_model.set_personality(traits)
        except Exception as e:
            self.logger.error(f"Failed to set personality: {e}")
    
    def universal_generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Universal generation across all modalities"""
        try:
            tera_model = self.model_manager.load_model("tera_scale_ai")
            return tera_model.universal_generate(prompt, **kwargs)
        except Exception as e:
            self.logger.error(f"Universal generation failed: {e}")
            raise
    
    def process_any_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Process any file type automatically"""
        try:
            processor = self.model_manager.load_model("universal_processor")
            return processor.process_file(file_path, **kwargs)
        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        self.model_manager.unload_all()
        self.executor.shutdown(wait=True)
        self.logger.info("Multimodal AI system cleaned up")