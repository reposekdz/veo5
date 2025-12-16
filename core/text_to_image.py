import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
    DiffusionPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL
)
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
import cv2
from controlnet_aux import (
    CannyDetector, OpenposeDetector, MidasDetector,
    HEDdetector, MLSDdetector, NormalBaeDetector
)
from .base_model import BaseMultimodalModel
from config import config

class AdvancedTextToImageModel(BaseMultimodalModel):
    """Advanced Text-to-Image generation with multiple model support"""
    
    def __init__(self, model_name: str = "sdxl", device: str = "cuda"):
        super().__init__(model_name, device)
        self.pipeline = None
        self.refiner = None
        self.controlnets = {}
        self.preprocessors = {}
        self.vae = None
        self.scheduler_configs = {
            "dpm": DPMSolverMultistepScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "kdpm2_a": KDPM2AncestralDiscreteScheduler
        }
        
    def load_model(self):
        """Load the text-to-image model"""
        if self.is_loaded:
            return
            
        model_id = config.text_to_image_models[self.model_name]
        
        try:
            # Load main pipeline
            if self.model_name == "sdxl":
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                ).to(self.device)
                
                # Load refiner
                refiner_id = config.text_to_image_models["sdxl_refiner"]
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                ).to(self.device)
                
            elif self.model_name == "sd3":
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                ).to(self.device)
                
            elif self.model_name == "flux":
                self.pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16
                ).to(self.device)
                
            else:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                ).to(self.device)
            
            # Optimize pipeline
            self._optimize_pipeline()
            
            # Load ControlNets
            self._load_controlnets()
            
            # Load preprocessors
            self._load_preprocessors()
            
            self.is_loaded = True
            self.logger.info(f"Loaded {self.model_name} text-to-image model")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _optimize_pipeline(self):
        """Optimize pipeline for performance"""
        if config.enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                if self.refiner:
                    self.refiner.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        if config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
            if self.refiner:
                self.refiner.enable_model_cpu_offload()
        
        if config.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
            if self.refiner:
                self.refiner.enable_attention_slicing()
        
        if config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
            if self.refiner:
                self.refiner.enable_vae_slicing()
        
        # Compile models for faster inference
        if config.compile_models:
            try:
                self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")
                if self.refiner:
                    self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead")
            except:
                pass
    
    def _load_controlnets(self):
        """Load ControlNet models"""
        for control_type, model_id in config.controlnet_models.items():
            try:
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                )
                self.controlnets[control_type] = controlnet
            except Exception as e:
                self.logger.warning(f"Failed to load ControlNet {control_type}: {e}")
    
    def _load_preprocessors(self):
        """Load image preprocessors for ControlNet"""
        try:
            self.preprocessors = {
                "canny": CannyDetector(),
                "openpose": OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
                "depth": MidasDetector.from_pretrained("lllyasviel/Annotators"),
                "hed": HEDdetector.from_pretrained("lllyasviel/Annotators"),
                "mlsd": MLSDdetector.from_pretrained("lllyasviel/Annotators"),
                "normal": NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
            }
        except Exception as e:
            self.logger.warning(f"Failed to load preprocessors: {e}")
    
    def unload_model(self):
        """Unload model from memory"""
        if not self.is_loaded:
            return
            
        del self.pipeline
        if self.refiner:
            del self.refiner
        for controlnet in self.controlnets.values():
            del controlnet
        
        self.pipeline = None
        self.refiner = None
        self.controlnets = {}
        self.is_loaded = False
        self.optimize_memory()
        
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        scheduler: str = "dpm",
        use_refiner: bool = True,
        refiner_strength: float = 0.3,
        control_image: Optional[Image.Image] = None,
        control_type: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompt"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Set scheduler
        if scheduler in self.scheduler_configs:
            self.pipeline.scheduler = self.scheduler_configs[scheduler].from_config(
                self.pipeline.scheduler.config
            )
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Prepare generation parameters
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "generator": generator,
            **kwargs
        }
        
        # Handle ControlNet
        if control_image and control_type and control_type in self.controlnets:
            # Preprocess control image
            if control_type in self.preprocessors:
                control_image = self.preprocessors[control_type](control_image)
            
            # Create ControlNet pipeline
            controlnet_pipeline = StableDiffusionXLControlNetPipeline(
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                text_encoder_2=self.pipeline.text_encoder_2,
                tokenizer=self.pipeline.tokenizer,
                tokenizer_2=self.pipeline.tokenizer_2,
                unet=self.pipeline.unet,
                controlnet=self.controlnets[control_type],
                scheduler=self.pipeline.scheduler
            ).to(self.device)
            
            gen_kwargs["image"] = control_image
            gen_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            
            # Generate with ControlNet
            with torch.autocast(self.device):
                result = controlnet_pipeline(**gen_kwargs)
            
            del controlnet_pipeline
        else:
            # Standard generation
            with torch.autocast(self.device):
                result = self.pipeline(**gen_kwargs)
        
        images = result.images
        
        # Apply refiner if available and requested
        if self.refiner and use_refiner and self.model_name == "sdxl":
            refined_images = []
            for img in images:
                with torch.autocast(self.device):
                    refined = self.refiner(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=img,
                        strength=refiner_strength,
                        num_inference_steps=num_inference_steps // 2,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]
                refined_images.append(refined)
            images = refined_images
        
        self.optimize_memory()
        return images
    
    def img2img(
        self,
        prompt: str,
        image: Image.Image,
        strength: float = 0.8,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Image-to-image generation"""
        
        if not hasattr(self.pipeline, 'img2img') and not self.refiner:
            raise NotImplementedError("Image-to-image not supported for this model")
        
        # Use refiner for img2img if available
        pipeline = self.refiner if self.refiner else self.pipeline
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        with torch.autocast(self.device):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            )
        
        self.optimize_memory()
        return result.images
    
    def inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Inpainting generation"""
        
        # Load inpainting pipeline if needed
        from diffusers import StableDiffusionXLInpaintPipeline
        
        inpaint_pipeline = StableDiffusionXLInpaintPipeline(
            vae=self.pipeline.vae,
            text_encoder=self.pipeline.text_encoder,
            text_encoder_2=self.pipeline.text_encoder_2,
            tokenizer=self.pipeline.tokenizer,
            tokenizer_2=self.pipeline.tokenizer_2,
            unet=self.pipeline.unet,
            scheduler=self.pipeline.scheduler
        ).to(self.device)
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        with torch.autocast(self.device):
            result = inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            )
        
        del inpaint_pipeline
        self.optimize_memory()
        return result.images