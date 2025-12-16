import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class ModelConfig:
    """Configuration for multimodal AI models"""
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_models: bool = True
    
    # Text-to-Image Models
    text_to_image_models: Dict[str, str] = None
    
    # Text-to-Video Models  
    text_to_video_models: Dict[str, str] = None
    
    # Image-to-Video Models
    image_to_video_models: Dict[str, str] = None
    
    # Image Enhancement Models
    image_enhancement_models: Dict[str, str] = None
    
    # Video Enhancement Models
    video_enhancement_models: Dict[str, str] = None
    
    # ControlNet Models
    controlnet_models: Dict[str, str] = None
    
    # Face Models
    face_models: Dict[str, str] = None
    
    # Generation Parameters
    default_image_size: Tuple[int, int] = (1024, 1024)
    default_video_size: Tuple[int, int] = (1024, 576)
    default_video_fps: int = 24
    default_video_frames: int = 120
    max_batch_size: int = 4
    
    # Quality Settings
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    eta: float = 0.0
    
    # Memory Optimization
    enable_cpu_offload: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_xformers: bool = True
    
    # Paths
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    dataset_dir: str = "./datasets"
    
    def __post_init__(self):
        # Initialize model configurations
        self.text_to_image_models = {
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "sdxl_refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
            "flux": "black-forest-labs/FLUX.1-dev",
            "playground": "playgroundai/playground-v2.5-1024px-aesthetic",
            "kandinsky": "kandinsky-community/kandinsky-3"
        }
        
        self.text_to_video_models = {
            "zeroscope": "cerspense/zeroscope_v2_576w",
            "modelscope": "damo-vilab/text-to-video-ms-1.7b",
            "lavie": "vdo/LaVie_base",
            "cogvideo": "THUDM/CogVideoX-2b",
            "animatediff": "guoyww/animatediff-motion-adapter-v1-5-2"
        }
        
        self.image_to_video_models = {
            "stable_video": "stabilityai/stable-video-diffusion-img2vid-xt",
            "i2vgen": "damo-vilab/i2vgen-xl",
            "dynamicrafter": "Doubiiu/DynamiCrafter_512_Interp",
            "consisti2v": "TIGER-Lab/ConsistI2V"
        }
        
        self.image_enhancement_models = {
            "realesrgan": "RealESRGAN/RealESRGAN_x4plus",
            "gfpgan": "TencentARC/GFPGAN",
            "codeformer": "sczhou/CodeFormer",
            "swinir": "JingyunLiang/SwinIR",
            "bsrgan": "cszn/BSRGAN"
        }
        
        self.video_enhancement_models = {
            "basicvsr": "xinntao/BasicVSR_PlusPlus",
            "real_basicvsr": "xinntao/RealBasicVSR",
            "edvr": "xinntao/EDVR",
            "tdan": "xinntao/TDAN"
        }
        
        self.controlnet_models = {
            "canny": "lllyasviel/sd-controlnet-canny",
            "depth": "lllyasviel/sd-controlnet-depth", 
            "pose": "lllyasviel/sd-controlnet-openpose",
            "scribble": "lllyasviel/sd-controlnet-scribble",
            "seg": "lllyasviel/sd-controlnet-seg",
            "normal": "lllyasviel/sd-controlnet-normal",
            "mlsd": "lllyasviel/sd-controlnet-mlsd",
            "hed": "lllyasviel/sd-controlnet-hed"
        }
        
        self.face_models = {
            "insightface": "buffalo_l",
            "face_parsing": "79999_iter.pth",
            "face_detection": "detection_Resnet50_Final.pth"
        }
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

# Global configuration instance
config = ModelConfig()