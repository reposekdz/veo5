import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any, Tuple
import cv2
from diffusers import (
    StableVideoDiffusionPipeline,
    I2VGenXLPipeline,
    DiffusionPipeline
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torchvision.transforms as transforms
from .base_model import BaseMultimodalModel
from config import config

class AdvancedImageToVideoModel(BaseMultimodalModel):
    """Advanced Image-to-Video generation with multiple model support"""
    
    def __init__(self, model_name: str = "stable_video", device: str = "cuda"):
        super().__init__(model_name, device)
        self.pipeline = None
        self.image_encoder = None
        self.feature_extractor = None
        
    def load_model(self):
        """Load the image-to-video model"""
        if self.is_loaded:
            return
            
        model_id = config.image_to_video_models[self.model_name]
        
        try:
            if self.model_name == "stable_video":
                # Load Stable Video Diffusion
                self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to(self.device)
                
            elif self.model_name == "i2vgen":
                # Load I2VGen-XL
                self.pipeline = I2VGenXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to(self.device)
                
            elif self.model_name == "dynamicrafter":
                # Load DynamiCrafter
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
                
            elif self.model_name == "consisti2v":
                # Load ConsistI2V
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
                
            else:
                # Generic image-to-video pipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
            
            # Load additional components
            self._load_additional_components()
            
            # Optimize pipeline
            self._optimize_pipeline()
            
            self.is_loaded = True
            self.logger.info(f"Loaded {self.model_name} image-to-video model")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_additional_components(self):
        """Load additional model components"""
        try:
            # Load CLIP vision encoder for better image understanding
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            ).to(self.device)
            
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load additional components: {e}")
    
    def _optimize_pipeline(self):
        """Optimize pipeline for performance"""
        if config.enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        if config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        
        if config.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
    
    def unload_model(self):
        """Unload model from memory"""
        if not self.is_loaded:
            return
            
        del self.pipeline
        if self.image_encoder:
            del self.image_encoder
        if self.feature_extractor:
            del self.feature_extractor
            
        self.pipeline = None
        self.image_encoder = None
        self.feature_extractor = None
        self.is_loaded = False
        self.optimize_memory()
    
    def generate(
        self,
        image: Union[Image.Image, np.ndarray, str],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 576,
        num_frames: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        decode_chunk_size: int = 8,
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Generate video from input image"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Process input image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image to target dimensions
        image = image.resize((width, height), Image.LANCZOS)
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Prepare generation parameters based on model
        gen_kwargs = {
            "image": image,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "decode_chunk_size": decode_chunk_size,
            **kwargs
        }
        
        if self.model_name == "stable_video":
            gen_kwargs.update({
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "motion_bucket_id": motion_bucket_id,
                "noise_aug_strength": noise_aug_strength,
                "fps": fps
            })
        elif self.model_name == "i2vgen":
            gen_kwargs.update({
                "prompt": prompt or "high quality video",
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale
            })
        elif self.model_name in ["dynamicrafter", "consisti2v"]:
            gen_kwargs.update({
                "prompt": prompt or "high quality video",
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale
            })
        
        # Generate video
        with torch.autocast(self.device):
            result = self.pipeline(**gen_kwargs)
        
        # Extract frames
        if hasattr(result, 'frames'):
            frames = result.frames[0]
        else:
            frames = result.images
        
        # Convert to numpy arrays
        if isinstance(frames[0], Image.Image):
            frames = [np.array(frame) for frame in frames]
        
        # Create metadata
        metadata = {
            "input_image_size": image.size,
            "output_size": (width, height),
            "num_frames": len(frames),
            "fps": fps,
            "model": self.model_name,
            "seed": seed,
            "motion_bucket_id": motion_bucket_id,
            "noise_aug_strength": noise_aug_strength,
            "num_inference_steps": num_inference_steps,
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }
        
        self.optimize_memory()
        return frames, metadata
    
    def animate_with_motion(
        self,
        image: Union[Image.Image, np.ndarray],
        motion_vectors: Optional[np.ndarray] = None,
        motion_strength: float = 1.0,
        num_frames: int = 16,
        **kwargs
    ) -> List[np.ndarray]:
        """Animate image with specific motion patterns"""
        
        if motion_vectors is None:
            # Generate default motion (zoom + pan)
            motion_vectors = self._generate_default_motion(num_frames)
        
        # Apply motion-guided generation
        frames = []
        base_frame = np.array(image) if isinstance(image, Image.Image) else image
        
        for i, motion in enumerate(motion_vectors):
            # Apply motion transformation
            transformed = self._apply_motion_transform(
                base_frame, motion, motion_strength
            )
            frames.append(transformed)
        
        return frames
    
    def _generate_default_motion(self, num_frames: int) -> np.ndarray:
        """Generate default motion vectors"""
        # Create smooth motion patterns
        t = np.linspace(0, 2 * np.pi, num_frames)
        
        # Combine different motion types
        zoom = 1.0 + 0.1 * np.sin(t)
        pan_x = 10 * np.sin(t * 0.5)
        pan_y = 5 * np.cos(t * 0.3)
        rotation = 2 * np.sin(t * 0.2)
        
        motion_vectors = np.stack([zoom, pan_x, pan_y, rotation], axis=1)
        return motion_vectors
    
    def _apply_motion_transform(
        self,
        frame: np.ndarray,
        motion: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply motion transformation to frame"""
        
        zoom, pan_x, pan_y, rotation = motion * strength
        h, w = frame.shape[:2]
        
        # Create transformation matrix
        center = (w // 2, h // 2)
        
        # Rotation and zoom
        M_rot = cv2.getRotationMatrix2D(center, rotation, zoom)
        
        # Translation
        M_rot[0, 2] += pan_x
        M_rot[1, 2] += pan_y
        
        # Apply transformation
        transformed = cv2.warpAffine(
            frame, M_rot, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return transformed
    
    def interpolate_between_images(
        self,
        start_image: Union[Image.Image, np.ndarray],
        end_image: Union[Image.Image, np.ndarray],
        num_frames: int = 16,
        interpolation_method: str = "morphing",
        **kwargs
    ) -> List[np.ndarray]:
        """Create smooth transition between two images"""
        
        # Convert to numpy arrays
        if isinstance(start_image, Image.Image):
            start_array = np.array(start_image)
        else:
            start_array = start_image
            
        if isinstance(end_image, Image.Image):
            end_array = np.array(end_image)
        else:
            end_array = end_image
        
        # Ensure same dimensions
        if start_array.shape != end_array.shape:
            h, w = min(start_array.shape[0], end_array.shape[0]), min(start_array.shape[1], end_array.shape[1])
            start_array = cv2.resize(start_array, (w, h))
            end_array = cv2.resize(end_array, (w, h))
        
        frames = []
        
        if interpolation_method == "linear":
            # Simple linear interpolation
            for i in range(num_frames):
                alpha = i / (num_frames - 1)
                interpolated = (1 - alpha) * start_array + alpha * end_array
                frames.append(interpolated.astype(np.uint8))
                
        elif interpolation_method == "morphing":
            # Optical flow-based morphing
            frames = self._optical_flow_morphing(
                start_array, end_array, num_frames
            )
            
        elif interpolation_method == "ai_guided":
            # Use the model for AI-guided interpolation
            frames = self._ai_guided_interpolation(
                start_array, end_array, num_frames, **kwargs
            )
        
        return frames
    
    def _optical_flow_morphing(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_frames: int
    ) -> List[np.ndarray]:
        """Create morphing using optical flow"""
        
        # Convert to grayscale for flow calculation
        gray1 = cv2.cvtColor(start_frame, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(end_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, None, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )[0]
        
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            
            # Interpolate flow
            intermediate_flow = flow * alpha
            
            # Warp start frame towards end frame
            h, w = start_frame.shape[:2]
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + intermediate_flow[:, :, 0]).astype(np.float32)
            map_y = (map_y + intermediate_flow[:, :, 1]).astype(np.float32)
            
            warped = cv2.remap(start_frame, map_x, map_y, cv2.INTER_LINEAR)
            
            # Blend with linear interpolation
            blended = (1 - alpha) * warped + alpha * end_frame
            frames.append(blended.astype(np.uint8))
        
        return frames
    
    def _ai_guided_interpolation(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_frames: int,
        **kwargs
    ) -> List[np.ndarray]:
        """AI-guided interpolation using the loaded model"""
        
        # Create intermediate images using the model
        frames = []
        
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            
            # Simple blend as fallback
            blended = (1 - alpha) * start_frame + alpha * end_frame
            frames.append(blended.astype(np.uint8))
        
        # In a real implementation, you would use the model to generate
        # more sophisticated interpolations
        
        return frames
    
    def extract_keyframes(
        self,
        video_frames: List[np.ndarray],
        num_keyframes: int = 5,
        method: str = "uniform"
    ) -> List[Tuple[int, np.ndarray]]:
        """Extract keyframes from video"""
        
        if method == "uniform":
            # Uniform sampling
            indices = np.linspace(0, len(video_frames) - 1, num_keyframes, dtype=int)
            keyframes = [(i, video_frames[i]) for i in indices]
            
        elif method == "difference":
            # Based on frame differences
            differences = []
            for i in range(1, len(video_frames)):
                diff = np.mean(np.abs(
                    video_frames[i].astype(float) - video_frames[i-1].astype(float)
                ))
                differences.append((i, diff))
            
            # Sort by difference and take top frames
            differences.sort(key=lambda x: x[1], reverse=True)
            indices = [0] + [x[0] for x in differences[:num_keyframes-1]]
            indices.sort()
            keyframes = [(i, video_frames[i]) for i in indices]
            
        return keyframes
    
    def enhance_temporal_consistency(
        self,
        frames: List[np.ndarray],
        consistency_strength: float = 0.5
    ) -> List[np.ndarray]:
        """Enhance temporal consistency between frames"""
        
        if len(frames) < 2:
            return frames
        
        enhanced_frames = [frames[0]]  # Keep first frame as is
        
        for i in range(1, len(frames)):
            current_frame = frames[i].astype(float)
            prev_frame = enhanced_frames[-1].astype(float)
            
            # Apply temporal smoothing
            smoothed = (1 - consistency_strength) * current_frame + \
                      consistency_strength * prev_frame
            
            enhanced_frames.append(smoothed.astype(np.uint8))
        
        return enhanced_frames