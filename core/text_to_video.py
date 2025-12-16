import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any, Tuple
import cv2
import imageio
from diffusers import (
    TextToVideoSDPipeline,
    VideoToVideoSDPipeline,
    DiffusionPipeline,
    AnimateDiffPipeline,
    MotionAdapter
)
from transformers import CLIPTextModel, CLIPTokenizer
import decord
from .base_model import BaseMultimodalModel
from config import config

class AdvancedTextToVideoModel(BaseMultimodalModel):
    """Advanced Text-to-Video generation with multiple model support"""
    
    def __init__(self, model_name: str = "zeroscope", device: str = "cuda"):
        super().__init__(model_name, device)
        self.pipeline = None
        self.motion_adapter = None
        self.upscaler = None
        
    def load_model(self):
        """Load the text-to-video model"""
        if self.is_loaded:
            return
            
        model_id = config.text_to_video_models[self.model_name]
        
        try:
            if self.model_name == "zeroscope":
                # Load ZeroScope model
                self.pipeline = TextToVideoSDPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                ).to(self.device)
                
                # Load upscaler
                upscaler_id = "cerspense/zeroscope_v2_XL"
                self.upscaler = VideoToVideoSDPipeline.from_pretrained(
                    upscaler_id,
                    torch_dtype=torch.float16
                ).to(self.device)
                
            elif self.model_name == "modelscope":
                # Load ModelScope T2V
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
                
            elif self.model_name == "animatediff":
                # Load AnimateDiff
                adapter_id = config.text_to_video_models["animatediff"]
                self.motion_adapter = MotionAdapter.from_pretrained(
                    adapter_id,
                    torch_dtype=torch.float16
                )
                
                # Use with Stable Diffusion base
                base_model = "runwayml/stable-diffusion-v1-5"
                self.pipeline = AnimateDiffPipeline.from_pretrained(
                    base_model,
                    motion_adapter=self.motion_adapter,
                    torch_dtype=torch.float16
                ).to(self.device)
                
            elif self.model_name == "cogvideo":
                # Load CogVideoX
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
                
            else:
                # Generic video pipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
            
            # Optimize pipeline
            self._optimize_pipeline()
            
            self.is_loaded = True
            self.logger.info(f"Loaded {self.model_name} text-to-video model")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _optimize_pipeline(self):
        """Optimize pipeline for performance"""
        if config.enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                if self.upscaler:
                    self.upscaler.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        if config.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
            if self.upscaler:
                self.upscaler.enable_model_cpu_offload()
        
        if config.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
            if self.upscaler:
                self.upscaler.enable_attention_slicing()
        
        if config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
            if self.upscaler:
                self.upscaler.enable_vae_slicing()
    
    def unload_model(self):
        """Unload model from memory"""
        if not self.is_loaded:
            return
            
        del self.pipeline
        if self.upscaler:
            del self.upscaler
        if self.motion_adapter:
            del self.motion_adapter
            
        self.pipeline = None
        self.upscaler = None
        self.motion_adapter = None
        self.is_loaded = False
        self.optimize_memory()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 576,
        height: int = 320,
        num_frames: int = 24,
        fps: int = 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        seed: Optional[int] = None,
        upscale: bool = True,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Generate video from text prompt"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Prepare generation parameters
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **kwargs
        }
        
        # Model-specific parameters
        if self.model_name == "zeroscope":
            gen_kwargs.update({
                "height": height,
                "width": width,
                "num_frames": num_frames
            })
        elif self.model_name == "modelscope":
            gen_kwargs.update({
                "num_frames": num_frames,
                "height": height,
                "width": width
            })
        elif self.model_name == "animatediff":
            gen_kwargs.update({
                "height": height,
                "width": width,
                "num_frames": num_frames
            })
        elif self.model_name == "cogvideo":
            gen_kwargs.update({
                "num_frames": num_frames,
                "height": height,
                "width": width
            })
        
        # Generate base video
        with torch.autocast(self.device):
            result = self.pipeline(**gen_kwargs)
        
        if hasattr(result, 'frames'):
            frames = result.frames[0]
        else:
            frames = result.images
        
        # Convert to numpy arrays
        if isinstance(frames[0], Image.Image):
            frames = [np.array(frame) for frame in frames]
        
        # Upscale if available and requested
        if self.upscaler and upscale and self.model_name == "zeroscope":
            upscaled_frames = self._upscale_video(
                frames, prompt, negative_prompt, generator
            )
            frames = upscaled_frames
        
        # Create metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": frames[0].shape[1] if frames else width,
            "height": frames[0].shape[0] if frames else height,
            "num_frames": len(frames),
            "fps": fps,
            "model": self.model_name,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        }
        
        self.optimize_memory()
        return frames, metadata
    
    def _upscale_video(
        self,
        frames: List[np.ndarray],
        prompt: str,
        negative_prompt: Optional[str],
        generator: Optional[torch.Generator]
    ) -> List[np.ndarray]:
        """Upscale video using video-to-video pipeline"""
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        with torch.autocast(self.device):
            upscaled_result = self.upscaler(
                prompt=prompt,
                negative_prompt=negative_prompt,
                video=pil_frames,
                strength=0.6,
                generator=generator
            )
        
        if hasattr(upscaled_result, 'frames'):
            upscaled_frames = upscaled_result.frames[0]
        else:
            upscaled_frames = upscaled_result.images
        
        # Convert back to numpy
        return [np.array(frame) for frame in upscaled_frames]
    
    def video_to_video(
        self,
        prompt: str,
        video_frames: List[Union[Image.Image, np.ndarray]],
        strength: float = 0.8,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Video-to-video generation"""
        
        if not self.upscaler:
            raise NotImplementedError("Video-to-video not supported for this model")
        
        # Convert frames to PIL if needed
        if isinstance(video_frames[0], np.ndarray):
            video_frames = [Image.fromarray(frame) for frame in video_frames]
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        with torch.autocast(self.device):
            result = self.upscaler(
                prompt=prompt,
                negative_prompt=negative_prompt,
                video=video_frames,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs
            )
        
        if hasattr(result, 'frames'):
            frames = result.frames[0]
        else:
            frames = result.images
        
        # Convert to numpy arrays
        frames = [np.array(frame) for frame in frames]
        
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": frames[0].shape[1],
            "height": frames[0].shape[0],
            "num_frames": len(frames),
            "model": self.model_name,
            "seed": seed,
            "strength": strength
        }
        
        self.optimize_memory()
        return frames, metadata
    
    def interpolate_frames(
        self,
        start_frame: Union[Image.Image, np.ndarray],
        end_frame: Union[Image.Image, np.ndarray],
        num_interpolation_frames: int = 8,
        prompt: Optional[str] = None
    ) -> List[np.ndarray]:
        """Interpolate between two frames"""
        
        # Convert to PIL if needed
        if isinstance(start_frame, np.ndarray):
            start_frame = Image.fromarray(start_frame)
        if isinstance(end_frame, np.ndarray):
            end_frame = Image.fromarray(end_frame)
        
        # Create interpolation frames using the model
        frames = [start_frame]
        
        for i in range(1, num_interpolation_frames + 1):
            alpha = i / (num_interpolation_frames + 1)
            
            # Simple linear interpolation for now
            # In a real implementation, you'd use the model for better interpolation
            start_array = np.array(start_frame)
            end_array = np.array(end_frame)
            interpolated = (1 - alpha) * start_array + alpha * end_array
            frames.append(Image.fromarray(interpolated.astype(np.uint8)))
        
        frames.append(end_frame)
        
        return [np.array(frame) for frame in frames]
    
    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 24,
        codec: str = "libx264",
        quality: str = "high"
    ):
        """Save video frames to file"""
        
        # Quality settings
        quality_settings = {
            "low": {"crf": 28, "preset": "fast"},
            "medium": {"crf": 23, "preset": "medium"},
            "high": {"crf": 18, "preset": "slow"},
            "lossless": {"crf": 0, "preset": "veryslow"}
        }
        
        settings = quality_settings.get(quality, quality_settings["high"])
        
        # Use imageio for video writing
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            ffmpeg_params=[
                "-crf", str(settings["crf"]),
                "-preset", settings["preset"],
                "-pix_fmt", "yuv420p"
            ]
        )
        
        for frame in frames:
            writer.append_data(frame)
        
        writer.close()
        
        self.logger.info(f"Video saved to {output_path}")
    
    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Extract frames from video file"""
        
        # Use decord for efficient video reading
        vr = decord.VideoReader(video_path)
        
        if max_frames:
            indices = np.linspace(0, len(vr) - 1, max_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
        else:
            frames = vr[:].asnumpy()
        
        return [frame for frame in frames]