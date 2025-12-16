import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any, Tuple
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import torchvision.transforms as transforms
from .base_model import BaseMultimodalModel
from config import config

class AdvancedEnhancementModel(BaseMultimodalModel):
    """Advanced Image and Video Enhancement with multiple model support"""
    
    def __init__(self, model_name: str = "realesrgan", device: str = "cuda"):
        super().__init__(model_name, device)
        self.upsampler = None
        self.face_enhancer = None
        self.video_enhancer = None
        self.scale_factor = 4
        
    def load_model(self):
        """Load enhancement models"""
        if self.is_loaded:
            return
            
        try:
            if self.model_name == "realesrgan":
                self._load_realesrgan()
            elif self.model_name == "gfpgan":
                self._load_gfpgan()
            elif self.model_name == "swinir":
                self._load_swinir()
            elif self.model_name == "basicvsr":
                self._load_basicvsr()
            
            self.is_loaded = True
            self.logger.info(f"Loaded {self.model_name} enhancement model")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_realesrgan(self):
        """Load RealESRGAN model"""
        model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=self.device
        )
        
        # Also load GFPGAN for face enhancement
        gfpgan_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
        self.face_enhancer = GFPGANer(
            model_path=gfpgan_path,
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
            device=self.device
        )
    
    def _load_gfpgan(self):
        """Load GFPGAN model"""
        model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
        
        self.face_enhancer = GFPGANer(
            model_path=model_path,
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            device=self.device
        )
    
    def _load_swinir(self):
        """Load SwinIR model"""
        from basicsr.archs.swinir_arch import SwinIR
        
        model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        ).to(self.device)
        
        # Load pretrained weights
        model_path = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
        loadnet = torch.load(model_path, map_location=self.device)
        model.load_state_dict(loadnet['params'], strict=True)
        model.eval()
        
        self.upsampler = model
    
    def _load_basicvsr(self):
        """Load BasicVSR++ for video enhancement"""
        from basicsr.archs.basicvsr_arch import BasicVSRPlusPlus
        
        model = BasicVSRPlusPlus(
            num_feat=64,
            num_block=7,
            max_residue_magnitude=10,
            is_low_res_input=True,
            spynet_path=None
        ).to(self.device)
        
        # Load pretrained weights
        model_path = "https://github.com/xinntao/BasicSR/releases/download/v1.4.0/BasicVSRPlusPlus_REDS4.pth"
        loadnet = torch.load(model_path, map_location=self.device)
        model.load_state_dict(loadnet['params'], strict=True)
        model.eval()
        
        self.video_enhancer = model
    
    def unload_model(self):
        """Unload model from memory"""
        if not self.is_loaded:
            return
            
        del self.upsampler
        if self.face_enhancer:
            del self.face_enhancer
        if self.video_enhancer:
            del self.video_enhancer
            
        self.upsampler = None
        self.face_enhancer = None
        self.video_enhancer = None
        self.is_loaded = False
        self.optimize_memory()
    
    def enhance_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        scale: int = 4,
        face_enhance: bool = True,
        tile_size: int = 512,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhance single image"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Process input image
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray) and image.shape[2] == 3:
            # Assume RGB, convert to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        original_shape = image.shape
        
        try:
            if self.model_name == "realesrgan" and self.upsampler:
                # Use RealESRGAN
                if face_enhance and self.face_enhancer:
                    # Use GFPGAN for face enhancement
                    _, _, enhanced = self.face_enhancer.enhance(
                        image, has_aligned=False, only_center_face=False, paste_back=True
                    )
                else:
                    # Standard upscaling
                    enhanced, _ = self.upsampler.enhance(image, outscale=scale)
                    
            elif self.model_name == "gfpgan" and self.face_enhancer:
                # Use GFPGAN
                _, _, enhanced = self.face_enhancer.enhance(
                    image, has_aligned=False, only_center_face=False, paste_back=True
                )
                
            elif self.model_name == "swinir" and self.upsampler:
                # Use SwinIR
                enhanced = self._enhance_with_swinir(image, scale)
                
            else:
                raise ValueError(f"Model {self.model_name} not properly loaded")
            
            # Convert back to RGB
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            # Create metadata
            metadata = {
                "original_shape": original_shape,
                "enhanced_shape": enhanced_rgb.shape,
                "scale_factor": enhanced_rgb.shape[0] / original_shape[0],
                "model": self.model_name,
                "face_enhance": face_enhance
            }
            
            self.optimize_memory()
            return enhanced_rgb, metadata
            
        except Exception as e:
            self.logger.error(f"Enhancement failed: {e}")
            raise
    
    def _enhance_with_swinir(self, image: np.ndarray, scale: int) -> np.ndarray:
        """Enhance image using SwinIR"""
        
        # Prepare image tensor
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Pad image to multiple of window size
        h, w = img_tensor.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')
        
        # Enhance
        with torch.no_grad():
            enhanced_tensor = self.upsampler(img_tensor)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            enhanced_tensor = enhanced_tensor[:, :, :h*scale, :w*scale]
        
        # Convert back to numpy
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255.0).clip(0, 255).astype(np.uint8)
        
        return enhanced
    
    def enhance_video(
        self,
        video_frames: List[Union[Image.Image, np.ndarray]],
        scale: int = 4,
        temporal_consistency: bool = True,
        batch_size: int = 4,
        **kwargs
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Enhance video frames"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Convert frames to numpy arrays
        frames = []
        for frame in video_frames:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            if frame.shape[2] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        
        enhanced_frames = []
        
        if self.video_enhancer and temporal_consistency:
            # Use video-specific enhancement
            enhanced_frames = self._enhance_video_temporal(frames, scale, batch_size)
        else:
            # Enhance frame by frame
            for i, frame in enumerate(frames):
                try:
                    enhanced, _ = self.enhance_image(frame, scale=scale, **kwargs)
                    enhanced_frames.append(enhanced)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Enhanced {i + 1}/{len(frames)} frames")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to enhance frame {i}: {e}")
                    # Use original frame as fallback
                    enhanced_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        metadata = {
            "num_frames": len(enhanced_frames),
            "original_resolution": frames[0].shape[:2] if frames else None,
            "enhanced_resolution": enhanced_frames[0].shape[:2] if enhanced_frames else None,
            "scale_factor": scale,
            "model": self.model_name,
            "temporal_consistency": temporal_consistency
        }
        
        self.optimize_memory()
        return enhanced_frames, metadata
    
    def _enhance_video_temporal(
        self,
        frames: List[np.ndarray],
        scale: int,
        batch_size: int
    ) -> List[np.ndarray]:
        """Enhance video with temporal consistency using BasicVSR++"""
        
        if not self.video_enhancer:
            raise RuntimeError("Video enhancer not loaded")
        
        # Prepare video tensor
        video_tensor = []
        for frame in frames:
            # Convert BGR to RGB and normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
            video_tensor.append(frame_tensor)
        
        video_tensor = torch.stack(video_tensor).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        
        # Process in batches
        enhanced_frames = []
        num_batches = (len(frames) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(frames))
            
            batch_tensor = video_tensor[:, start_idx:end_idx]
            
            with torch.no_grad():
                enhanced_batch = self.video_enhancer(batch_tensor)
            
            # Convert back to numpy
            for j in range(enhanced_batch.shape[1]):
                enhanced_frame = enhanced_batch[0, j].permute(1, 2, 0).cpu().numpy()
                enhanced_frame = (enhanced_frame * 255.0).clip(0, 255).astype(np.uint8)
                enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def batch_enhance_images(
        self,
        image_paths: List[str],
        output_dir: str,
        scale: int = 4,
        face_enhance: bool = True,
        **kwargs
    ) -> List[str]:
        """Batch enhance multiple images"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        enhanced_paths = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load and enhance image
                enhanced, metadata = self.enhance_image(
                    image_path, scale=scale, face_enhance=face_enhance, **kwargs
                )
                
                # Save enhanced image
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
                
                enhanced_pil = Image.fromarray(enhanced)
                enhanced_pil.save(output_path, quality=95)
                enhanced_paths.append(output_path)
                
                self.logger.info(f"Enhanced {i + 1}/{len(image_paths)}: {output_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to enhance {image_path}: {e}")
        
        return enhanced_paths
    
    def compare_enhancement(
        self,
        original: Union[Image.Image, np.ndarray],
        enhanced: Union[Image.Image, np.ndarray]
    ) -> Dict[str, float]:
        """Compare original and enhanced images using quality metrics"""
        
        # Convert to numpy arrays
        if isinstance(original, Image.Image):
            original = np.array(original)
        if isinstance(enhanced, Image.Image):
            enhanced = np.array(enhanced)
        
        # Resize enhanced to match original for comparison
        if original.shape != enhanced.shape:
            scale_factor = enhanced.shape[0] // original.shape[0]
            original_upscaled = cv2.resize(
                original, 
                (enhanced.shape[1], enhanced.shape[0]), 
                interpolation=cv2.INTER_CUBIC
            )
        else:
            original_upscaled = original
            scale_factor = 1
        
        # Calculate metrics
        metrics = {}
        
        # PSNR
        mse = np.mean((original_upscaled.astype(float) - enhanced.astype(float)) ** 2)
        if mse == 0:
            metrics['psnr'] = float('inf')
        else:
            metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # SSIM (simplified version)
        def ssim_simple(img1, img2):
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return ssim
        
        # Calculate SSIM for each channel
        if len(enhanced.shape) == 3:
            ssim_values = []
            for c in range(enhanced.shape[2]):
                ssim_c = ssim_simple(
                    original_upscaled[:, :, c].astype(float),
                    enhanced[:, :, c].astype(float)
                )
                ssim_values.append(ssim_c)
            metrics['ssim'] = np.mean(ssim_values)
        else:
            metrics['ssim'] = ssim_simple(
                original_upscaled.astype(float),
                enhanced.astype(float)
            )
        
        # Sharpness metric (Laplacian variance)
        def sharpness(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        
        metrics['original_sharpness'] = sharpness(original)
        metrics['enhanced_sharpness'] = sharpness(enhanced)
        metrics['sharpness_improvement'] = metrics['enhanced_sharpness'] / metrics['original_sharpness']
        
        metrics['scale_factor'] = scale_factor
        
        return metrics