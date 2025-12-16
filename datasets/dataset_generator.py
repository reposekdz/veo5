import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import Dataset as HFDataset, load_dataset
import requests
from io import BytesIO
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class MultimodalDatasetGenerator:
    """Advanced dataset generator for multimodal AI training"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset configurations
        self.image_sizes = [(512, 512), (1024, 1024), (768, 768)]
        self.video_sizes = [(256, 256), (512, 512), (768, 432)]
        self.video_lengths = [16, 24, 32, 48]
        
        # Prompt templates for different categories
        self.prompt_templates = {
            "portrait": [
                "a portrait of {subject} {style}",
                "headshot of {subject} {lighting}",
                "{subject} portrait {background}",
                "professional photo of {subject} {mood}"
            ],
            "landscape": [
                "a {time} view of {location} {weather}",
                "{location} landscape {season}",
                "scenic {location} {atmosphere}",
                "panoramic view of {location} {lighting}"
            ],
            "object": [
                "a {adjective} {object} {context}",
                "{object} on {surface} {lighting}",
                "detailed view of {object} {style}",
                "{object} in {environment} {mood}"
            ],
            "abstract": [
                "{color} abstract {pattern} {texture}",
                "geometric {shape} composition {style}",
                "flowing {element} {color} {mood}",
                "minimalist {concept} {atmosphere}"
            ],
            "action": [
                "{subject} {action} {location} {time}",
                "dynamic scene of {subject} {action}",
                "{subject} performing {action} {style}",
                "motion blur of {subject} {action}"
            ]
        }
        
        # Style modifiers
        self.styles = [
            "photorealistic", "cinematic", "artistic", "vintage", "modern",
            "dramatic", "soft", "vibrant", "muted", "high contrast",
            "film noir", "cyberpunk", "fantasy", "sci-fi", "steampunk"
        ]
        
        # Quality descriptors
        self.quality_terms = [
            "high quality", "ultra detailed", "sharp focus", "professional",
            "masterpiece", "award winning", "stunning", "breathtaking",
            "highly detailed", "photographic", "realistic", "lifelike"
        ]
    
    def generate_text_image_pairs(
        self,
        num_samples: int = 10000,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate text-image training pairs"""
        
        if categories is None:
            categories = list(self.prompt_templates.keys())
        
        samples = []
        
        for i in range(num_samples):
            category = random.choice(categories)
            template = random.choice(self.prompt_templates[category])
            
            # Generate prompt based on category
            if category == "portrait":
                prompt = self._generate_portrait_prompt(template)
            elif category == "landscape":
                prompt = self._generate_landscape_prompt(template)
            elif category == "object":
                prompt = self._generate_object_prompt(template)
            elif category == "abstract":
                prompt = self._generate_abstract_prompt(template)
            elif category == "action":
                prompt = self._generate_action_prompt(template)
            
            # Add quality and style modifiers
            style = random.choice(self.styles)
            quality = random.choice(self.quality_terms)
            
            full_prompt = f"{prompt}, {style}, {quality}"
            
            # Generate negative prompt
            negative_prompt = self._generate_negative_prompt()
            
            # Create sample metadata
            sample = {
                "id": f"sample_{i:06d}",
                "prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "category": category,
                "style": style,
                "quality": quality,
                "image_size": random.choice(self.image_sizes),
                "guidance_scale": random.uniform(5.0, 15.0),
                "num_inference_steps": random.choice([20, 30, 50, 75, 100])
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_text_video_pairs(
        self,
        num_samples: int = 5000,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate text-video training pairs"""
        
        if categories is None:
            categories = ["action", "landscape", "object"]
        
        samples = []
        
        # Video-specific prompt templates
        video_templates = {
            "action": [
                "{subject} {action} in {location}, {camera_movement}",
                "time-lapse of {subject} {action}",
                "slow motion {subject} {action}",
                "{subject} {action} with {effect}"
            ],
            "landscape": [
                "flowing {element} through {location}",
                "clouds moving over {location}",
                "waves crashing on {location}",
                "wind blowing through {location}"
            ],
            "object": [
                "{object} rotating slowly",
                "{object} transforming into {other_object}",
                "close-up of {object} {action}",
                "{object} in motion {environment}"
            ]
        }
        
        for i in range(num_samples):
            category = random.choice(categories)
            template = random.choice(video_templates[category])
            
            # Generate video-specific prompt
            if category == "action":
                prompt = self._generate_video_action_prompt(template)
            elif category == "landscape":
                prompt = self._generate_video_landscape_prompt(template)
            elif category == "object":
                prompt = self._generate_video_object_prompt(template)
            
            # Add video quality terms
            video_quality = random.choice([
                "smooth motion", "fluid animation", "cinematic",
                "high frame rate", "stable camera", "professional video"
            ])
            
            full_prompt = f"{prompt}, {video_quality}"
            negative_prompt = self._generate_video_negative_prompt()
            
            sample = {
                "id": f"video_{i:06d}",
                "prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "category": category,
                "video_size": random.choice(self.video_sizes),
                "num_frames": random.choice(self.video_lengths),
                "fps": random.choice([8, 12, 16, 24]),
                "guidance_scale": random.uniform(7.0, 15.0),
                "motion_bucket_id": random.randint(100, 200)
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_image_video_pairs(
        self,
        num_samples: int = 3000
    ) -> List[Dict[str, Any]]:
        """Generate image-to-video training pairs"""
        
        samples = []
        
        motion_types = [
            "zoom in", "zoom out", "pan left", "pan right", "pan up", "pan down",
            "rotate clockwise", "rotate counterclockwise", "dolly forward", "dolly backward",
            "parallax effect", "morphing", "breathing effect", "floating"
        ]
        
        for i in range(num_samples):
            motion = random.choice(motion_types)
            
            # Generate description for the motion
            motion_prompt = f"smooth {motion} motion"
            
            sample = {
                "id": f"i2v_{i:06d}",
                "motion_description": motion_prompt,
                "motion_type": motion,
                "video_size": random.choice(self.video_sizes),
                "num_frames": random.choice(self.video_lengths),
                "fps": random.choice([7, 12, 16, 24]),
                "motion_bucket_id": random.randint(50, 250),
                "noise_aug_strength": random.uniform(0.0, 0.1)
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_enhancement_pairs(
        self,
        num_samples: int = 5000
    ) -> List[Dict[str, Any]]:
        """Generate image enhancement training pairs"""
        
        samples = []
        
        degradation_types = [
            "blur", "noise", "compression", "low_resolution", "overexposure",
            "underexposure", "color_shift", "artifacts", "pixelation"
        ]
        
        for i in range(num_samples):
            degradation = random.choice(degradation_types)
            
            sample = {
                "id": f"enh_{i:06d}",
                "degradation_type": degradation,
                "scale_factor": random.choice([2, 4, 8]),
                "noise_level": random.uniform(0.0, 0.3),
                "blur_kernel": random.choice([3, 5, 7, 9]),
                "compression_quality": random.randint(10, 90)
            }
            
            samples.append(sample)
        
        return samples
    
    def create_synthetic_dataset(
        self,
        dataset_type: str,
        num_samples: int,
        save_path: str
    ) -> str:
        """Create synthetic dataset with generated samples"""
        
        if dataset_type == "text_to_image":
            samples = self.generate_text_image_pairs(num_samples)
        elif dataset_type == "text_to_video":
            samples = self.generate_text_video_pairs(num_samples)
        elif dataset_type == "image_to_video":
            samples = self.generate_image_video_pairs(num_samples)
        elif dataset_type == "enhancement":
            samples = self.generate_enhancement_pairs(num_samples)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, f"{save_path}.json")
        with open(dataset_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"Created {dataset_type} dataset with {len(samples)} samples: {dataset_path}")
        return dataset_path
    
    def download_external_datasets(self):
        """Download and prepare external datasets"""
        
        datasets_to_download = [
            {
                "name": "laion-aesthetics",
                "hf_path": "laion/laion2B-en-aesthetic",
                "subset": "aesthetic_score_6_plus",
                "split": "train[:10000]"
            },
            {
                "name": "conceptual-captions",
                "hf_path": "conceptual_captions",
                "split": "train[:50000]"
            },
            {
                "name": "coco-captions",
                "hf_path": "coco",
                "subset": "2017",
                "split": "train[:25000]"
            }
        ]
        
        for dataset_info in datasets_to_download:
            try:
                print(f"Downloading {dataset_info['name']}...")
                
                if "subset" in dataset_info:
                    dataset = load_dataset(
                        dataset_info["hf_path"],
                        dataset_info["subset"],
                        split=dataset_info["split"]
                    )
                else:
                    dataset = load_dataset(
                        dataset_info["hf_path"],
                        split=dataset_info["split"]
                    )
                
                # Save processed dataset
                output_path = os.path.join(self.output_dir, f"{dataset_info['name']}.json")
                dataset.to_json(output_path)
                
                print(f"Saved {dataset_info['name']} to {output_path}")
                
            except Exception as e:
                print(f"Failed to download {dataset_info['name']}: {e}")
    
    def create_training_splits(
        self,
        dataset_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, str]:
        """Split dataset into train/validation/test sets"""
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Shuffle data
        random.shuffle(data)
        
        # Calculate split indices
        total_samples = len(data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # Create splits
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        # Save splits
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        
        splits = {}
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            split_path = os.path.join(self.output_dir, f"{base_name}_{split_name}.json")
            with open(split_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            splits[split_name] = split_path
            print(f"Created {split_name} split with {len(split_data)} samples: {split_path}")
        
        return splits
    
    def _generate_portrait_prompt(self, template: str) -> str:
        subjects = ["person", "woman", "man", "child", "elderly person", "artist", "scientist"]
        styles = ["in studio lighting", "with natural lighting", "in golden hour", "with dramatic shadows"]
        lightings = ["with soft lighting", "with hard lighting", "backlit", "with rim lighting"]
        backgrounds = ["with blurred background", "in nature", "in urban setting", "with solid background"]
        moods = ["smiling", "serious", "contemplative", "confident", "peaceful"]
        
        return template.format(
            subject=random.choice(subjects),
            style=random.choice(styles),
            lighting=random.choice(lightings),
            background=random.choice(backgrounds),
            mood=random.choice(moods)
        )
    
    def _generate_landscape_prompt(self, template: str) -> str:
        times = ["sunrise", "sunset", "golden hour", "blue hour", "midday", "twilight"]
        locations = ["mountain", "forest", "beach", "desert", "valley", "lake", "river", "canyon"]
        weathers = ["with clear skies", "with dramatic clouds", "in fog", "in rain", "in snow"]
        seasons = ["in spring", "in summer", "in autumn", "in winter"]
        atmospheres = ["peaceful", "dramatic", "mysterious", "serene", "majestic"]
        lightings = ["with warm lighting", "with cool lighting", "with dramatic lighting"]
        
        return template.format(
            time=random.choice(times),
            location=random.choice(locations),
            weather=random.choice(weathers),
            season=random.choice(seasons),
            atmosphere=random.choice(atmospheres),
            lighting=random.choice(lightings)
        )
    
    def _generate_object_prompt(self, template: str) -> str:
        adjectives = ["beautiful", "elegant", "modern", "vintage", "sleek", "rustic", "ornate"]
        objects = ["vase", "chair", "lamp", "book", "flower", "cup", "sculpture", "jewelry"]
        contexts = ["in studio", "on table", "in room", "outdoors", "in gallery"]
        surfaces = ["wooden table", "marble surface", "glass table", "fabric", "stone"]
        environments = ["modern room", "vintage setting", "natural environment", "minimalist space"]
        lightings = ["with soft lighting", "with dramatic lighting", "with natural light"]
        styles = ["minimalist style", "artistic style", "commercial style", "fine art style"]
        moods = ["elegant", "dramatic", "peaceful", "vibrant", "sophisticated"]
        
        return template.format(
            adjective=random.choice(adjectives),
            object=random.choice(objects),
            context=random.choice(contexts),
            surface=random.choice(surfaces),
            environment=random.choice(environments),
            lighting=random.choice(lightings),
            style=random.choice(styles),
            mood=random.choice(moods)
        )
    
    def _generate_abstract_prompt(self, template: str) -> str:
        colors = ["blue", "red", "green", "purple", "orange", "yellow", "pink", "teal"]
        patterns = ["swirls", "waves", "geometric patterns", "flowing lines", "fractals"]
        textures = ["smooth", "rough", "metallic", "organic", "crystalline", "fluid"]
        shapes = ["circles", "triangles", "squares", "spirals", "curves", "polygons"]
        concepts = ["energy", "movement", "harmony", "chaos", "balance", "transformation"]
        atmospheres = ["dreamy", "energetic", "calming", "dynamic", "ethereal"]
        styles = ["digital art", "watercolor", "oil painting", "3D render", "vector art"]
        
        return template.format(
            color=random.choice(colors),
            pattern=random.choice(patterns),
            texture=random.choice(textures),
            shape=random.choice(shapes),
            concept=random.choice(concepts),
            atmosphere=random.choice(atmospheres),
            style=random.choice(styles)
        )
    
    def _generate_action_prompt(self, template: str) -> str:
        subjects = ["person", "athlete", "dancer", "musician", "artist", "worker"]
        actions = ["running", "jumping", "dancing", "painting", "playing", "working"]
        locations = ["in park", "on stage", "in studio", "outdoors", "in gym", "at beach"]
        times = ["at sunset", "in morning", "at night", "during day"]
        styles = ["dynamic", "energetic", "graceful", "powerful", "fluid"]
        
        return template.format(
            subject=random.choice(subjects),
            action=random.choice(actions),
            location=random.choice(locations),
            time=random.choice(times),
            style=random.choice(styles)
        )
    
    def _generate_video_action_prompt(self, template: str) -> str:
        subjects = ["person", "animal", "vehicle", "object"]
        actions = ["moving", "transforming", "rotating", "flowing", "dancing", "flying"]
        locations = ["through space", "in nature", "in city", "underwater", "in air"]
        camera_movements = ["camera panning", "zooming in", "tracking shot", "dolly movement"]
        effects = ["motion blur", "particle effects", "light trails", "slow motion"]
        
        return template.format(
            subject=random.choice(subjects),
            action=random.choice(actions),
            location=random.choice(locations),
            camera_movement=random.choice(camera_movements),
            effect=random.choice(effects)
        )
    
    def _generate_video_landscape_prompt(self, template: str) -> str:
        elements = ["water", "clouds", "trees", "grass", "sand", "snow"]
        locations = ["mountains", "ocean", "forest", "desert", "valley", "plains"]
        
        return template.format(
            element=random.choice(elements),
            location=random.choice(locations)
        )
    
    def _generate_video_object_prompt(self, template: str) -> str:
        objects = ["sphere", "cube", "flower", "crystal", "machine", "sculpture"]
        other_objects = ["butterfly", "bird", "light", "energy", "particle"]
        actions = ["spinning", "glowing", "pulsing", "morphing", "floating"]
        environments = ["in space", "underwater", "in forest", "in laboratory"]
        
        return template.format(
            object=random.choice(objects),
            other_object=random.choice(other_objects),
            action=random.choice(actions),
            environment=random.choice(environments)
        )
    
    def _generate_negative_prompt(self) -> str:
        negative_terms = [
            "blurry", "low quality", "distorted", "ugly", "deformed",
            "bad anatomy", "extra limbs", "missing limbs", "watermark",
            "text", "signature", "low resolution", "pixelated", "artifacts"
        ]
        
        return ", ".join(random.sample(negative_terms, random.randint(3, 6)))
    
    def _generate_video_negative_prompt(self) -> str:
        video_negative_terms = [
            "static", "no movement", "flickering", "unstable", "jittery",
            "low frame rate", "compression artifacts", "blurry motion",
            "inconsistent", "choppy", "frozen frames", "glitches"
        ]
        
        general_negative = self._generate_negative_prompt()
        video_negative = ", ".join(random.sample(video_negative_terms, random.randint(2, 4)))
        
        return f"{general_negative}, {video_negative}"

# Usage example and dataset creation
if __name__ == "__main__":
    generator = MultimodalDatasetGenerator()
    
    # Create comprehensive datasets
    datasets = [
        ("text_to_image", 50000, "comprehensive_t2i_dataset"),
        ("text_to_video", 25000, "comprehensive_t2v_dataset"),
        ("image_to_video", 15000, "comprehensive_i2v_dataset"),
        ("enhancement", 20000, "comprehensive_enhancement_dataset")
    ]
    
    for dataset_type, num_samples, filename in datasets:
        dataset_path = generator.create_synthetic_dataset(dataset_type, num_samples, filename)
        generator.create_training_splits(dataset_path)
    
    # Download external datasets
    generator.download_external_datasets()
    
    print("Dataset generation completed!")