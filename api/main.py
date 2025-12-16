from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import asyncio
import io
import base64
import numpy as np
from PIL import Image
import json
import os
import uuid
import logging
from datetime import datetime
import redis
import celery
from celery import Celery

# Import our multimodal AI system
import sys
sys.path.append('..')
from core.multimodal_ai import MultimodalAI
from config import config
from .research_endpoints import router as research_router

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Multimodal AI API",
    description="Powerful multimodal AI system for text-to-image, text-to-video, image-to-video, and enhancement",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include research router
app.include_router(research_router)

# Initialize multimodal AI system
ai_system = MultimodalAI()

# Redis for caching and task queue
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Celery for background tasks
celery_app = Celery(
    'multimodal_ai',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Pydantic models for API requests
class TextToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    model: str = Field("sdxl", description="Model to use")
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    num_images: int = Field(1, ge=1, le=4)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, ge=10, le=150)
    seed: Optional[int] = Field(None, description="Random seed")
    enhance: bool = Field(False, description="Apply enhancement")
    enhancement_model: str = Field("realesrgan", description="Enhancement model")

class TextToVideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    model: str = Field("zeroscope", description="Model to use")
    width: int = Field(576, ge=256, le=1024)
    height: int = Field(320, ge=256, le=1024)
    num_frames: int = Field(24, ge=8, le=120)
    fps: int = Field(8, ge=4, le=30)
    guidance_scale: float = Field(9.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(50, ge=10, le=100)
    seed: Optional[int] = Field(None, description="Random seed")
    enhance: bool = Field(False, description="Apply enhancement")

class ImageToVideoRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Optional text prompt")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    model: str = Field("stable_video", description="Model to use")
    width: int = Field(1024, ge=256, le=1024)
    height: int = Field(576, ge=256, le=1024)
    num_frames: int = Field(25, ge=8, le=120)
    fps: int = Field(7, ge=4, le=30)
    motion_bucket_id: int = Field(127, ge=1, le=255)
    seed: Optional[int] = Field(None, description="Random seed")
    enhance: bool = Field(False, description="Apply enhancement")

class EnhanceImageRequest(BaseModel):
    model: str = Field("realesrgan", description="Enhancement model")
    scale: int = Field(4, ge=2, le=8)
    face_enhance: bool = Field(True, description="Apply face enhancement")

class EnhanceVideoRequest(BaseModel):
    model: str = Field("basicvsr", description="Enhancement model")
    scale: int = Field(4, ge=2, le=8)
    temporal_consistency: bool = Field(True, description="Apply temporal consistency")

class WorkflowStep(BaseModel):
    type: str = Field(..., description="Step type")
    args: Dict[str, Any] = Field(..., description="Step arguments")
    input_from: Optional[int] = Field(None, description="Input from previous step")

class WorkflowRequest(BaseModel):
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")

class BatchRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="Batch tasks")
    max_concurrent: int = Field(2, ge=1, le=4)

# Response models
class GenerationResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# Utility functions
def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def save_result(task_id: str, result: Dict[str, Any]) -> str:
    """Save generation result to disk and return file path"""
    output_dir = os.path.join(config.output_dir, task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(result.get('metadata', {}), f, indent=2, default=str)
    
    file_paths = []
    
    # Save images
    if 'images' in result:
        for i, image in enumerate(result['images']):
            image_path = os.path.join(output_dir, f"image_{i:03d}.png")
            image.save(image_path)
            file_paths.append(image_path)
    
    # Save video frames
    if 'frames' in result:
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(result['frames']):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            if isinstance(frame, np.ndarray):
                Image.fromarray(frame).save(frame_path)
            else:
                frame.save(frame_path)
            file_paths.append(frame_path)
        
        # Create video file
        video_path = os.path.join(output_dir, "video.mp4")
        create_video_from_frames(result['frames'], video_path, result['metadata'].get('fps', 24))
        file_paths.append(video_path)
    
    return output_dir

def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 24):
    """Create video file from frames"""
    import cv2
    
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
    
    out.release()

# Celery tasks for background processing
@celery_app.task(bind=True)
def process_text_to_image(self, task_id: str, request_data: dict):
    """Background task for text-to-image generation"""
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Generating images...'})
        
        result = ai_system.text_to_image(**request_data)
        
        # Save result
        output_dir = save_result(task_id, result)
        
        # Store in Redis
        redis_client.setex(
            f"result:{task_id}",
            3600,  # 1 hour expiry
            json.dumps({
                'status': 'completed',
                'output_dir': output_dir,
                'metadata': result['metadata']
            }, default=str)
        )
        
        return {'status': 'completed', 'output_dir': output_dir}
        
    except Exception as e:
        redis_client.setex(
            f"result:{task_id}",
            3600,
            json.dumps({'status': 'failed', 'error': str(e)})
        )
        raise

@celery_app.task(bind=True)
def process_text_to_video(self, task_id: str, request_data: dict):
    """Background task for text-to-video generation"""
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Generating video...'})
        
        result = ai_system.text_to_video(**request_data)
        
        # Save result
        output_dir = save_result(task_id, result)
        
        # Store in Redis
        redis_client.setex(
            f"result:{task_id}",
            3600,
            json.dumps({
                'status': 'completed',
                'output_dir': output_dir,
                'metadata': result['metadata']
            }, default=str)
        )
        
        return {'status': 'completed', 'output_dir': output_dir}
        
    except Exception as e:
        redis_client.setex(
            f"result:{task_id}",
            3600,
            json.dumps({'status': 'failed', 'error': str(e)})
        )
        raise

@celery_app.task(bind=True)
def process_image_to_video(self, task_id: str, image_data: str, request_data: dict):
    """Background task for image-to-video generation"""
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Generating video from image...'})
        
        # Decode image
        image = decode_base64_to_image(image_data)
        request_data['image'] = image
        
        result = ai_system.image_to_video(**request_data)
        
        # Save result
        output_dir = save_result(task_id, result)
        
        # Store in Redis
        redis_client.setex(
            f"result:{task_id}",
            3600,
            json.dumps({
                'status': 'completed',
                'output_dir': output_dir,
                'metadata': result['metadata']
            }, default=str)
        )
        
        return {'status': 'completed', 'output_dir': output_dir}
        
    except Exception as e:
        redis_client.setex(
            f"result:{task_id}",
            3600,
            json.dumps({'status': 'failed', 'error': str(e)})
        )
        raise

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Advanced Multimodal AI API",
        "version": "1.0.0",
        "description": "Powerful multimodal AI system",
        "endpoints": {
            "text_to_image": "/generate/text-to-image",
            "text_to_video": "/generate/text-to-video",
            "image_to_video": "/generate/image-to-video",
            "enhance_image": "/enhance/image",
            "enhance_video": "/enhance/video",
            "workflow": "/workflow",
            "batch": "/batch"
        }
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return ai_system.get_available_models()

@app.get("/stats")
async def get_system_stats():
    """Get system performance statistics"""
    return ai_system.get_stats()

@app.post("/generate/text-to-image")
async def generate_text_to_image(request: TextToImageRequest):
    """Generate images from text prompt"""
    task_id = str(uuid.uuid4())
    
    # Start background task
    task = process_text_to_image.delay(task_id, request.dict())
    
    return {
        "task_id": task_id,
        "celery_task_id": task.id,
        "status": "queued",
        "message": "Image generation started"
    }

@app.post("/generate/text-to-video")
async def generate_text_to_video(request: TextToVideoRequest):
    """Generate video from text prompt"""
    task_id = str(uuid.uuid4())
    
    # Start background task
    task = process_text_to_video.delay(task_id, request.dict())
    
    return {
        "task_id": task_id,
        "celery_task_id": task.id,
        "status": "queued",
        "message": "Video generation started"
    }

@app.post("/generate/image-to-video")
async def generate_image_to_video(
    request: ImageToVideoRequest,
    image: UploadFile = File(...)
):
    """Generate video from input image"""
    task_id = str(uuid.uuid4())
    
    # Read and encode image
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    image_base64 = encode_image_to_base64(pil_image)
    
    # Start background task
    task = process_image_to_video.delay(task_id, image_base64, request.dict())
    
    return {
        "task_id": task_id,
        "celery_task_id": task.id,
        "status": "queued",
        "message": "Image-to-video generation started"
    }

@app.post("/enhance/image")
async def enhance_image_endpoint(
    request: EnhanceImageRequest,
    image: UploadFile = File(...)
):
    """Enhance image quality"""
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Enhance image
        result = ai_system.enhance_image(
            image=pil_image,
            **request.dict()
        )
        
        # Encode result
        enhanced_base64 = encode_image_to_base64(result['image'])
        
        return {
            "status": "completed",
            "image": enhanced_base64,
            "metadata": result['metadata']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result"""
    try:
        # Check Redis for result
        result_data = redis_client.get(f"result:{task_id}")
        
        if result_data:
            result = json.loads(result_data)
            return result
        else:
            return {"status": "not_found", "message": "Task not found"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """Download generated file"""
    try:
        # Get result info
        result_data = redis_client.get(f"result:{task_id}")
        if not result_data:
            raise HTTPException(status_code=404, detail="Task not found")
        
        result = json.loads(result_data)
        output_dir = result.get('output_dir')
        
        if not output_dir:
            raise HTTPException(status_code=404, detail="No output directory found")
        
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow")
async def execute_workflow(request: WorkflowRequest):
    """Execute multi-step workflow"""
    try:
        result = ai_system.create_workflow(
            steps=[step.dict() for step in request.steps]
        )
        
        return {
            "status": "completed" if result['success'] else "failed",
            "results": result['steps']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def process_batch(request: BatchRequest):
    """Process multiple tasks in batch"""
    try:
        results = ai_system.batch_process(
            tasks=request.tasks,
            max_concurrent=request.max_concurrent
        )
        
        return {
            "status": "completed",
            "results": results,
            "total_tasks": len(request.tasks),
            "successful_tasks": len([r for r in results if "error" not in r])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MultimodalAI-API")
    logger.info("Multimodal AI API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    ai_system.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)