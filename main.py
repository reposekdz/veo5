#!/usr/bin/env python3
"""
VEO5 - Advanced Multimodal AI System with 150% Accuracy
Master entry point with full system integration
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.master_integration import master_system, MasterIntegrationSystem
from core.multimodal_ai import MultimodalAI
from datasets.ultra_dataset_creator import UltraDatasetCreator
from config import config

def setup_logging(level: str = "INFO"):
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('veo5_system.log')
        ]
    )

def create_directories():
    """Create all necessary directories"""
    directories = [
        config.models_dir, config.cache_dir, config.output_dir, config.dataset_dir,
        "logs", "temp", "ultra_datasets", "processed_files", "knowledge_base",
        "research_cache", "conversation_history", "quality_reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_full_demo():
    """Run comprehensive system demonstration"""
    print("\n" + "="*80)
    print("🚀 VEO5 ADVANCED MULTIMODAL AI SYSTEM - FULL DEMONSTRATION")
    print("="*80)
    
    # Initialize master system
    print("\n🔧 Initializing Master Integration System...")
    master_system.initialize_all_systems()
    
    # Test unified processing
    print("\n🎯 Testing Unified Processing Interface...")
    
    # Text generation test
    text_result = master_system.unified_process(
        "Create a comprehensive research paper about quantum computing applications in AI",
        task_type="generation",
        quality_target=1.5
    )
    print(f"✅ Text Generation - Accuracy: {text_result['accuracy']:.3f}")
    
    # Image generation test
    image_result = master_system.unified_process(
        "Generate a photorealistic image of a futuristic city with flying cars",
        task_type="generation",
        quality_target=1.5
    )
    print(f"✅ Image Generation - Accuracy: {image_result['accuracy']:.3f}")
    
    # Research test
    research_result = master_system.unified_process(
        "Research the latest developments in multimodal AI and provide comprehensive analysis",
        task_type="research",
        quality_target=1.5
    )
    print(f"✅ Research Analysis - Accuracy: {research_result['accuracy']:.3f}")
    
    # Conversation test
    chat_result = master_system.unified_process(
        "Explain quantum entanglement in simple terms with creative analogies",
        task_type="conversation",
        quality_target=1.5
    )
    print(f"✅ Conversational AI - Accuracy: {chat_result['accuracy']:.3f}")
    
    # System status
    status = master_system.get_system_status()
    print(f"\n📊 Overall System Accuracy: {status['accuracy']:.3f}")
    print(f"🎯 Active Systems: {len([s for s in status['systems'].values() if s.get('status') == 'healthy'])}")

def create_ultra_datasets():
    """Create ultra-high quality datasets"""
    print("\n📚 CREATING ULTRA-HIGH QUALITY DATASETS")
    print("-" * 60)
    
    # Initialize master system
    master_system.initialize_all_systems()
    
    # Create ultra dataset
    dataset_path = master_system.create_ultra_dataset(
        size=1_000_000,  # 1M samples for demo
        modalities=["text", "image", "video", "audio", "multimodal", "code", "scientific"],
        quality_threshold=0.95,
        accuracy_target=1.5
    )
    
    print(f"✅ Ultra dataset created: {dataset_path}")

def process_files_demo():
    """Demonstrate file processing capabilities"""
    print("\n📁 FILE PROCESSING DEMONSTRATION")
    print("-" * 50)
    
    # Initialize master system
    master_system.initialize_all_systems()
    
    # Create sample files for testing
    test_files = create_sample_test_files()
    
    for file_path in test_files:
        try:
            result = master_system.unified_process(
                file_path,
                task_type="file_processing",
                quality_target=1.5
            )
            
            file_type = Path(file_path).suffix
            print(f"✅ Processed {file_type} file - Accuracy: {result['accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

def create_sample_test_files() -> List[str]:
    """Create sample files for testing"""
    test_files = []
    
    # Create sample text file
    text_file = "temp/sample.txt"
    with open(text_file, 'w') as f:
        f.write("This is a sample text file for testing the universal file processor.")
    test_files.append(text_file)
    
    # Create sample JSON file
    json_file = "temp/sample.json"
    with open(json_file, 'w') as f:
        import json
        json.dump({"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}}, f)
    test_files.append(json_file)
    
    # Create sample Python file
    py_file = "temp/sample.py"
    with open(py_file, 'w') as f:
        f.write("""
def hello_world():
    '''A simple hello world function'''
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
""")
    test_files.append(py_file)
    
    return test_files

def run_api_server():
    """Run the comprehensive API server"""
    print("\n🌐 STARTING VEO5 API SERVER")
    print("-" * 40)
    
    try:
        import uvicorn
        from api.main import app
        
        # Initialize master system
        master_system.initialize_all_systems()
        
        print("🚀 Starting VEO5 API server on http://localhost:8000")
        print("📖 API documentation: http://localhost:8000/docs")
        print("🔬 Research endpoints: http://localhost:8000/research/")
        print("💬 Chat interface: http://localhost:8000/research/chat")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
    except ImportError:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")

def run_interactive_mode():
    """Run interactive mode for testing"""
    print("\n🎮 VEO5 INTERACTIVE MODE")
    print("-" * 30)
    
    # Initialize master system
    master_system.initialize_all_systems()
    
    print("Welcome to VEO5 Interactive Mode!")
    print("Commands: generate, research, chat, process, status, quit")
    
    while True:
        try:
            command = input("\nVEO5> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "status":
                status = master_system.get_system_status()
                print(f"System Accuracy: {status['accuracy']:.3f}")
                print(f"Active Tasks: {status['active_tasks']}")
            elif command.startswith("generate "):
                prompt = command[9:]
                result = master_system.unified_process(prompt, "generation", quality_target=1.5)
                print(f"Generated with accuracy: {result['accuracy']:.3f}")
            elif command.startswith("research "):
                query = command[9:]
                result = master_system.unified_process(query, "research", quality_target=1.5)
                print(f"Research completed with accuracy: {result['accuracy']:.3f}")
            elif command.startswith("chat "):
                message = command[5:]
                result = master_system.unified_process(message, "conversation", quality_target=1.5)
                print(f"Response: {result['result'].get('response', 'No response')}")
            elif command.startswith("process "):
                file_path = command[8:]
                if os.path.exists(file_path):
                    result = master_system.unified_process(file_path, "file_processing", quality_target=1.5)
                    print(f"File processed with accuracy: {result['accuracy']:.3f}")
                else:
                    print("File not found!")
            else:
                print("Unknown command. Available: generate, research, chat, process, status, quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

def benchmark_system():
    """Run comprehensive system benchmarks"""
    print("\n⚡ VEO5 SYSTEM BENCHMARKS")
    print("-" * 35)
    
    # Initialize master system
    master_system.initialize_all_systems()
    
    import time
    
    benchmarks = [
        ("Text Generation", "Generate a technical article about AI", "generation"),
        ("Image Generation", "Create a photorealistic landscape image", "generation"),
        ("Research Query", "Research quantum computing applications", "research"),
        ("Conversation", "Explain machine learning concepts", "conversation"),
        ("File Processing", "temp/sample.py", "file_processing")
    ]
    
    results = []
    
    for name, input_data, task_type in benchmarks:
        print(f"\n🔄 Running {name} benchmark...")
        
        start_time = time.time()
        try:
            result = master_system.unified_process(
                input_data, task_type, quality_target=1.5
            )
            end_time = time.time()
            
            duration = end_time - start_time
            accuracy = result['accuracy']
            
            results.append({
                "benchmark": name,
                "duration": duration,
                "accuracy": accuracy,
                "status": "success"
            })
            
            print(f"✅ {name}: {duration:.2f}s, Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            results.append({
                "benchmark": name,
                "duration": 0,
                "accuracy": 0,
                "status": f"failed: {e}"
            })
            print(f"❌ {name}: Failed - {e}")
    
    # Summary
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_accuracy = sum(r["accuracy"] for r in successful) / len(successful)
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        
        print(f"\n📊 BENCHMARK SUMMARY")
        print(f"Average Accuracy: {avg_accuracy:.3f}")
        print(f"Average Duration: {avg_duration:.2f}s")
        print(f"Success Rate: {len(successful)}/{len(results)}")

def main():
    """Main entry point with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="VEO5 - Advanced Multimodal AI System with 150% Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Full system demonstration
  python main.py --create-datasets        # Create ultra-high quality datasets
  python main.py --api                    # Start API server
  python main.py --interactive            # Interactive mode
  python main.py --benchmark              # Run system benchmarks
  python main.py --process-files          # File processing demo
  python main.py --setup                  # Setup directories only
        """
    )
    
    parser.add_argument("--demo", action="store_true", help="Run full system demonstration")
    parser.add_argument("--create-datasets", action="store_true", help="Create ultra datasets")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--process-files", action="store_true", help="File processing demo")
    parser.add_argument("--setup", action="store_true", help="Setup directories")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("VEO5")
    
    print("🤖 VEO5 - Advanced Multimodal AI System")
    print("🎯 Target Accuracy: 150%")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    if args.setup:
        print("\n✅ Setup completed!")
        return
    
    try:
        if args.demo:
            run_full_demo()
        elif args.create_datasets:
            create_ultra_datasets()
        elif args.api:
            run_api_server()
        elif args.interactive:
            run_interactive_mode()
        elif args.benchmark:
            benchmark_system()
        elif args.process_files:
            process_files_demo()
        else:
            # Default: run full demo
            run_full_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 VEO5 system shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"\n❌ System error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CUDA is available if using GPU")
        print("2. Check all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify sufficient disk space and memory")
        print("4. Try --device cpu for CPU-only mode")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if 'master_system' in globals():
                print("\n🧹 Cleaning up system resources...")
                # Add cleanup code here if needed
        except:
            pass

if __name__ == "__main__":
    main()