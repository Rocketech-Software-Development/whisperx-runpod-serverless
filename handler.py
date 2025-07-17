#!/usr/bin/env python3
"""
RunPod Serverless Handler for WhisperX v3.4.2
With built-in compatibility patch for faster-whisper API changes
"""

import os
import gc
import tempfile
import logging
import traceback
import time
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import urllib.request

# Apply compatibility patch BEFORE importing WhisperX
import sys
import importlib.util

# First, import faster_whisper to patch it
spec = importlib.util.find_spec("faster_whisper.transcribe")
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Store original TranscriptionOptions
    OriginalTranscriptionOptions = module.TranscriptionOptions
    
    # Create patched version
    class PatchedTranscriptionOptions(OriginalTranscriptionOptions):
        """Patched TranscriptionOptions that filters out incompatible parameters"""
        
        def __init__(self, *args, **kwargs):
            # Remove problematic parameters that don't exist in newer faster-whisper
            problematic_params = [
                'multilingual', 'output_language', 'max_new_tokens',
                'clip_timestamps', 'hallucination_silence_threshold'
            ]
            
            # Filter out problematic parameters
            filtered = [p for p in problematic_params if p in kwargs]
            if filtered:
                logging.info(f"🔧 WhisperX compatibility patch: Filtered parameters {filtered}")
            
            # Clean kwargs
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in problematic_params}
            
            # Call parent constructor with cleaned parameters
            super().__init__(*args, **cleaned_kwargs)
    
    # Apply the patch
    module.TranscriptionOptions = PatchedTranscriptionOptions
    sys.modules['faster_whisper.transcribe'] = module
    logging.info("✅ WhisperX v3.4.2 compatibility patch applied successfully")

# Now we can safely import whisperx
import torch
import whisperx
import runpod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperXHandler:
    def __init__(self):
        self.device = None
        self.compute_type = None
        self.whisper_model = None
        self.align_model = None
        self.diarize_model = None
        self.language = None
        self.model_name = None
        self._setup_device()
    
    def _setup_device(self):
        """Setup CUDA device and compute type"""
        if torch.cuda.is_available():
            self.device = "cuda"
            # Use float16 for newer GPUs, int8 for older ones
            gpu_name = torch.cuda.get_device_name(0).lower()
            if any(x in gpu_name for x in ['a100', 'a6000', 'a40', 'h100', 'l4', 'l40', '4090', '3090']):
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)} with {self.compute_type}")
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            logger.warning("⚠️ No GPU available, using CPU (will be slow)")
    
    def load_whisper_model(self):
        """Load WhisperX model with caching"""
        model_name = os.getenv('MODEL_NAME', 'large-v3')
        
        if self.whisper_model is None or self.model_name != model_name:
            logger.info(f"📥 Loading Whisper model: {model_name}")
            self.model_name = model_name
            
            try:
                self.whisper_model = whisperx.load_model(
                    model_name,
                    self.device,
                    compute_type=self.compute_type,
                    language=self.language
                )
                logger.info(f"✅ Whisper model loaded: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {str(e)}")
                raise
    
    def load_align_model(self, language_code: str):
        """Load alignment model for word-level timestamps"""
        if self.align_model is None or self.language != language_code:
            logger.info(f"📥 Loading alignment model for language: {language_code}")
            self.align_model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            self.language = language_code
            logger.info(f"✅ Alignment model loaded for: {language_code}")
    
    def load_diarize_model(self):
        """Load speaker diarization model"""
        if self.diarize_model is None:
            hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
            if not hf_token:
                logger.warning("⚠️ No HuggingFace token provided, diarization may fail")
            
            logger.info("📥 Loading diarization model...")
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device
            )
            logger.info("✅ Diarization model loaded")
    
    def transcribe_audio(self, audio_path: str, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Main transcription pipeline"""
        # Get parameters
        language = job_input.get('language', None)
        enable_diarization = job_input.get('enable_diarization', True)
        batch_size = job_input.get('batch_size', 16)
        
        # Load models
        self.language = language
        self.load_whisper_model()
        
        # Step 1: Transcribe
        logger.info("🎤 Starting transcription...")
        audio = whisperx.load_audio(audio_path)
        
        result = self.whisper_model.transcribe(
            audio,
            batch_size=batch_size,
            language=language
        )
        
        logger.info(f"✅ Transcription complete. Detected language: {result.get('language', 'unknown')}")
        
        # Step 2: Align (word-level timestamps)
        if result.get('language'):
            try:
                logger.info("🔄 Aligning segments for word-level timestamps...")
                self.load_align_model(result['language'])
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
                logger.info("✅ Alignment complete")
            except Exception as e:
                logger.warning(f"⚠️ Alignment failed: {str(e)}")
        
        # Step 3: Diarize (speaker identification)
        if enable_diarization:
            try:
                logger.info("👥 Starting speaker diarization...")
                self.load_diarize_model()
                diarize_segments = self.diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("✅ Diarization complete")
            except Exception as e:
                logger.warning(f"⚠️ Diarization failed: {str(e)}")
        
        return result, audio


def download_file(url: str, dest: str) -> str:
    """Download file from URL"""
    logger.info(f"📥 Downloading from: {url}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"✅ Downloaded to: {dest}")
    return dest


def handler(job: Dict[str, Any]) -> Union[Dict[str, Any], str]:
    """RunPod serverless handler"""
    logger.info("🚀 Starting WhisperX job processing...")
    
    try:
        job_input = job['input']
        
        # Validate input
        if 'audio_url' not in job_input:
            raise ValueError("Missing required field: audio_url")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio
            audio_ext = job_input.get('audio_format', 'wav')
            audio_path = os.path.join(temp_dir, f"audio.{audio_ext}")
            download_file(job_input['audio_url'], audio_path)
            
            # Initialize handler
            whisperx_handler = WhisperXHandler()
            
            # Process audio
            result, audio = whisperx_handler.transcribe_audio(audio_path, job_input)
            
            # Format output
            output = {
                "segments": result.get("segments", []),
                "language": result.get("language", "unknown"),
                "text": " ".join([s.get("text", "") for s in result.get("segments", [])]),
            }
            
            # Add word timestamps if available
            if any('words' in s for s in result.get("segments", [])):
                output["word_segments"] = result["segments"]
            
            logger.info("✅ Job completed successfully")
            return output
            
    except Exception as e:
        logger.error(f"❌ Error processing job: {str(e)}")
        logger.error(f"📄 Traceback: {traceback.format_exc()}")
        return {"error": str(e)}
    
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Start RunPod serverless worker
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod WhisperX serverless worker...")
    runpod.serverless.start({"handler": handler}) 