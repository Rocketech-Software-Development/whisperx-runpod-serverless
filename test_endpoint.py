#!/usr/bin/env python3
"""
Test script for RunPod WhisperX endpoint
"""
import runpod
import base64
import requests

# Replace with your endpoint ID and API key
ENDPOINT_ID = "ae9ewyg6bmlp75"  # Your endpoint ID
API_KEY = "your-runpod-api-key"  # Get from RunPod dashboard

# Initialize RunPod client
runpod.api_key = API_KEY
endpoint = runpod.Endpoint(ENDPOINT_ID)

# Test with a sample audio file
def test_with_url():
    """Test with a public audio URL"""
    payload = {
        "input": {
            "audio_url": "https://www.example.com/sample.mp3",
            "language": "en",
            "batch_size": 16
        }
    }
    
    print("Testing with audio URL...")
    run = endpoint.run(payload)
    print(f"Job ID: {run.job_id}")
    
    # Wait for result
    result = run.output()
    print(f"Result: {result}")

# Test with base64 audio
def test_with_base64():
    """Test with base64 encoded audio"""
    # Read a local audio file
    with open("test_audio.wav", "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "input": {
            "audio_base64": audio_base64,
            "language": "en",
            "batch_size": 16
        }
    }
    
    print("Testing with base64 audio...")
    run = endpoint.run(payload)
    result = run.output()
    print(f"Result: {result}")

if __name__ == "__main__":
    print(f"Testing RunPod endpoint: {ENDPOINT_ID}")
    test_with_url()
    # test_with_base64()  # Uncomment if you have a local audio file 