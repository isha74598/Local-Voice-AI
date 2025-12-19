#!/usr/bin/env python3
"""Test script for camera vision"""
from camera_vision import CameraVision

def test_camera():
    print("Testing camera at /dev/video0...")
    cam = CameraVision("/dev/video0")
    
    if cam.cap is None:
        print("❌ Camera not available")
        return
    
    print("✅ Camera opened successfully")
    
    frame = cam.capture_frame()
    if frame is not None:
        print(f"✅ Frame captured! Shape: {frame.shape}")
        b64 = cam.get_latest_frame_base64()
        if b64:
            print(f"✅ Base64 encoded! Length: {len(b64)} characters")
        else:
            print("⚠️ Failed to encode to base64")
    else:
        print("❌ Failed to capture frame")
    
    cam.close()
    print("✅ Camera test complete")

if __name__ == "__main__":
    test_camera()
