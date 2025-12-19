"""
Camera vision module for capturing and processing camera frames.
"""
import cv2
import base64
import io
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("local-agent.camera")


class CameraVision:
    def __init__(self, device_path: str = "/dev/video0"):
        """
        Initialize camera capture.
        
        Args:
            device_path: Path to the video device (e.g., /dev/video0)
        """
        self.device_path = device_path
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera capture."""
        try:
            # Try multiple methods to open the camera
            if self.device_path.startswith("/dev/video"):
                # Extract device index from /dev/video0 -> 0
                device_index = int(self.device_path.split("video")[1])
                
                # Method 1: Try with device index
                self.cap = cv2.VideoCapture(device_index)
                
                if not self.cap.isOpened():
                    # Method 2: Try with V4L2 backend explicitly
                    self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
                
                if not self.cap.isOpened():
                    # Method 3: Try with the full device path
                    self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
                
                if not self.cap.isOpened():
                    # Method 4: Try with device path as string
                    self.cap = cv2.VideoCapture(self.device_path)
            else:
                self.cap = cv2.VideoCapture(self.device_path)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera at {self.device_path}. Make sure the camera is not in use by another application (like VLC).")
            
            # Set some reasonable defaults
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test capture to ensure it works
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                raise RuntimeError(f"Camera opened but cannot capture frames from {self.device_path}")
            
            logger.info(f"Camera initialized successfully at {self.device_path}")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            if self.cap is not None:
                self.cap.release()
            self.cap = None
    
    def capture_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: The captured frame (BGR format), or None if capture failed
        """
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera not available")
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to capture frame from camera")
                return None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def frame_to_base64(self, frame, format: str = "JPEG", quality: int = 85):
        """
        Convert a frame to base64 encoded string.
        
        Args:
            frame: numpy array (BGR format from OpenCV)
            format: Image format (JPEG or PNG)
            quality: JPEG quality (1-100)
        
        Returns:
            str: Base64 encoded image string
        """
        if frame is None:
            return None
        
        try:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to bytes
            buffer = io.BytesIO()
            if format.upper() == "JPEG":
                pil_image.save(buffer, format="JPEG", quality=quality)
            else:
                pil_image.save(buffer, format="PNG")
            
            # Encode to base64
            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/{format.lower()};base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return None
    
    def get_latest_frame_base64(self, format: str = "JPEG", quality: int = 85):
        """
        Capture the latest frame and return it as base64.
        
        Args:
            format: Image format (JPEG or PNG)
            quality: JPEG quality (1-100)
        
        Returns:
            str: Base64 encoded image string, or None if capture failed
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        return self.frame_to_base64(frame, format=format, quality=quality)
    
    def describe_frame_simple(self, frame):
        """
        Simple frame description (can be enhanced with vision models later).
        For now, returns basic info about the frame.
        
        Args:
            frame: numpy array frame
        
        Returns:
            str: Basic description of the frame
        """
        if frame is None:
            return "No frame available"
        
        height, width = frame.shape[:2]
        return f"A {width}x{height} image captured from the camera"
    
    def close(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
