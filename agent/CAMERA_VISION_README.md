# Camera Vision Integration

The voice assistant now has camera vision capabilities! It can see what your camera captures and describe it when you ask.

## How It Works

1. **Camera Access**: The agent connects to `/dev/video0` (your camera device)
2. **Vision Query Detection**: When you ask questions like:
   - "What can you see?"
   - "What do you see?"
   - "Describe what you see"
   - "What's in the camera?"
   
   The agent automatically captures a frame from the camera.

3. **Response**: The agent describes what it sees in the camera view.

## Setup

The camera module is already integrated. Just make sure:
- Your camera is accessible at `/dev/video0` (or change the path in `myagent.py`)
- No other application (like VLC) is using the camera when you run the agent
- Required packages are installed: `opencv-python-headless` and `Pillow`

## Using Vision Models (Optional)

For better vision descriptions, you can use a vision-capable model with Ollama:

1. **Install a vision model** (e.g., LLaVA):
   ```bash
   ollama pull llava
   ```

2. **Update the agent** to use the vision model:
   ```python
   agent = LocalAgent(use_vision_model=True)  # Uses 'llava' model
   ```

   Or change the model name in `myagent.py` line 145.

## Testing

Test the camera with:
```python
from camera_vision import CameraVision
cam = CameraVision("/dev/video0")
frame = cam.capture_frame()
print("Frame shape:", frame.shape if frame else "None")
cam.close()
```

## Troubleshooting

- **Camera not accessible**: Make sure no other app is using it (close VLC, etc.)
- **Permission denied**: You may need to add your user to the `video` group:
  ```bash
  sudo usermod -a -G video $USER
  ```
  Then log out and back in.

- **Wrong device**: Check available cameras:
  ```bash
  ls -l /dev/video*
  ```
