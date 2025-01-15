import pybullet as p
import cv2
import numpy as np
import os
from datetime import datetime

class SimulationRecorder:
    def __init__(self, width=1920, height=1080, fps=30):
        """Initialize the simulation recorder.
        
        Args:
            width (int): Width of the recorded video
            height (int): Height of the recorded video
            fps (int): Frames per second for the video
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.is_recording = False
        
        # Create output directory
        self.output_dir = "simulation_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"sim_recording_{timestamp}.mp4")
        
        # Initialize video writer
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = None
        
    def start_recording(self):
        """Start recording the simulation"""
        try:
            self.video_writer = cv2.VideoWriter(
                self.video_path,
                self.fourcc,
                self.fps,
                (self.width, self.height)
            )
            self.is_recording = True
            print(f"Started recording to: {self.video_path}")
        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            self.is_recording = False
        
    def capture_frame(self):
        """Capture current frame from PyBullet and add to video"""
        if not self.is_recording or self.video_writer is None:
            return
            
        try:
            # Get the current view from PyBullet
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=self.width,
                height=self.height,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert the RGB array to a format suitable for OpenCV
            rgb_array = np.reshape(rgb_img, (height, width, 4))
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            
            # Write the frame
            self.video_writer.write(bgr_array)
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
        
    def stop_recording(self):
        """Stop recording and release resources"""
        if not self.is_recording:
            return

        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.is_recording = False
                print(f"Finished recording: {self.video_path}")
                
                # Check if the video file exists and has content
                if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 0:
                    temp_path = self.video_path.replace(".mp4", "_temp.mp4")
                    
                    try:
                        # Try to convert to H.264 only if ffmpeg is available
                        if os.system('ffmpeg -version') == 0:  # Check if ffmpeg is installed
                            os.rename(self.video_path, temp_path)
                            os.system(f'ffmpeg -i {temp_path} -c:v libx264 -preset medium -crf 23 {self.video_path} -y')
                            os.remove(temp_path)
                            print("Video converted to H.264 codec for better compatibility")
                        else:
                            print("ffmpeg not found, keeping original video format")
                    except Exception as e:
                        print(f"Could not convert video format: {str(e)}")
                        # If conversion fails, ensure we keep the original file
                        if os.path.exists(temp_path):
                            os.rename(temp_path, self.video_path)
                else:
                    print("No video file was created or file is empty")
                    
        except Exception as e:
            print(f"Error stopping recording: {str(e)}")
        finally:
            self.video_writer = None
            self.is_recording = False
                
    def __del__(self):
        """Ensure video writer is properly released"""
        if self.is_recording:
            self.stop_recording()