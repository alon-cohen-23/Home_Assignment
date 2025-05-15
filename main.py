import cv2
from detector import detect_motion_in_frames
from display_detections import display_detections

# Load your frames
frames_dir = "frames_opencv/dog_video.MP4"
results = detect_motion_in_frames(frames_dir, min_area=500, display=False)

# Convert results to format needed by display function
detection_dict = {}
for frame_num, boxes in results:
    detection_dict[frame_num] = boxes

# Load the original frames
import glob
import os
frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
frames = [cv2.imread(fp) for fp in frame_paths]

# Display frames with detections
from display_detections import run_display
run_display(frames, detection_dict)
