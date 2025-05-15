import cv2
import os
import time

frames_dir = 'frames_opencv'


def stream_video (video_path: str,
                  min_width = 640,
                  min_height=360, max_video_size_gb = 1):
    
    if not os.path.exists(video_path):
       raise FileNotFoundError(f"{video_path} not found.")
    
    filename = os.path.basename(video_path)
    output_path = os.path.join(frames_dir, filename)
    
    # Checks if the video was already converted to frames
    if os.path.isdir(output_path) and os.listdir(output_path):
        print(f"[Info] Frames already exist in {output_path}. Skipping extraction.")
        return 
    
    os.makedirs(output_path, exist_ok=True)
    
    # Calculate the video size and Raise an error if the video is too large
    size_bytes = os.path.getsize('dog_video.MP4')
    size_gb = size_bytes / (1024 ** 3)
    
    if size_gb > max_video_size_gb:
        raise ValueError ("Your video is too large, you can either resize it, cut it's length, lower it's fps or change max_video_size_gb.")
        

    # Raise errors if the video can't be opened corectly    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("FPS is zero — unsupported or corrupted file.")    
    
    idx = 0
    consecutive_skipped = 0
    max_consecutive_skips = 4
    max_retries_per_frame = 3
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        success, frame = cap.read()
        
        if not success:
            if idx < frame_count:
                retry_count = 0
                backoff = 1
                # Adds exponential backoff if a frame fails 
                while retry_count < max_retries_per_frame:
                    print(f"[Warning] Failed to read frame {idx}. Retrying in {backoff} second(s)...")
                    time.sleep(backoff)
                    retry_count += 1
                    backoff *= 2  # exponential backoff
                    success, frame = cap.read()
                    if success:
                        break
                
                if not success:
                    print(f"[Error] Skipping frame {idx} after {max_retries_per_frame} retries.")
                    consecutive_skipped += 1
                    if consecutive_skipped >= max_consecutive_skips:
                        raise RuntimeError("Too many consecutive frame failures. Aborting.")
                    idx += 1
                    continue
                else:
                    consecutive_skipped = 0
            else:
                break
        
        # Only try to save if we have a valid frame
        if success:
            
            # Save the frame
            cv2.imwrite(os.path.join(output_path, f"frame_{idx:05d}.jpg"), frame)
            idx += 1
            if idx % 100 == 0 and idx > 0:
                print(f"Processed {idx} frames, only {frame_count-idx} left.")
        
    cap.release()
    return output_path 
    



if __name__ == "__main__":
    stream_video('dog_video.MP4')

   
























