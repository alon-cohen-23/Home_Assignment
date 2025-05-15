# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import os
import glob

def detect_motion_in_frames(frames_directory, min_area=500, display=True):
    """
    Process a directory of frame images to detect motion
    
    Parameters:
    frames_directory (str): Directory containing frame images (should be named in sequence)
    min_area (int): Minimum contour area to be considered motion
    display (bool): Whether to display the processed frames
    
    Returns:
    list: Frames where motion was detected [(frame_number, bounding_boxes), ...]
    """
    # Get all frame images and sort them
    frame_paths = glob.glob(os.path.join(frames_directory, "frame_*.jpg"))
    frame_paths.sort()
    
    if len(frame_paths) == 0:
        print("No frames found in the directory.")
        return []
    
    # Initialize the first frame in the sequence
    firstFrame = None
    results = []
    
    for i, frame_path in enumerate(frame_paths):
        # Initialize the occupied/unoccupied text
        text = "Unoccupied"
        
        # Read the current frame
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Could not read frame: {frame_path}")
            continue
        
        # Resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # If the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue
        
        # Compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes, then find contours
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        boxes = []
        
        # Loop over the contours
        for c in cnts:
            # If the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
                
            # Compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"
        
        # If motion was detected in this frame, add it to results
        if text == "Occupied":
            frame_number = i
            results.append((frame_number, boxes))
            
        if display:
            # Draw the text and timestamp on the frame
            cv2.putText(frame, f"Room Status: {text}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {i} - {os.path.basename(frame_path)}",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                
            # Show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)
            key = cv2.waitKey(25) & 0xFF  # Slight delay to control "playback" speed
            
            # If the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break
    
    # Close any open windows
    if display:
        cv2.destroyAllWindows()
        
    return results

def save_results(results, output_path="motion_results.txt"):
    """Save motion detection results to a file"""
    with open(output_path, 'w') as f:
        f.write(f"Motion detected in {len(results)} frames\n")
        for frame_num, boxes in results:
            f.write(f"Frame {frame_num}: {len(boxes)} objects detected\n")
            for i, (x, y, w, h) in enumerate(boxes):
                f.write(f"  Object {i+1}: x={x}, y={y}, width={w}, height={h}\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    results = detect_motion_in_frames("frames_opencv/dog_video.MP4")
    save_results = save_results(results)
