import cv2
import datetime

def display_detections(frame, detections, display_time=True, wait_time=1):
    """
    Display a frame with detection boxes and timestamp
    
    Parameters:
    frame (numpy.ndarray): The original frame image
    detections (list): List of detection boxes [(x, y, w, h), ...] 
    display_time (bool): Whether to display the current time on the frame
    wait_time (int): Time to wait between frames in milliseconds (controls display speed)
    
    Returns:
    numpy.ndarray: The processed frame with detections and timestamp
    """
    # Make a copy of the frame to avoid modifying the original
    display_frame = frame.copy()
    
    # Draw each detection box
    for (x, y, w, h) in detections:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add current time in the top left corner
    if display_time:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, current_time, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame with detections
    cv2.imshow("Detection Display", display_frame)
    
    # Wait for the specified time and check for key press
    key = cv2.waitKey(wait_time) & 0xFF
    
    # Return the processed frame and key press
    return display_frame, key

def run_display(frames, detection_results, save_output=None, wait_time=25):
    """
    Run the display on a set of frames with their detection results
    
    Parameters:
    frames (list): List of frame images
    detection_results (list): List of detection boxes for each frame
    save_output (str): Path to save the output video (None to not save)
    wait_time (int): Time to wait between frames in milliseconds
    
    Returns:
    None
    """
    # Initialize video writer if saving output
    if save_output:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_output, fourcc, 30.0, (width, height))
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Get detections for this frame (empty list if no detections)
        detections = detection_results.get(i, [])
        
        # Display frame with detections
        processed_frame, key = display_detections(frame, detections, wait_time=wait_time)
        
        # Save the processed frame if required
        if save_output:
            out.write(processed_frame)
        
        
        if key == ord('q'):
            break
    
    
    if save_output:
        out.release()
    cv2.destroyAllWindows()

