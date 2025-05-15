import cv2
import datetime
import numpy as np

def blur_region(image, x, y, w, h, blur_method='gaussian', blur_strength=25):
    """
    Apply blurring to a specific region in the image
    
    Parameters:
    image (numpy.ndarray): The image to blur
    x, y, w, h (int): Region coordinates and dimensions
    blur_method (str): Blurring method it can be ('gaussian', 'median', 'pixelate', 'box')
    blur_strength (int): Strength/kernel size of blur effect
    
    Returns:
    numpy.ndarray: The image with the blurred region
    """
    # Make sure coordinates are within image bounds
    x, y = max(0, x), max(0, y)
    right = min(image.shape[1], x + w)
    bottom = min(image.shape[0], y + h)
    
    # Extract the region to blur
    region = image[y:bottom, x:right].copy()
    
    # Apply the selected blur method
    if blur_method == 'gaussian':
        # Ensure kernel size is odd
        k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred_region = cv2.GaussianBlur(region, (k_size, k_size), 0)
    elif blur_method == 'median':
        k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred_region = cv2.medianBlur(region, k_size)
    elif blur_method == 'pixelate':
        # Pixelation effect by downsampling and upsampling
        scale = max(1, blur_strength // 10)  # Adjust scale based on strength
        small = cv2.resize(region, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        blurred_region = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif blur_method == 'box':
        k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred_region = cv2.blur(region, (k_size, k_size))
    else:
        # Default to Gaussian if method not recognized
        k_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred_region = cv2.GaussianBlur(region, (k_size, k_size), 0)
    
    # Replace the region in the original image
    image[y:bottom, x:right] = blurred_region
    
    return image

def display_detections(frame, detections, display_time=True, wait_time=1, blur=True, 
                       blur_method='gaussian', blur_strength=25, draw_boxes=True):
    """
    Display a frame with detection boxes, timestamp, and optional blurring
    
    Parameters:
    frame (numpy.ndarray): The original frame image
    detections (list): List of detection boxes [(x, y, w, h), ...] 
    display_time (bool): Whether to display the current time on the frame
    wait_time (int): Time to wait between frames in milliseconds (controls display speed)
    blur (bool): Whether to blur the detection regions
    blur_method (str): Method for blurring ('gaussian', 'median', 'pixelate', 'box')
    blur_strength (int): Strength of the blur effect (kernel size)
    draw_boxes (bool): Whether to draw bounding boxes around detections
    
    Returns:
    numpy.ndarray: The processed frame with detections and timestamp
    int: Key pressed during display
    """
    # Make a copy of the frame to avoid modifying the original
    display_frame = frame.copy()
    
    # Draw each detection box
    for (x, y, w, h) in detections:
        if blur:
            display_frame = blur_region(display_frame, x, y, w, h, blur_method, blur_strength)
        
        # Draw detection box if requested
        if draw_boxes:
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

