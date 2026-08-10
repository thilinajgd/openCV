import cv2

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

#Set Resolutioin
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720 pixels
cap.set(cv2.CAP_PROP_FPS, 30)  # Set frames per second to 30


# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#Print actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Webcam resolution set to: {width}x{height} @ {fps} FPS")

#Make adjestable window
cv2.namedWindow('Webcam Feed', cv2.WINDOW_AUTOSIZE)

# Loop to continuously capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Failed to capture frame.")
        break

    #Flip the frame in horizontaly to mirror effect
    flipped_frame =cv2.flip(frame, 1) #1 for horizontal fli, 0 for vertical flip, -1 both horizontal and vertical flip


    #get current window size
    #window_size = cv2.getWindowImageRect('Webcam Feed')
    #resize the frame to fit the window size
    #resized_frame = cv2.resize(flipped_frame, (window_size[2], window_size[3]))

    # Get current window size
    _, _, win_w, win_h = cv2.getWindowImageRect('Webcam Feed')
    # Resize frame to fit the window
    resized_frame = cv2.resize(flipped_frame, (win_w, win_h))


    # Display the frame in a window
    cv2.imshow('Webcam Feed', flipped_frame)

    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()