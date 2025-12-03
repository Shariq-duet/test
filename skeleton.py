import cv2
import mediapipe as mp

# 1. Setup MediaPipe
# mp_drawing: This helps us draw the lines (the "bones") and dots.
mp_drawing = mp.solutions.drawing_utils
# mp_pose: This is the AI model that detects the body.
mp_pose = mp.solutions.pose

htps='http://192.168.100.24:8080/video'
# 2. Start Video Capture
cap = cv2.VideoCapture(htps)

print("[INFO] Starting Pose Estimation...")

# 3. Initiate the Pose Model
# min_detection_confidence=0.5 means "I need to be 50% sure it's a person before I draw."
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # --- PRE-PROCESSING ---
        # MediaPipe expects RGB images. OpenCV gives us BGR.
        # We must convert it, or the colors will be wrong and detection will fail.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, mark the image as "not writeable" 
        # so we pass it by reference.
        image.flags.writeable = False
        
        # --- THE MAGIC HAPPENS HERE ---
        # This line asks the AI: "Where are the body parts?"
        results = pose.process(image)

        # --- DRAWING ---
        # Convert back to BGR so we can draw on it and show it in OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if any body parts were found
        if results.pose_landmarks:
            # Draw the landmarks (dots) and connections (lines)
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                # Optional: Make the dots and lines look nicer
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Dot color
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Line color
            )

        # Show the result
        cv2.imshow('MediaPipe Feed', image)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()