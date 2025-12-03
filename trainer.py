import cv2
import mediapipe as mp
import numpy as np

htps='http://192.168.100.24:8080/video'

# --- 1. MATH FUNCTION (Same as before) ---
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

# --- 2. SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(htps) # Use 0 for webcam

# SEPARATE COUNTERS
left_counter = 0 
left_stage = None

right_counter = 0
right_stage = None

# --- 3. MAIN LOOP ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Color conversion
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # --- 4. EXTRACT LANDMARKS ---
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- LEFT ARM LOGIC (11, 13, 15) ---
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # --- RIGHT ARM LOGIC (12, 14, 16) ---
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate Angles
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            # Visualize Angles (Left Elbow & Right Elbow)
            cv2.putText(image, str(int(l_angle)), 
                           tuple(np.multiply(l_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, str(int(r_angle)), 
                           tuple(np.multiply(r_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- 5. CURL LOGIC (DUAL) ---
            
            # LEFT ARM
            if l_angle > 160:
                left_stage = "down"
            if l_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1
                
            # RIGHT ARM
            if r_angle > 160:
                right_stage = "down"
            if r_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1
                
        except:
            pass

        # --- 6. DRAW INTERFACE (Split Screen) ---
        
        # Draw LEFT Box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(image, 'LEFT REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(left_counter), (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Draw RIGHT Box (Top Right Corner)
        # We start drawing at x=415 (approx width 640 - 225)
        cv2.rectangle(image, (415,0), (640,73), (245,117,16), -1)
        cv2.putText(image, 'RIGHT REPS', (430,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(right_counter), (425,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Dual Arm AI Trainer', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()