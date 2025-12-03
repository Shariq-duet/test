import cv2
import dlib
import winsound
import numpy as np
from scipy.spatial import distance as dist

# --- CONFIGURATION ---
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # If EAR is smaller than this, eye is closed
EYE_ASPECT_RATIO_CONSEC_FRAMES = 60   # How many frames must eyes be closed to trigger alarm?
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


htps='http://192.168.1.105:8080/video'
# --- HELPER FUNCTION: CALCULATE EYE RATIO ---
def eye_aspect_ratio(eye):
    # Vertical landmarks distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Horizontal landmark distance
    C = dist.euclidean(eye[0], eye[3])

    # The EAR Equation
    ear = (A + B) / (2.0 * C)
    return ear

# --- INITIALIZATION ---
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except RuntimeError:
    print(f"[ERROR] Could not find {SHAPE_PREDICTOR_PATH}. Did you download and extract it?")
    exit()

# Indexes for Dlib's 68-point model
(lStart, lEnd) = (42, 48) # Left Eye
(rStart, rEnd) = (36, 42) # Right Eye

print("[INFO] Starting Video Stream...")
cap = cv2.VideoCapture(0) # 0 is usually your webcam

COUNTER = 0
ALARM_ON = False
TOTAL_BLINKS = 0
FRAMES_SINCE_LAST_BLINK = 0
# Assuming 30 FPS, 60 seconds = 1800 frames
LIVENESS_THRESHOLD = 300
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera failed to capture frame!")
        break
    
    # Resize frame (optional, for speed) and convert to Grayscale
    # frame = cv2.resize(frame, (650, 500)) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- NUCLEAR OPTION FIX ---
    gray = np.array(gray, dtype=np.uint8)
    
    # Detect faces
    faces = detector(frame, 0)

    # --- [NEW] Liveness Logic Part 1: Increase Timer ---
    # We count every single frame where a face exists
    if len(faces) > 0:
        FRAMES_SINCE_LAST_BLINK += 1

    for face in faces:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, face)
        
        # Convert dlib's result to a numpy array for easier handling
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        # Extract the left and right eye coordinates
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]

        # Calculate Aspect Ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the EAR together
        avgEAR = (leftEAR + rightEAR) / 2.0

        # --- DRAWING ON SCREEN ---
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # --- [NEW] Liveness Logic Part 2: Check Timer ---
        if FRAMES_SINCE_LAST_BLINK > LIVENESS_THRESHOLD:
            cv2.putText(frame, "FAKE FACE / NO BLINK DETECTED", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            winsound.Beep(2000, 150)
            # You could add a specific sound here if you want

        # --- LOGIC CHECK ---
        if avgEAR < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1

            # If eyes are closed for sufficient frames, sound alarm
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                winsound.Beep(2000, 150)

        else:
            # --- [NEW] Liveness Logic Part 3: Detect the Blink ---
            # Eyes are OPEN now. Was the previous closing a blink?
            # A blink is fast (usually 2 to 10 frames)
            if COUNTER >= 2 and COUNTER < 10:
                TOTAL_BLINKS += 1
                FRAMES_SINCE_LAST_BLINK = 0  # <--- RESET THE TIMER
                print(f"Blink Detected! Total: {TOTAL_BLINKS}")

            # Reset the closed-eye counter 
            COUNTER = 0
            ALARM_ON = False

        # Display EAR on screen for debugging
        cv2.putText(frame, "EAR: {:.2f}".format(avgEAR), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display Blink Info
        cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINKS), (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # Press 'q' to quit
    if key == ord("q"):
        break