
import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils     # Connecting Keypoints Visuals
mp_pose = mp.solutions.pose                 # Keypoint detection model

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

global counter
counter = 0    # Storage for count of bicep curls

def pushup_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0     
    stage = None     # Stage which stores hand position(Either UP or DOWN)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        def rescale_frame(frame, percent=100):
            width = int(frame.shape[1] * percent/ 100)
            height = int(frame.shape[0] * percent/ 100)
            dim = (width, height)
            return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time

            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic 
                if angle_L > 160 and angle_R > 160:
                    stage = "down"
                if angle_L < 30 and angle_R < 30 and stage == 'down':
                    stage = "up"
                    # global counter
                    counter += 1
                    print(counter)
                        
            except:
                pass
                        
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def dumbbellflys_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    flag = None     # Flag which stores hand position(Either UP or DOWN)
       
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time

            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(hip_L, shoulder_L, elbow_L)
                angle_R = calculate_angle(hip_R, shoulder_R, elbow_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(shoulder_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(shoulder_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L < 80 and angle_R < 80:
                    flag = 'down'
                if angle_L > 90 and angle_R > 90 and flag=='down':
                    # global counter
                    counter += 1
                    flag = 'up'
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, flag, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def dumbbellpullover_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    stage = None     # Stage which stores hand position(Either UP or DOWN)
    global counter
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Landmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # Recolor the image BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(hip_L, shoulder_L, elbow_L)
                angle_R = calculate_angle(hip_R, shoulder_R, elbow_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(shoulder_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(shoulder_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160 and angle_R > 160:
                    stage = 'up'
                if angle_L < 50 and angle_R < 50 and stage == 'up':
                    counter += 1
                    stage = 'down'
                    print("Count: ",counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(246,118,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(246,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def flatbenchpress_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None    # Stage which stores hand position(Either UP or DOWN)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)

        while cap.isOpened():
            
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left chest
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Right chest
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]          
                
                # Calculate angle
                angle1 = calculate_angle(L_hip, R_shoulder, R_elbow)
                angle2 = calculate_angle(R_shoulder, L_shoulder, L_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(R_shoulder, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(L_shoulder, [808, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 140 and angle2 > 140:
                    stage = "DOWN"
                if angle1 < 100 and angle2 < 100 and stage =='DOWN':
                    stage="UP"
                    
                    counter +=1
                    print(counter)     # printing the counter
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (266,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            
def inclinebenchpress_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None    # Stage which stores hand position(Either UP or DOWN)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left chest
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Right chest
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]          
                
                # Calculate angle
                angle1 = calculate_angle(L_hip, R_shoulder, R_elbow)
                angle2 = calculate_angle(R_shoulder, L_shoulder, L_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(R_shoulder, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(L_shoulder, [808, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 140 and angle2 > 140:
                    stage = "UP"
                if angle1 < 100 and angle2 < 100 and stage =='UP':
                    stage="DOWN"
                    
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (266,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def declinebenchpress_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None    # Stage which stores hand position(Either UP or DOWN)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left chest
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Right chest
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]          
                
                # Calculate angle
                angle1 = calculate_angle(L_hip, R_shoulder, R_elbow)
                angle2 = calculate_angle(R_shoulder, L_shoulder, L_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(R_shoulder, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(L_shoulder, [808, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 140 and angle2 > 140:
                    stage = "UP"
                if angle1 < 100 and angle2 < 100 and stage =='UP':
                    stage="DOWN"
                    
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (266,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
                       
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def alternatedumbbellpress_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0
    counter1 = 0 
    stage1 = None   
    counter2 = 0
    stage2= None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left shoulder
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                # Right shoulder
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Calculate angle
                angle2 = calculate_angle(L_hip, L_shoulder, L_elbow)
                angle1 = calculate_angle(R_hip, R_shoulder, R_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_shoulder, [1008, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_shoulder, [900, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                                
                # Curl counter logic
                if angle1 > 140 :
                    stage1 = "Down"
                if angle1 < 100 and stage1 =='Down':
                    stage1 = "UP"
                    counter1 +=1
                    
                if angle2 > 140:
                    stage2 = "Down"
                if angle2 < 100 and stage2 =='Down':
                    stage2 ="UP"
                    counter2 +=1

                counter = counter1 + counter2
                print(counter)

            except:
                pass
            
            # Render curl counter
            # Setup status box        
            cv2.rectangle(image, (0,0), (260,70), (245,117,16),-1)  # creating a counter box1
            cv2.rectangle(image, (730,0), (1100,70), (245,117,16),-1)  # creating a counter box2
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter1), 
                        (10,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (730,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter2), 
                        (730,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage1, 
                        (100,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (830,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage2, 
                        (830,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def deadlift_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left biceps
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                # Right biceps
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(R_shoulder, R_hip, R_knee)
                angle2 = calculate_angle(L_shoulder, L_hip, L_knee)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(R_hip, [1008, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(L_hip, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 170 and angle2 > 170:
                    stage = "UP"
                if angle1 < 150 and angle2 < 150 and stage =='UP':
                    stage="Down"
                    
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def pullup_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    stage = None     # Stage which stores hand position(Either UP or DOWN)
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160 and angle_R > 160:
                    stage = 'down'
                    
                if angle_L < 90 and angle_R < 90 and stage=='down':
                    stage = 'up'
                    counter += 1
                    print(counter)                    
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
        
def barbellbentoverrow_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left biceps
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                L_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Right biceps
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_shoulder, L_elbow, L_wrist)
                angle2 = calculate_angle(R_shoulder, R_elbow, R_wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_elbow, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_elbow, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 160 and angle2 > 160:
                    stage = "down"
                if angle1 < 95 and angle2 < 95 and stage =='down':
                    stage="up"
                    
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
           
def seatedrows_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left biceps
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                L_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Right biceps
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_shoulder, L_elbow, L_wrist)
                angle2 = calculate_angle(R_shoulder, R_elbow, R_wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_elbow, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_elbow, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 170 and angle2 > 170:
                    stage = "Open"
                if angle1 < 100 and angle2 < 100 and stage =='Open':
                    stage="Close"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def dumbbellbentoverrow_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0

    counter1 = 0 
    stage1 = None

    counter2 = 0
    stage2= None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            # cv2.imshow('Mediapipe Feed', image)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left biceps
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                L_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Right biceps
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_shoulder, L_elbow, L_wrist)
                angle2 = calculate_angle(R_shoulder, R_elbow, R_wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_elbow, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_elbow, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                            
                # Curl counter logic
                if angle1 > 160 :
                    stage1 = "Down"
                if angle1 < 30 and stage1 =='Down':
                    stage1 = "UP"
                    counter1 +=1
                    
                if angle2 > 160:
                    stage2 = "Down"
                if angle2 < 30 and stage2 =='Down':
                    stage2 ="UP"
                    counter2 +=1
                counter = counter1 + counter2
                print(counter)

            except:
                pass
            
            # Render curl counter
            # Setup status box        
            cv2.rectangle(image, (0,0), (260,70), (245,117,16),-1)  # creating a counter box1
            cv2.rectangle(image, (730,0), (1100,70), (245,117,16),-1)  # creating a counter box2
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter1), 
                        (10,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (730,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter2), 
                        (730,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage1, 
                        (100,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (830,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage2, 
                        (830,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def doubledumbbellshoulderpress_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left shoulder
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                # Right shoulder
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_hip, L_shoulder, L_elbow)
                angle2 = calculate_angle(R_hip, R_shoulder, R_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_shoulder, [1008, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_shoulder, [900, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 150 and angle2 > 150:
                    stage = "up"
                if angle1 < 80 and angle2 < 80 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)
                    
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def shoulderlateralraise_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    stage = None     # Flag which stores hand position(Either UP or DOWN)
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(hip_L, shoulder_L, elbow_L)
                angle_R = calculate_angle(hip_R, shoulder_R, elbow_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(shoulder_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(shoulder_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L < 30 and angle_R < 30:
                    stage = 'down'
                    
                if angle_L > 90 and angle_R > 90 and stage=='down':
                    counter += 1
                    stage = 'up'
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def shoulderuprightrows_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left shoulder
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                # Right shoulder
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_hip, L_shoulder, L_elbow)
                angle2 = calculate_angle(R_hip, R_shoulder, R_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_shoulder, [1008, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_shoulder, [900, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 90 and angle2 > 90:
                    stage = "up"
                if angle1 < 30 and angle2 < 30 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def alternatefrontraises_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables/
    global counter
    counter = 0

    counter1 = 0 
    stage1 = None

    counter2 = 0
    stage2= None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left shoulder
                L_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                # Right shoulder
                R_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                # Calculate angle
                angle2 = calculate_angle(L_hip, L_shoulder, L_elbow)
                angle1 = calculate_angle(R_hip, R_shoulder, R_elbow)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_shoulder, [1008, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_shoulder, [900, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 80 :
                    stage1 = "Up"
                if angle1 < 20 and stage1 =='Up':
                    stage1 = "Down"
                    counter1 +=1
                
                if angle2 > 80:
                    stage2 = "Up"
                if angle2 < 20 and stage2 =='Up':
                    stage2 ="Down"
                    counter2 +=1

                counter = counter1 + counter2
                print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box        
            cv2.rectangle(image, (0,0), (260,70), (245,117,16),-1)  # creating a counter box1
            cv2.rectangle(image, (730,0), (1100,70), (245,117,16),-1)  # creating a counter box2
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter1), 
                        (10,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'REPS', (730,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter2), 
                        (730,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage1, 
                        (100,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (830,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage2, 
                        (830,62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def tricepscablepushdown_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left Triceps 
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                L_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Right Triceps
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_shoulder, L_elbow, L_wrist)
                angle2 = calculate_angle(R_shoulder, R_elbow, R_wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_elbow, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_elbow, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 160 and angle2 >160:
                    stage = "down"
                if angle1 < 70 and angle2 < 70 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def standingoverheaddumbbell_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    stage = None     # Flag which stores hand position(Either UP or DOWN)
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Curl counter logic
                if angle_L > 140 and angle_R > 140:
                    stage = "UP"
                if angle_L < 90 and angle_R < 90 and stage =='UP':
                    stage="Down"
                    counter +=1
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def armkickback_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left Triceps 
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                L_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Right Triceps
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(L_shoulder, L_elbow, L_wrist)
                angle2 = calculate_angle(R_shoulder, R_elbow, R_wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(L_elbow, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_elbow, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 160 and angle2 > 160:
                    stage = "Open"
                if angle1 < 100 and angle2 < 100 and stage =='Open':
                    stage="Close"
                    
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def benchdips_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    stage = None     # Flag which stores hand position(Either UP or DOWN)
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160 and angle_R > 160:
                    stage = 'up'
                    
                if angle_L < 95 and angle_R < 95 and stage=='up':
                    stage = 'down'
                    counter += 1
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def doublebicep_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    stage = None     # Flag which stores hand position(Either UP or DOWN)
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160 and angle_R > 160:
                    stage = 'down'
                    
                if angle_L < 30 and angle_R < 30 and stage=='down':
                    stage = 'up'
                    counter += 1
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def singlearmbicep_exerises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0
    stage_L = None     # Flag which stores hand position(Either UP or DOWN)
    count_L = 0       # Storage for count of bicep curls
    stage_R = None
    count_R = 0

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Landmark detection model instance
       # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # Recolor the image BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160:
                    stage_L = 'down'
                if angle_L < 30 and stage_L == 'down':
                    count_L += 1
                    stage_L = 'up'
                if angle_R > 160:
                    stage_R = 'down'
                if angle_R < 30 and stage_R == 'down':
                    count_R += 1
                    stage_R = 'up'

                counter = count_L + count_R
                print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count_L), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage_L, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (775,1), (1000,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (790,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count_R), 
                        (785,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (860,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage_R, 
                        (865,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(246,118,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(246,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def barbellbicep_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    stage = None     # Flag which stores hand position(Either UP or DOWN)
    counter = 0       # Storage for count of bicep curls

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Lnadmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates of Right side
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160 and angle_R > 160:
                    stage = 'down'
                    
                if angle_L < 30 and angle_R < 30 and stage=='down':
                    stage = 'up'
                    counter += 1
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def squat_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_L = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                knee_R = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculate angle
                angle_L = calculate_angle(hip_R, hip_L, knee_L)
                angle_R = calculate_angle(hip_L, hip_R, knee_R)

                # Visualize angle
                cv2.putText(image, str(angle_L),
                            tuple(np.multiply(hip_L, [1000, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, str(angle_R),
                            tuple(np.multiply(hip_R, [800, 600]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                            )
                
                # Curl counter logic
                if angle_L < 100 and angle_R < 100:
                    stage = "up"
                if angle_L > 102 and angle_R > 102 and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (75, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def lunge_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0
    stage_L = None     # Flag which stores hand position(Either UP or DOWN)
    count_L = 0       # Storage for count of bicep curls
    stage_R = None
    count_R = 0

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:      # Landmark detection model instance
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))

            # Recolor the image BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Convert BGR frame to RGB
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       # Get landmarks of the object in frame from the model

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Convert RGB back to BGR

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of Left side
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_L = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                angle_L = [landmarks[mp_pose.PoseLandmark.LEFT_ANGLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANGLE.value].y]
                
                # Get coordinates of Right side
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_R = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                angle_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ANGLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANGLE.value].y]
                
                # Calculate angle for both sides
                angle_L = calculate_angle(hip_L, knee_L, angle_L)
                angle_R = calculate_angle(hip_R, knee_R, angle_R)
                
                # Visualize angle for both sides
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(knee_L, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(knee_R, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # Counter 
                if angle_L > 160:
                    stage_L = 'up'
                if angle_L < 50 and stage_L == 'up':
                    count_L += 1
                    stage_L = 'down'
                if angle_R > 160:
                    stage_R = 'up'
                if angle_R < 50 and stage_R == 'up':
                    count_R += 1
                    stage_R = 'down'
                    counter = count_L + count_R
                    print(counter)
                
            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count_L), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (75,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage_L, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (775,1), (1000,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (790,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(count_R), 
                        (785,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (860,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage_R, 
                        (865,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(246,118,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(246,66,230), thickness=2, circle_radius=2) 
                                    )
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def calf_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left HIP
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                LEFT_FOOT_INDEX = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                
                # Right HIP
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                RIGHT_FOOT_INDEX = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX)
                angle2 = calculate_angle(RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(LEFT_ANKLE, [1000, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(RIGHT_ANKLE, [800, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 < 160 and angle2  < 160 :
                    stage = "down"
                if angle1 > 170 and angle2 > 170 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            
def glutealmuscles_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                # Left HIP
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                
                # Right HIP
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                # Calculate angle
                angle1 = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
                angle2 = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(LEFT_HIP, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(RIGHT_HIP, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle1 > 170 and angle2 > 170:
                    stage = "down"
                if angle1 < 120 and angle2 < 120 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (255,70), (245,117,16),-1)  # creating a counter box
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def bridgeyoga_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                ###################################################################################################################################
                
                #--->   Get coordinates  <---#
                
                ############################
                # Left HIP(angle1)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ############################
                # Right HIP(angle2)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ############################
                # Left KNEE(angle3)
                RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ############################
                # Right KNEE(angle4)
                RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                ############################
                # Left Elbow(angle5)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ############################
                # Right Elbow(angle6)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                ############################
                # Left Shoulder(angle7)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ############################
                # Right Shoulder(angle8)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                ################################################################################################################################
                
                # Calculate angle
                angle1 = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
                angle2 = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
                angle3 = calculate_angle(RIGHT_ANKLE, LEFT_KNEE, LEFT_HIP)
                angle4 = calculate_angle(RIGHT_ANKLE, RIGHT_KNEE, RIGHT_HIP)
                angle5 = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
                angle6 = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
                angle7 = calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
                angle8 = calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
                ######################################################################################################################################
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(LEFT_HIP, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(RIGHT_HIP, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle3), 
                            tuple(np.multiply(LEFT_KNEE, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle4), 
                            tuple(np.multiply(RIGHT_KNEE, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle5), 
                            tuple(np.multiply(LEFT_ELBOW, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle6), 
                            tuple(np.multiply(RIGHT_ELBOW, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
    
                ##########################################################################################################################
                
                # Curl counter logic
                
                if angle3 and angle4 > 160 and angle3 and angle4 < 90:
                    if angle3 and angle4 < 60:                                 # in (list(range(50,80))):  # For legs posture
                        if angle5 and angle6 > 170:                            # For arm's posture 
                            if angle1 > 170 and angle2 > 170:                  # For Glutes 
                                stage = "UP"                                         #:
                            if angle1 < 120 and angle2 < 120 and stage =='UP':       #:
                                stage="Down"                                   # till here
                                counter +=1
                                print(counter)
                        else:
                            stage="Put your Arm's straight"
                    else:
                        stage="Wrong posture(Legs)"
                else:
                    stage="Lay down for Bridge pose"
                    
                    
            except:
                pass
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def downwarddogpose_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                ###################################################################################################################################
                
                #--->   Get coordinates  <---#
                
                ############################
                # Left HIP(angle1)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ############################
                # Right HIP(angle2)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ############################
                # Left KNEE(angle3)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ############################
                # Right KNEE(angle4)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ############################
                # Left Elbow(angle5)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                ############################
                # Right Elbow(angle6)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                ############################
                # Left Shoulder(angle7)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                ############################
                # Right Shoulder(angle8)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                ################################################################################################################################
                
                # Calculate angle
                angle1 = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)           #
                angle2 = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)              # Knee
                angle3 = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)        # 
                angle4 = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)           # Hip
                angle5 = calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)       #
                angle6 = calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)          # Shoulder
                angle7 = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)     #
                angle8 = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)        # Elbow
                
                ######################################################################################################################################
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(RIGHT_KNEE, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(LEFT_KNEE, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle3), 
                            tuple(np.multiply(RIGHT_HIP, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle4), 
                            tuple(np.multiply(LEFT_HIP, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle5), 
                            tuple(np.multiply(RIGHT_SHOULDER, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle6), 
                            tuple(np.multiply(LEFT_SHOULDER, [950, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle7), 
                            tuple(np.multiply(RIGHT_ELBOW, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle8), 
                            tuple(np.multiply(LEFT_ELBOW, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                ##########################################################################################################################
                
                # Curl counter logic
                
                if angle1 and angle2 > 150 and angle5 and angle6 < 20:                  # For Seeing that user is standing right.
                    if angle1 and angle2 > 170:                                         # For straigt KNEE'S.
                        if angle7 and angle8 > 140:                                     # For Straight Arms.
    #                         if angle5 and angle6 > 140:                                 # For Straight Shoulder's.
                            if angle3 > 170 and angle4 > 170:                           # For Straight HIP
                                stage = "UP"                                         
                            if angle3 < 150 and angle4 < 150 and stage =='UP':          # For Bend HIP
                                stage="Down"                                   
                                counter +=1
                                print(counter)
    #                         else:
    #                             stage="Straight your Shoulder"  
                        else:
                            stage="Straight your Arms"
                    else:
                        stage="Straigt your KNEE'S"
                else:
                    stage="Stand Straight for pose"
                
                    
                    
            except:
                pass
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def cobrapose_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                ###################################################################################################################################
                
                #--->   Get coordinates  <---#
                
                ############################
                # Left HIP(angle1)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ############################
                # Right HIP(angle2)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ############################
                # Left KNEE(angle3)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ############################
                # Right KNEE(angle4)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ############################
                # Left Elbow(angle5)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                ############################
                # Right Elbow(angle6)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                ############################
                # Left Shoulder(angle7)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                ############################
                # Right Shoulder(angle8)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                ################################################################################################################################
                
                # Calculate angle
                angle1 = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)           #
                angle2 = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)              # Knee
                angle3 = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)        # 
                angle4 = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)           # Hip
                angle5 = calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)       #
                angle6 = calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)          # Shoulder
                angle7 = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)     #
                angle8 = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)        # Elbow
                
                ######################################################################################################################################
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(RIGHT_KNEE, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(LEFT_KNEE, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle3), 
                            tuple(np.multiply(RIGHT_HIP, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle4), 
                            tuple(np.multiply(LEFT_HIP, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle5), 
                            tuple(np.multiply(RIGHT_SHOULDER, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle6), 
                            tuple(np.multiply(LEFT_SHOULDER, [950, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle7), 
                            tuple(np.multiply(RIGHT_ELBOW, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle8), 
                            tuple(np.multiply(LEFT_ELBOW, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                ##########################################################################################################################
                
                # Curl counter logic
                
                if angle1 and angle2 > 150 and angle5 and angle6 < 20:                  # For Seeing that user is standing right.
                    if angle1 and angle2 > 170:                                         # For straigt KNEE'S.
                        if angle7 and angle8 > 140:                                     # For Straight Arms.
    #                         if angle5 and angle6 > 140:                                 # For Straight Shoulder's.
                            if angle3 > 170 and angle4 > 170:                           # For Straight HIP
                                stage = "UP"                                         
                            if angle3 < 150 and angle4 < 150 and stage =='UP':          # For Bend HIP
                                stage="Down"                                   
                                counter +=1
                                print(counter)
    #                         else:
    #                             stage="Straight your Shoulder"  
                        else:
                            stage="Straight your Arms"
                    else:
                        stage="Straigt your KNEE'S"
                else:
                    stage="Stand Straight for pose"
                
                    
                    
            except:
                pass
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            
def trianglepose_exercises():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    global counter
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        # Countdown for exercise
        countdown = 5
        start_time = time.time()
        while(countdown > 0):
            current_time = time.time()
            if current_time - start_time >= 1:
                countdown -= 1
                start_time = current_time
        
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            cv2.rectangle(frame, (0,0), (535,40), (0,0,0), -1)
            image = cv2.putText(frame, f"Ready to Workout in {countdown} seconds", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            k = cv2.waitKey(1)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1000, 620))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                ###################################################################################################################################
                
                #--->   Get coordinates  <---#
                
                ############################
                # Left HIP(angle1)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                ############################
                # Right HIP(angle2)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                ############################
                # Left KNEE(angle3)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ############################
                # Right KNEE(angle4)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ############################
                # Left Elbow(angle5)
                RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                ############################
                # Right Elbow(angle6)
                LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                ############################
                # Left Shoulder(angle7)
                RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                ############################
                # Right Shoulder(angle8)
                LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                ################################################################################################################################
                
                # Calculate angle
                angle1 = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)           #
                angle2 = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)              # Knee
                angle3 = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)        # 
                angle4 = calculate_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)           # Hip
                angle5 = calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)       #
                angle6 = calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)          # Shoulder
                angle7 = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)     #
                angle8 = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)        # Elbow
                
                ######################################################################################################################################
                
                # Visualize angle
                cv2.putText(image, str(angle1), 
                            tuple(np.multiply(RIGHT_KNEE, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(LEFT_KNEE, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle3), 
                            tuple(np.multiply(RIGHT_HIP, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle4), 
                            tuple(np.multiply(LEFT_HIP, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle5), 
                            tuple(np.multiply(RIGHT_SHOULDER, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle6), 
                            tuple(np.multiply(LEFT_SHOULDER, [950, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle7), 
                            tuple(np.multiply(RIGHT_ELBOW, [1020, 600]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(angle8), 
                            tuple(np.multiply(LEFT_ELBOW, [870, 605]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                ##########################################################################################################################
                
                # Curl counter logic
                
                if angle1 and angle2 > 150 and angle5 and angle6 < 20:                  # For Seeing that user is standing right.
                    if angle1 and angle2 > 170:                                         # For straigt KNEE'S.
                        if angle7 and angle8 > 140:                                     # For Straight Arms.
    #                         if angle5 and angle6 > 140:                                 # For Straight Shoulder's.
                            if angle3 > 170 and angle4 > 170:                           # For Straight HIP
                                stage = "UP"                                         
                            if angle3 < 150 and angle4 < 150 and stage =='UP':          # For Bend HIP
                                stage="Down"                                   
                                counter +=1
                                print(counter)
    #                         else:
    #                             stage="Straight your Shoulder"  
                        else:
                            stage="Straight your Arms"
                    else:
                        stage="Straigt your KNEE'S"
                else:
                    stage="Stand Straight for pose"
                
                    
                    
            except:
                pass
            
            # Rep data
            cv2.putText(image, 'REPS', (10,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (100,15),   # stage = up & down counter
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (100,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,225), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0,225,0), thickness=2, circle_radius=2) 
                                    )               
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

def get_counter():
    return counter