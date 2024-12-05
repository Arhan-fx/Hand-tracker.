import cv2
import mediapipe as mp
import time

def is_left_or_right(hand_landmarks):
    wrist = hand_landmarks.landmark[0]  
    if wrist.x < 0.5:  
        return 'Left'
    else:
        return 'Right'
 
def main():
     
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

   
    prev_time = 0

     
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,   
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                 
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape

                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                 
                result = hands.process(rgb_frame)

                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )

                        
                        hand_type = is_left_or_right(hand_landmarks)

                        
                        if hand_type == 'Right':
                            cv2.putText(frame, "Right Hand", (int(hand_landmarks.landmark[0].x * frame_width),
                                                              int(hand_landmarks.landmark[0].y * frame_height)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)   
                        elif hand_type == 'Left':
                            cv2.putText(frame, "Left Hand", (int(hand_landmarks.landmark[0].x * frame_width),
                                                             int(hand_landmarks.landmark[0].y * frame_height)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)   

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (frame_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # White color for text

                cv2.imshow('Hand Gesture Recognition', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
