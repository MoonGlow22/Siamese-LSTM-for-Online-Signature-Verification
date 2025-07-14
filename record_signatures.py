###############################################################################
# Signature Capture Application
# Controls:
# z - Start capturing signature
# x - Save the current signature
# c - Convert captured signatures to images
# q - Quit application
###############################################################################

import numpy as np
import cv2
import mediapipe as mp
import math as mt
import os
from pathlib import Path
from timeit import default_timer as timer
import sys
import createImage as ci

# Configuration constants
CAMERA_NUMBER = 1
DIST_MAX = 80
CAM_WIDTH = 2000
CAM_HEIGHT = 1000
SIGNATURE_AREA_P1 = (450, 150)
SIGNATURE_AREA_P2 = (1250, 400)


class SignatureCapture:
    def __init__(self, camera_number=CAMERA_NUMBER):
        """Initialize the signature capture application"""
        self.camera_number = camera_number
        self.coords = []
        self.npCoords = []
        self.getCoord = False
        self.drawSignature = False
        self.camTitle = 'signature press z to start signing'
        self.nxt = 1
        self.person_name = ""
        
        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Initialize camera
        self.setup_camera()
        
    def setup_camera(self):
        "Set up and configure the camera"
        try:
            self.cap = cv2.VideoCapture(self.camera_number)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_number}")
                sys.exit(1)
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        except Exception as e:
            print(f"Error initializing camera: {e}")
            sys.exit(1)
    
    def create_person_directory(self):
        "Create directory for the person's signatures"
        self.person_name = input("Name Surname: ")
        self.save_path = os.path.join("Signatures", self.person_name)
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def handle_key_events(self, key):
        """Handle keyboard inputs"""
        if key == ord('z'):  # Start capturing signature
            cv2.destroyAllWindows()
            self.camTitle = f'put middle and index fingers together, sign 50 cm far from screen, press x to save {self.nxt}'
            self.coords = []
            self.npCoords = []
            self.getCoord = True
            self.drawSignature = True
            self.strt = timer()
            
        elif key == ord('x'):  # Save current signature
            self.getCoord = False
            self.drawSignature = True
            self.save_signature()
            self.camTitle = 'Saved. Press z to get next signature'
            self.nxt += 1
            
        elif key == ord('c'):  # Convert signatures to images
            self.drawSignature = False
            self.getCoord = False
            cv2.destroyAllWindows()
            self.camTitle = 'signature'
            self.convert_signatures_to_images()
            
        elif key == ord('q'):  # Quit application
            cv2.destroyAllWindows()
            self.cap.release()
            sys.exit(0)
    
    def save_signature(self):
        "Save the captured signature to a file"
        with open(os.path.join(self.save_path, f"{self.nxt}.txt"), "w") as txt_file:
            for crd in self.npCoords:
                for line in crd:
                    adjusted_line = [
                        line[0] - SIGNATURE_AREA_P1[0] + 5, 
                        line[1] - SIGNATURE_AREA_P1[1] + 5, 
                        line[2]
                    ]
                    txt_file.write(", ".join(map(str, adjusted_line)) + "\n")
                    cv2.destroyAllWindows()
                txt_file.write("-100, -100\n")
    
    def convert_signatures_to_images(self):
        "Convert all saved signatures to images"
        for itr in range(self.nxt - 1):
            signature_width = SIGNATURE_AREA_P2[0] - SIGNATURE_AREA_P1[0]
            signature_height = SIGNATURE_AREA_P2[1] - SIGNATURE_AREA_P1[1]
            ci.createImageFromPoints2(
                os.path.join(self.save_path, f"{itr+1}.txt"),
                signature_width,
                signature_height,
                self.person_name,
                itr
            )
    
    def process_hand_landmarks(self, hand_landmarks, image_width, image_height):
        """Process hand landmarks to capture signature points"""
        # Get index and middle finger coordinates
        index_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
        index_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
        middle_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
        middle_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
        
        # Calculate distance between fingers
        dist = mt.sqrt(mt.pow(index_x - middle_x, 2) + mt.pow(index_y - middle_y, 2))
        
        # Check if fingers are apart enough and within signature area
        if dist > DIST_MAX and (
            SIGNATURE_AREA_P1[0] < index_x < SIGNATURE_AREA_P2[0] and 
            SIGNATURE_AREA_P1[1] < index_y < SIGNATURE_AREA_P2[1]
        ):
            elapsed_time = timer() - self.strt
            self.coords.append([index_x, index_y, elapsed_time])
        else:
            # If fingers are close or outside area, end current stroke
            if len(self.coords) > 0:
                self.npCoords.append(self.coords)
                self.coords = []
    
    def draw_frame(self, image):
        "Draw signature area and captured points on frame"
        if self.drawSignature:
            # Draw signature area rectangle
            cv2.rectangle(image, SIGNATURE_AREA_P1, SIGNATURE_AREA_P2, (0, 0, 0), 2)
            
            # Draw current stroke
            if len(self.coords) > 0:
                co = [item[0:2] for item in self.coords]
                cv2.polylines(image, [np.int32(co)], False, (255, 0, 0), 2)
            
            # Draw saved strokes
            if len(self.npCoords) > 0:
                for crd in self.npCoords:
                    points = [item[0:2] for item in crd]
                    cv2.polylines(image, [np.int32(points)], False, (255, 0, 0), 2)
    
    def run(self):
        "Main application loop"
        self.create_person_directory()
        
        with self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            while self.cap.isOpened():
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self.handle_key_events(key)
                
                # Read camera frame
                success, image = self.cap.read()
                if not success:
                    continue
                
                # Process image for hand detection
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                
                # Prepare for drawing
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = image.shape
                
                # Draw signature area and points
                self.draw_frame(image)
                
                # Process hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Process hand landmarks for signature if capturing
                        if self.getCoord:
                            self.process_hand_landmarks(
                                hand_landmarks, image_width, image_height)
                
                # Display the frame
                cv2.imshow(self.camTitle, image)
        
        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SignatureCapture(CAMERA_NUMBER)
    app.run()