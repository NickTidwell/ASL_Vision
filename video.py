import cv2
import mediapipe as mp
import torch
from torchvision import transforms
import json
import argparse
import torch.nn.functional as F
from train_utils import load_model
cap = cv2.VideoCapture(0) # Capture Data from videocam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FpsCounter:
    def __init__(self):
        self.start_time = cv2.getTickCount()
        self.frames = 0
        self.fps = 0
    
    def update(self):
        self.frames += 1
        elapsed_time = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
        self.fps = self.frames / elapsed_time

    def draw(self, frame):
        # Draw FPS on the frame
        fps_text = f"FPS: {self.fps:.2f}"  # Format FPS text to 2 decimal places
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def reset(self):
        self.start_time = cv2.getTickCount()
        self.frames = 0

def baseline_landmarks(frame):
    # Convert the frame to BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform hand landmark detection
    results = hands.process(rgb_frame)
    
    # Draw landmarks on the frame if hand is detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for lm in landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

resize_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL Image
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert PIL Image back to tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])
def annotate_frame(frame):
    hand_tensor = None
    # Convert the frame to BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform hand landmark detection
    results = hands.process(rgb_frame)
    hand_landmarks_list = results.multi_hand_landmarks
    handedness_list = results.multi_handedness
    if not hand_landmarks_list or not handedness_list:
        return hand_tensor
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Convert the hand landmarks to pixel coordinates
        height, width, _ = frame.shape
        landmarks_x = [int(landmark.x * width) for landmark in hand_landmarks.landmark]
        landmarks_y = [int(landmark.y * height) for landmark in hand_landmarks.landmark]

        # Calculate the bounding box around the detected hand
        min_x, min_y = min(landmarks_x), min(landmarks_y)
        max_x, max_y = max(landmarks_x), max(landmarks_y)
        margin = 50
        min_y = min_y - margin
        min_x = min_x - margin
        max_x = max_x + margin
        max_y = max_y + margin

        min_y = max(0, min_y)
        min_x = max(0, min_x)
        # Draw the bounding box around the hand
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        hand_image = frame[min_y:max_y, min_x:max_x]

        hand_tensor = torch.tensor(hand_image).permute(2, 0, 1).float() / 255.0
        rescaled_image_tensor = resize_transform(hand_tensor)

        # Display original and resized images using OpenCV
        cv2.imshow('Original Image', hand_tensor.permute(1, 2, 0).numpy())
        cv2.imshow('Resized Image', rescaled_image_tensor.permute(1, 2, 0).numpy() )
        # Draw handedness (left or right hand) on the image above the bounding box
        text_x = min_x
        text_y = max(0, min_y - 10)  # Adjust the text position above the bounding box
        cv2.putText(frame, f"{handedness.classification[0].label[0]}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (255, 0, 0), 2, cv2.LINE_AA)
        return rescaled_image_tensor

    with open('idx_to_class.json', 'r') as json_file:
        idx_to_class = json.load(json_file)
    return model, idx_to_class


  
def pred_symbol(input, model):
    input = input.unsqueeze(0)
    input = input.to(device)
    out = model(input)
    return torch.argmax(out), torch.max(F.softmax(out)), F.softmax(out)
    
THRESHOLD = .0
fps_counter = FpsCounter()
checkpoint_path = "checkpoints/Simple/Simple"
checkpoint_name = "best_model.pth"

parser = argparse.ArgumentParser(description='Trainer for ASLV Model')
parser.add_argument('--model_type', type=str, default='Simple', help='Name of the model')
parser.add_argument('--num_classes', type=int, default=29, help='Number of classes')
parser.add_argument('--checkpoint_name', type=str, default="best_model.pth", help='Model name')
args = parser.parse_args()

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
model, class_index = load_model(args)
# cap.set(3, 1280)
# cap.set(4, 720)
while True:
    ret, frame = cap.read()
    rescaled_image_tensor = annotate_frame(frame)
    if rescaled_image_tensor != None:
        out, max_out, _ = pred_symbol(rescaled_image_tensor, model)
        if max_out > THRESHOLD:
            print(f"{class_index[str(out.item())]} : {max_out}")

    fps = fps_counter.update()  # Update FPS counter and get current FPS
    fps_counter.draw(frame)
    cv2.imshow('Processed Video', frame)

    # Break the loop if the window is closed (user hits the 'X' button)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Processed Video', cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()