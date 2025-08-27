from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import sys
import os

def main():
    video_path = "./videos/1.mp4"
    # Optional: Set webcam resolution
    # cap = cv2.VideoCapture(0)  # For webcam
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    

    weights_path = "./Yolo-Weights/yolov8l.pt"
    if not os.path.exists(weights_path):
        print(f"Error: YOLO weights '{weights_path}' not found!")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return
    try:
        model = YOLO(weights_path)
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # COCO class names
    classNames = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Initialize FPS 
    prev_frame_time = 0
    frame_count = 0
    print("Starting object detection... Press 'q' to quit or click the close button.")
    
    try:
        while True:
            success, img = cap.read()
        
            if not success:
                print("End of video or failed to read frame")
                break
            # Calculate FPS
            new_frame_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = new_frame_time
            
            # Run YOLO detection
            results = model(img, stream=True, verbose=False)  # verbose=False to reduce console output
            
            detection_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)                        
                        w, h = x2 - x1, y2 - y1
                        # Get confidence and class
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        
                        # Only display detections with confidence > 0.3
                        if conf > 0.3:                    
                            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 255))                                                
                            label = f'{classNames[cls]} {conf}'
                            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), 
                                             scale=0.8, thickness=1, colorR=(255, 0, 255))
                            detection_count += 1
            
            # Display FPS and detection on vdo or image
            fps_text = f'FPS: {int(fps)}'
            detection_text = f'Detections: {detection_count}'
            cvzone.putTextRect(img, fps_text, (10, 50), scale=1, thickness=2, colorR=(0, 255, 0))
            cvzone.putTextRect(img, detection_text, (10, 100), scale=1, thickness=2, colorR=(0, 255, 0))
            
            # Display frame
            cv2.imshow("YOLO Object Detection", img)            
            # Handle key 
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' key or ESC key
                break
            
            if cv2.getWindowProperty("YOLO Object Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"FPS: {int(fps)}, Detections: {detection_count}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Video capture released and windows closed.")

if __name__ == "__main__":
    main()