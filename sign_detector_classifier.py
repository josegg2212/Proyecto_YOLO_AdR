import cv2
import numpy as np


# Import yolo detector class
from yolo_detection_signs import YoloDetector

# Import yolo classifier class
from signs_classification import SignClassifier
# 

# Model weights
DETECTOR_MODEL_PATH = "traffic_sign_model.pt"
CLASSIFIER_MODEL_PATH= "traffic_sign_net_5clases.pth"


# Class for sign detection and classification
class SignDetectorClassifier():
    def __init__(self):

        # Load YOLO detector and classifier
        self.detector = YoloDetector(DETECTOR_MODEL_PATH)
        self.classifier = SignClassifier(CLASSIFIER_MODEL_PATH)

 
    # Process the image file
    def process_image(self,image_path):

        # Read the image
        image = cv2.imread(image_path)

        # Detect the sign
        detected_image , bboxes = self.detect_sign(image_path)

        # Save the detected image
        cv2.imwrite("detected_image.jpg", detected_image)

        i=0
        for bbox in bboxes:
            # Crop the image
            cropped_image = self.crop_image(image , bbox)

            # Read the cropped image and save it
            cv2.imwrite("cropped_image.jpg", cropped_image)
            cropped_image_path = "cropped_image.jpg"

            # Classify the cropped image
            processed_image , pred= self.classifier.process_image(cropped_image_path)

            # Save the processed image
            cv2.imwrite(f"processed_image_{pred}_{i}.jpg", processed_image)

            # Display the processed image
            cv2.imshow("Processed Image", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            i+=1

    # Sign detection using a YOLO model 
    def detect_sign(self, image_path):
       
        # Detect the sign
        detected_image , bbox=self.detector.process_image(image_path)

        return detected_image , bbox

        
    # Crop the sign
    def crop_image(self, image, bbox):
        
        # Crop the image using bounding box
        x1, y1, x2, y2 = bbox
                      
        cropped_image = image[y1:y2 , x1:x2]

        return cropped_image

    # Sign classification using a YOLO model 
    def classify_sign(self, cropped_image_path):
       
        # Classify the sign
        classified_image =self.classifier.process_image(cropped_image_path)
        
        return classified_image


if __name__ == "__main__":
    sign_detector_classifier = SignDetectorClassifier()

    # Process the image and save it
    sign_detector_classifier.process_image("../sign5.png")



    
