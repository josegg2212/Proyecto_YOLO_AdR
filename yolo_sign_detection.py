from ultralytics import YOLO

# Class for Yolo detection
class YoloDetector:
    def __init__(self, modelo_path):

        # Load the YOLO model
        self.model = YOLO(modelo_path)
    
    # Method to process an image and return the detected image and bounding boxes
    def process_image(self, fuente, show = False, save = False):

        # Run inference (detection)
        results = self.model(source=fuente, show=show, save=save, conf=0.5)

        res=results[0]
        
        # Get the detected image
        detected_img = res.plot()

        # Get the bounding boxes coordinates
        bboxes = res.boxes.xyxy.cpu().numpy().astype(int)

        return detected_img , bboxes








