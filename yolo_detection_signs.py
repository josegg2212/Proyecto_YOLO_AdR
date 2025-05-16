from ultralytics import YOLO

class YoloDetector:
    def __init__(self, modelo_path):

        self.model = YOLO(modelo_path)
    
    def process_image(self, fuente, show = False, save = False):

        # Run inference (detection)
        results = self.model(source=fuente, show=show, save=save, conf=0.5)

        res=results[0]
        
        # Get the detected image
        detected_img = res.plot()

        # Get the bounding boxes coordinates
        bboxes = res.boxes.xyxy.cpu().numpy().astype(int)

        return detected_img , bboxes[0]








