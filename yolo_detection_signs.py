from ultralytics import YOLO

# Load pre-trained model weights
model = YOLO('/ultralytics/yolo_share/training/first_train_signs_11/weights/best.pt')

# Source to be analyzed
source='https://previews.123rf.com/images/anmbph/anmbph1610/anmbph161000033/66064530-varios-diferentes-se%C3%B1ales-de-tr%C3%A1fico-de-cerca-en-una-calle-de-la-ciudad.jpg'

# Run interference (detection)
results = model(source=source, show=True)

