import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8-BFA-YOLO.yaml')
    
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='gpu',
                lr = 0.001,
                optimizer='SGD',
                project='BFA-YOLO/train',
                name='exp',
                )