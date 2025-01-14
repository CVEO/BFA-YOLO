import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt')
    model.val(data='dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=2,
              # rect=False,
            #   save_json=True, # if you need to cal coco metrice
              project='BFA-3D/val/yolov8x-1009',
              name='exp',
              )
    

# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('kunnan/yolov8x-C2f-RetBlock-FDPN-TADDH/exp/weights/best.pt')
#     model.val(data='dataset/data-Facade-WHU.yaml',
#               split='val',
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='Facade-WHU/val/yolov8x-C2f-RetBlock-FDPN-TADDH',
#               name='exp',
#               )
    
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('kunnan/yolov8x-C2f-RetBlock-FDPN-TADDH/exp/weights/best.pt')
#     model.val(data='dataset/data-Facade-WHU.yaml',
#               split='val',
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='Facade-WHU/val/yolov8x-C2f-RetBlock-FDPN-TADDH',
#               name='exp',
#               )
    
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('kunnan/yolov8x-C2f-RetBlock-FDPN-TADDH/exp/weights/best.pt')
#     model.val(data='dataset/data-Facade-WHU.yaml',
#               split='val',
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='Facade-WHU/val/yolov8x-C2f-RetBlock-FDPN-TADDH',
#               name='exp',
#               )
    
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('kunnan/yolov8x-C2f-RetBlock-FDPN-TADDH/exp/weights/best.pt')
#     model.val(data='dataset/data-Facade-WHU.yaml',
#               split='val',
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='Facade-WHU/val/yolov8x-C2f-RetBlock-FDPN-TADDH',
#               name='exp',
#               )
    
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('kunnan/yolov8x-C2f-RetBlock-FDPN-TADDH/exp/weights/best.pt')
#     model.val(data='dataset/data-Facade-WHU.yaml',
#               split='val',
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='Facade-WHU/val/yolov8x-C2f-RetBlock-FDPN-TADDH',
#               name='exp',
#               )
    
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('kunnan/yolov8x-C2f-RetBlock-FDPN-TADDH/exp/weights/best.pt')
#     model.val(data='dataset/data-Facade-WHU.yaml',
#               split='val',
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='Facade-WHU/val/yolov8x-C2f-RetBlock-FDPN-TADDH',
#               name='exp',
#               )