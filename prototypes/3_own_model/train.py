from ultralytics import YOLO
m = YOLO('w/yolo11n.pt')   # COCO-pretrained (everyday photos), never trained on aerial images
m.train(data='trainset/data.yaml', epochs=30, imgsz=640, batch=8, workers=2, device='cpu',
        degrees=180, flipud=0.5, fliplr=0.5, mosaic=1.0, scale=0.3, hsv_v=0.3,
        project='runs', name='cowc_n', exist_ok=True, plots=False, patience=100, seed=0)
