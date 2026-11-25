from ultralytics import YOLO
m=YOLO('/home/claude/cars/w/own_cowc_yolo11n.pt')   # start from our COWC model
m.train(data='set/data.yaml',epochs=40,imgsz=640,batch=8,workers=2,device='cpu',
        degrees=180,flipud=0.5,fliplr=0.5,mosaic=1.0,scale=0.3,hsv_v=0.3,
        project='runs',name='ign_ft',exist_ok=True,plots=False,patience=100,seed=0,deterministic=True)
