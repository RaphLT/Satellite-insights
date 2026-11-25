"""Test on Wazemmes (never used in training). Truth = team's hand labels.
A detection is right if within 1.5 m of a labelled car (one-to-one, Hungarian).
Reports TP / FP / FN (no true negatives exist when counting objects), precision, recall,
for the whole square and inside the OpenStreetMap street/car-park mask only."""
import sys, json, numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
GSD=0.20; TGT=0.10; S=640; R=1.5/GSD
picks=[(2240,2560),(320,1920),(320,1280),(1280,1600),(960,640),(640,0)]
mask=np.load('/home/claude/waz/street_mask.npy')
def detect(model,img,conf):
    from PIL import Image as I
    sc=GSD/TGT; big=np.asarray(I.fromarray(img).resize((round(img.shape[1]*sc),round(img.shape[0]*sc)),I.BICUBIC))
    H,W=big.shape[:2]; out=[]
    for y in list(range(0,max(H-S,0)+1,544))+[max(H-S,0)]:
        for x in list(range(0,max(W-S,0)+1,544))+[max(W-S,0)]:
            r=model.predict(np.ascontiguousarray(big[y:y+S,x:x+S][...,::-1]),imgsz=S,conf=conf,verbose=False)[0]
            for (cx,cy,w,h),s in zip(r.boxes.xywh.cpu().numpy(),r.boxes.conf.cpu().numpy()): out.append((x+cx,y+cy,s))
    d=np.array(out) if out else np.zeros((0,3)); d[:,:2]/=sc
    keep=[];  # merge duplicates within 1.5 m
    for i in np.argsort(-d[:,2]):
        if all(np.hypot(*(d[i,:2]-d[j,:2]))>R for j in keep): keep.append(i)
    return d[keep]
def match(p,t):
    if len(p)==0 or len(t)==0: return 0
    D=np.hypot(p[:,None,0]-t[None,:,0],p[:,None,1]-t[None,:,1]); r,c=linear_sum_assignment(D); return int((D[r,c]<=R).sum())
def run(wpath,conf,truth_dir):
    m=YOLO(wpath); res={'all':[0,0,0],'streets':[0,0,0]}
    for k,(x0,y0) in enumerate(picks):
        n=f'wazemmes_{k+1}'; img=np.asarray(Image.open(f'/home/claude/waz/label/test_wazemmes/{n}.jpg').convert('RGB'))
        L=np.loadtxt(f'{truth_dir}/{n}.txt',ndmin=2); t=L[:,1:3]*S
        p=detect(m,img,conf)[:,:2]
        for key,sel in [('all',lambda a:np.ones(len(a),bool)),('streets',lambda a:mask[np.clip(a[:,1].astype(int)+y0,0,4999),np.clip(a[:,0].astype(int)+x0,0,4999)] if len(a) else np.zeros(0,bool))]:
            pp=p[sel(p)]; tt=t[sel(t)]; tp=match(pp,tt)
            res[key][0]+=tp; res[key][1]+=len(pp)-tp; res[key][2]+=len(tt)-tp
    out={}
    for k,(tp,fp,fn) in res.items(): out[k]=dict(TP=tp,FP=fp,FN=fn,precision=round(tp/max(tp+fp,1),3),recall=round(tp/max(tp+fn,1),3))
    return out
if __name__=='__main__':
    truth=sys.argv[1]; rows={}
    for name,w,conf in [('before (COWC only)','/home/claude/cars/w/own_cowc_yolo11n.pt',0.25),('after (fine-tuned on Englos IGN)','runs/ign_ft/weights/best.pt',0.25)]:
        rows[name]=run(w,conf,truth); print(name,json.dumps(rows[name]))
    json.dump(rows,open(f'eval_{truth.split("/")[-1]}.json','w'),indent=1)
