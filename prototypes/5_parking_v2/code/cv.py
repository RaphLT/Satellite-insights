"""3-fold cross-validation on the 6 hand-labelled Wazemmes squares (seed 0).
Folds: {1,2}, {3,4}, {5,6}. Each fold: train on the recipe's data WITHOUT that fold, predict the fold.
Recipes:
  englos+waz : COWC model fine-tuned on Englos (4 squares) + the 4 other Wazemmes squares
  waz        : COWC model fine-tuned on the 4 other Wazemmes squares only
  englos+waz3: as englos+waz but Wazemmes tiles repeated 3x (weights the city streets more)
Predictions are saved at conf 0.05 so the threshold can be chosen afterwards without touching the test fold.
"""
import os, sys, shutil, json, numpy as np
from PIL import Image
sys.path.insert(0, '/home/claude/ign')
import eval_waz as E
from ultralytics import YOLO

WAZ_GSD=0.20; TGT=0.10; CROP=640; STEP=320; BOX=3.6/TGT
FOLDS=[[1,2],[3,4],[5,6]]
ENG='/home/claude/ign/set'

def waz_tiles(k, dst, rep):
    n=f'wazemmes_{k}'
    im=Image.open(f'/home/claude/waz/label/test_wazemmes/{n}.jpg').convert('RGB')
    sc=WAZ_GSD/TGT; im=im.resize((round(im.width*sc),round(im.height*sc)),Image.BICUBIC); a=np.array(im); H,W=a.shape[:2]
    L=np.loadtxt(f'/home/claude/waz/hand/test2/{n}.txt',ndmin=2); pts=L[:,1:3]*np.array([W,H])
    for y in range(0,H-CROP+1,STEP):
        for x in range(0,W-CROP+1,STEP):
            s=pts[(pts[:,0]>=x)&(pts[:,0]<x+CROP)&(pts[:,1]>=y)&(pts[:,1]<y+CROP)]
            for r in range(rep):
                nm=f'{n}_{x}_{y}_r{r}'
                Image.fromarray(a[y:y+CROP,x:x+CROP]).save(f'{dst}/images/train/{nm}.jpg',quality=92)
                with open(f'{dst}/labels/train/{nm}.txt','w') as f:
                    for px,py in s: f.write(f'0 {(px-x)/CROP:.5f} {(py-y)/CROP:.5f} {BOX/CROP:.5f} {BOX/CROP:.5f}\n')

def build(recipe, fold):
    dst=f'/home/claude/ign/cv/{recipe}_f{fold}'
    shutil.rmtree(dst,ignore_errors=True)
    for d in ['images/train','labels/train','images/val','labels/val']: os.makedirs(f'{dst}/{d}')
    if recipe.startswith('englos'):
        for sp in ['train','val']:   # all 4 Englos squares go to training
            for f in os.listdir(f'{ENG}/images/{sp}'):
                shutil.copy(f'{ENG}/images/{sp}/{f}',f'{dst}/images/train/{f}'); shutil.copy(f'{ENG}/labels/{sp}/{f[:-4]}.txt',f'{dst}/labels/train/{f[:-4]}.txt')
    rep=3 if ('3' in recipe) else 1
    train_k=[k for k in range(1,7) if k not in FOLDS[fold]]
    for k in train_k: waz_tiles(k,dst,rep)
    # val folder only needed by Ultralytics; one training tile, validation is switched off
    f0=sorted(os.listdir(f'{dst}/images/train'))[0]
    shutil.copy(f'{dst}/images/train/{f0}',f'{dst}/images/val/{f0}'); shutil.copy(f'{dst}/labels/train/{f0[:-4]}.txt',f'{dst}/labels/val/{f0[:-4]}.txt')
    open(f'{dst}/data.yaml','w').write(f'path: {dst}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n')
    return dst, train_k

def main(recipe, epochs):
    preds={}
    for fold in range(3):
        dst,train_k=build(recipe,fold)
        m=YOLO('/home/claude/cars/w/own_cowc_yolo11n.pt')
        extra=dict(freeze=10,lr0=0.002,lrf=0.5,warmup_epochs=0,mosaic=0.5) if recipe.endswith('gentle') else dict(mosaic=1.0)
        m.train(data=f'{dst}/data.yaml',epochs=epochs,**extra,imgsz=640,batch=8,workers=2,device='cpu',
                degrees=180,flipud=0.5,fliplr=0.5,scale=0.3,hsv_v=0.3,val=False,plots=False,
                project='/home/claude/ign/cv/runs',name=f'{recipe}_f{fold}',exist_ok=True,seed=0,deterministic=True,verbose=False)
        w=f'/home/claude/ign/cv/runs/{recipe}_f{fold}/weights/last.pt'
        mm=YOLO(w)
        for k in FOLDS[fold]:
            img=np.asarray(Image.open(f'/home/claude/waz/label/test_wazemmes/wazemmes_{k}.jpg').convert('RGB'))
            preds[k]=E.detect(mm,img,0.05).tolist()
        print('FOLD DONE',recipe,fold,'trained on Wazemmes',train_k,flush=True)
    json.dump(preds,open(f'/home/claude/ign/cv/preds_{recipe}.json','w'))

if __name__=='__main__':
    main(sys.argv[1], int(sys.argv[2]))
