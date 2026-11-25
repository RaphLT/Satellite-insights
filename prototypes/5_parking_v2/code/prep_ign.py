"""IGN fine-tuning set from the team's hand labels (Englos). Seed 0.
Split: train = englos2025_1, englos2025_2, englos2018_1; val = englos2018_2. Test = Wazemmes (never used here)."""
import os, numpy as np
from PIL import Image
SRC=0.2717; TGT=0.10; CROP=640; STEP=560; BOX=3.6/TGT
split={'englos2025_1':'train','englos2025_2':'train','englos2018_1':'train','englos2018_2':'val'}
out='set'
for stem,sp in split.items():
    os.makedirs(f'{out}/images/{sp}',exist_ok=True); os.makedirs(f'{out}/labels/{sp}',exist_ok=True)
    im=Image.open(f'/home/claude/waz/label/train_englos/{stem}.jpg').convert('RGB')
    sc=SRC/TGT; im=im.resize((round(im.width*sc),round(im.height*sc)),Image.BICUBIC); a=np.array(im); H,W=a.shape[:2]
    L=np.loadtxt(f'/home/claude/waz/hand/train/{stem}.txt',ndmin=2); pts=L[:,1:3]*np.array([W,H])
    n=0
    for y in list(range(0,H-CROP+1,STEP))+[H-CROP]:
        for x in list(range(0,W-CROP+1,STEP))+[W-CROP]:
            s=pts[(pts[:,0]>=x)&(pts[:,0]<x+CROP)&(pts[:,1]>=y)&(pts[:,1]<y+CROP)]
            nm=f'{stem}_{x}_{y}'; Image.fromarray(a[y:y+CROP,x:x+CROP]).save(f'{out}/images/{sp}/{nm}.jpg',quality=92)
            with open(f'{out}/labels/{sp}/{nm}.txt','w') as f:
                for px,py in s: f.write(f'0 {(px-x)/CROP:.5f} {(py-y)/CROP:.5f} {BOX/CROP:.5f} {BOX/CROP:.5f}\n')
            n+=1
    print(stem,sp,n,'crops',len(pts),'cars')
open(f'{out}/data.yaml','w').write(f'path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n')
