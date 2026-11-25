"""Score a model on the hand-corrected squares of other years (never used for training or for choosing anything).
Same rule as before: a detection is right if within 1.5 m of a labelled car, one-to-one."""
import sys, json, os, numpy as np
from PIL import Image
sys.path.insert(0,'/home/claude/ign'); import eval_waz as E
from ultralytics import YOLO
def score(wpath, conf, years, truth_dir='/home/claude/dates/lab_in'):
    m=YOLO(wpath); out={}
    for yr in years:
        tot={'all':np.zeros(3,int),'streets':np.zeros(3,int)}; per={}
        for k in [1,2,3]:
            nm=f'wazemmes{yr}_{k}'; tf=f'{truth_dir}/{nm}.txt'
            if not os.path.exists(tf): continue
            x0,y0=E.picks[k-1]  # mask lives in the reference photo's grid
            img=np.asarray(Image.open(f'/home/claude/dates/y{yr}/{nm}.jpg').convert('RGB'))
            t=np.loadtxt(tf,ndmin=2)[:,1:3]*640; p=E.detect(m,img,conf)[:,:2]
            for key in tot:
                pp,tt=p,t
                if key=='streets':
                    f=lambda a: E.mask[np.clip(a[:,1].astype(int)+y0,0,4999),np.clip(a[:,0].astype(int)+x0,0,4999)] if len(a) else np.zeros(0,bool)
                    pp,tt=p[f(p)],t[f(t)]
                tp=E.match(pp,tt); c=np.array([tp,len(pp)-tp,len(tt)-tp]); tot[key]+=c
                if key=='all': per[k]=dict(cars=len(t),found=tp,false=len(pp)-tp,missed=len(tt)-tp)
        res={}
        for key,(tp,fp,fn) in tot.items(): res[key]=dict(TP=int(tp),FP=int(fp),FN=int(fn),precision=round(tp/max(tp+fp,1),3),recall=round(tp/max(tp+fn,1),3))
        res['per_square']=per; out[yr]=res
    return out
if __name__=='__main__':
    r=score(sys.argv[1],float(sys.argv[2]),sys.argv[3].split(','))
    print(json.dumps(r,indent=1))
