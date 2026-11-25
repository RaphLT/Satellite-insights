"""Score every recipe on the 6 Wazemmes squares, each square predicted by a model that never saw it.
Confidence threshold for a fold is chosen on the OTHER folds' squares (best F1), so the test square never picks its own threshold."""
import json, sys, os, numpy as np
sys.path.insert(0,'/home/claude/ign'); import eval_waz as E
FOLDS=[[1,2],[3,4],[5,6]]; S=640
mask=E.mask
truth={k:np.loadtxt(f'/home/claude/waz/hand/test2/wazemmes_{k}.txt',ndmin=2)[:,1:3]*S for k in range(1,7)}
def counts(p,t,x0,y0,streets):
    if streets:
        inm=lambda a: mask[np.clip(a[:,1].astype(int)+y0,0,4999),np.clip(a[:,0].astype(int)+x0,0,4999)] if len(a) else np.zeros(0,bool)
        p=p[inm(p)]; t=t[inm(t)]
    tp=E.match(p,t); return np.array([tp,len(p)-tp,len(t)-tp])
def f1(c): tp,fp,fn=c; return 2*tp/max(2*tp+fp+fn,1)
def score(preds):
    tot={'all':np.zeros(3,int),'streets':np.zeros(3,int)}; chosen=[]
    for fold in FOLDS:
        others=[k for k in range(1,7) if k not in fold]
        best=max(np.arange(0.05,0.9,0.05),key=lambda c: f1(sum(counts(preds[k][preds[k][:,2]>=c,:2],truth[k],*E.picks[k-1],True) for k in others)))
        chosen.append(round(float(best),2))
        for k in fold:
            p=preds[k][preds[k][:,2]>=best,:2]
            for key in tot: tot[key]+=counts(p,truth[k],*E.picks[k-1],key=='streets')
    out={'thresholds':chosen}
    for key,(tp,fp,fn) in tot.items(): out[key]=dict(TP=int(tp),FP=int(fp),FN=int(fn),precision=round(tp/max(tp+fp,1),3),recall=round(tp/max(tp+fn,1),3),F1=round(f1((tp,fp,fn)),3))
    return out
if __name__=='__main__':
    res={}
    for f in sorted(os.listdir('/home/claude/ign/cv')):
        if f.startswith('preds_'):
            P=json.load(open(f'/home/claude/ign/cv/{f}')); P={int(k):np.array(v).reshape(-1,3) for k,v in P.items()}
            res[f[6:-5]]=score(P); print(f[6:-5],json.dumps(res[f[6:-5]]))
    json.dump(res,open('/home/claude/ign/cv/scores.json','w'),indent=1)
