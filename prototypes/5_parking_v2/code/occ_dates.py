"""Occupancy of Wazemmes public parking for each photo date, with the same model, threshold and corrections.
Model: COWC-trained YOLO11n, threshold 0.10 (chosen by cross-validation on the hand-labelled squares).
Correction: x precision / recall measured on streets by cross-validation (0.895 / 0.819), / (1 - tree share that date)."""
import sys, json, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'/home/claude/ign'); import eval_waz as E, occupancy as O
from ultralytics import YOLO
P,R,CONF=0.895,0.819,0.10
src={'current':('/home/claude/waz/waz_rgb.jpg','/home/claude/waz/waz_irc.jpg',None),
     '2021':('/home/claude/dates/y2021/rgb.jpg','/home/claude/dates/y2021/irc.jpg','/home/claude/dates/y2021/shifts.json'),
     '2018':('/home/claude/dates/y2018/rgb.jpg','/home/claude/dates/y2018/irc.jpg','/home/claude/dates/y2018/shifts.json')}
m=YOLO('/home/claude/cars/w/own_cowc_yolo11n.pt'); out={}
for yr in sys.argv[1].split(','):
    rgb,irc,sh=src[yr]
    a=np.asarray(Image.open(rgb).convert('RGB')); dets=E.detect(m,a,CONF)
    if sh:  # move detections into the reference photo's pixel grid (OSM outlines are drawn there)
        S=json.load(open(sh))
        for i,(x,y,_) in enumerate(dets):
            q=('N' if y<2500 else 'S')+('W' if x<2500 else 'E'); dets[i,0]-=S[q][1]; dets[i,1]-=S[q][0]
    np.save(f'/home/claude/dates/dets_{yr}.npy',dets)
    I=np.asarray(Image.open(irc).convert('RGB'),float); nir,red=I[...,0],I[...,1]
    veg=(nir-red)/(nir+red+1e-6)>0.25
    if sh:  # shift the tree mask the same way (whole-picture average shift is enough at 10 m street buffers)
        dy=int(round(np.mean([v[0] for v in S.values()]))); dx=int(round(np.mean([v[1] for v in S.values()])))
        veg=np.roll(veg,(-dy,-dx),axis=(0,1))
    O.veg=veg
    U=O.run(dets,P,R)
    st=[u for u in U if u['kind']=='street' and u['cap']>0]; cp=[u for u in U if u['kind']=='car park']
    res={'detections':int(len(dets)),'tree_share':round(float(veg.mean()),3)}
    for lab,G in (('streets',st),('car_parks',cp)):
        cap=sum(u['cap'] for u in G); vis=sum(u['visible'] for u in G); est=sum(u['est'] for u in G)
        res[lab]=dict(units=len(G),spaces=cap,visible=vis,estimated=round(est),occupancy=round(est/max(cap,1),3))
    res['nouvelle_aventure']=[dict(cap=u['cap'],visible=u['visible'],est=u['est'],hidden=u['hidden_share']) for u in cp if u['cap']>400]
    res['units']=U; out[yr]=res
    print(yr,json.dumps({k:v for k,v in res.items() if k!='units'}),flush=True)
json.dump(out,open(f'/home/claude/dates/occ_{sys.argv[1].replace(",","_")}.json','w'),indent=1)
