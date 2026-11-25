"""Build a correction pack for one year: mosaic, alignment to the reference photo, 2 test squares pre-marked.
Class 0 = car (model confident), class 1 = maybe (low confidence, on a street or car park) -> keep if car, delete if not."""
import sys, os, json, zipfile, numpy as np
from PIL import Image
sys.path.insert(0,'/home/claude/ign'); import eval_waz as E
from ultralytics import YOLO
from numpy.fft import fft2, ifft2
yr=sys.argv[1]; d=f'/home/claude/dates/y{yr}'
def mos(suf):
    M=Image.new('RGB',(5000,5000))
    for q,(x,y) in {'NW':(0,0),'NE':(2500,0),'SW':(0,2500),'SE':(2500,2500)}.items(): M.paste(Image.open(f'{d}/{q}{suf}.jpg').convert('RGB'),(x,y))
    return M
rgb=mos(''); rgb.save(f'{d}/rgb.jpg',quality=93); mos('_irc').save(f'{d}/irc.jpg',quality=93)
a=np.asarray(rgb.convert('L'),float); b=np.asarray(Image.open('/home/claude/waz/waz_rgb.jpg').convert('L'),float)
def shift(y,x):  # offset of this year vs reference around (y,x): content at ref(y,x) is at year(y+dy,x+dx)
    A=a[y:y+1024,x:x+1024]; B=b[y:y+1024,x:x+1024]
    g=lambda I: np.abs(np.diff(I,axis=1))[:-1]+np.abs(np.diff(I,axis=0))[:,:-1]
    A=g(A); B=g(B); F=fft2(A-A.mean())*np.conj(fft2(B-B.mean())); r=np.real(ifft2(F/(np.abs(F)+1e-9)))
    dy,dx=np.unravel_index(r.argmax(),r.shape); n=r.shape[0]
    return (dy-n if dy>n//2 else dy),(dx-n if dx>n//2 else dx)
shifts={}
for q,(y,x) in {'NW':(700,700),'NE':(700,3300),'SW':(3300,700),'SE':(3300,3300)}.items(): shifts[q]=shift(y,x)
shifts={k:[int(v[0]),int(v[1])] for k,v in shifts.items()}
json.dump(shifts,open(f'{d}/shifts.json','w')); print('shifts',shifts)
m=YOLO('/home/claude/cars/w/own_cowc_yolo11n.pt')
pack=f'/home/claude/out/label_{yr}.zip'; z=zipfile.ZipFile(pack,'w')
z.writestr('labels.txt','car\nmaybe\n')
for k in [1,2,3]:
    x0,y0=E.picks[k-1]; q=('N' if y0<2500 else 'S')+('W' if x0<2500 else 'E'); dy,dx=shifts[q]
    xs,ys=int(np.clip(x0+dx,0,4360)),int(np.clip(y0+dy,0,4360))
    crop=np.asarray(rgb)[ys:ys+640,xs:xs+640]; nm=f'wazemmes{yr}_{k}'
    Image.fromarray(crop).save(f'{d}/{nm}.jpg',quality=95); z.write(f'{d}/{nm}.jpg',f'{nm}.jpg')
    det=E.detect(m,crop,0.05); inm=E.mask[np.clip(det[:,1].astype(int)+ys,0,4999),np.clip(det[:,0].astype(int)+xs,0,4999)]
    lines=[]; nc=nm_=0
    for (cx,cy,s),im in zip(det,inm):
        if s>=0.25: c=0; nc+=1
        elif im: c=1; nm_+=1
        else: continue
        lines.append(f'{c} {cx/640:.5f} {cy/640:.5f} {18/640:.5f} {18/640:.5f}')
    open(f'{d}/{nm}.txt','w').write('\n'.join(lines)+'\n'); z.write(f'{d}/{nm}.txt',f'{nm}.txt')
    json.dump({'x':xs,'y':ys},open(f'{d}/{nm}_pos.json','w'))
    print(nm,'cars',nc,'maybes',nm_)
z.close(); print(pack)
