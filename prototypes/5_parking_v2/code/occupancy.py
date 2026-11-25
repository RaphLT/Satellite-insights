"""Occupancy of public street parking and open-air public car parks, Wazemmes (Lille).
Spaces: OpenStreetMap tags. Street side marked lane/street_side/half_on_kerb/on_kerb counts;
parallel space = 5.5 m of kerb, diagonal 3.0 m, perpendicular 2.5 m, minus 5 m at each street end (crossings).
Car parks: 'capacity' tag if present, else area / 25 m2 per space (aisles included). Private car parks excluded.
Cars: detections within 7 m of a tagged street's centre line, or inside a public car-park outline.
Corrections: (1) model errors: x precision / recall measured on the hand-labelled Wazemmes test squares;
(2) hidden by trees: / (1 - share of the parking area under vegetation, from the IGN infrared photo)."""
import json, sys, numpy as np
from PIL import Image, ImageDraw
GSD=0.20; LAT0,LAT1,LON0,LON1=50.6220,50.6310,3.0424,3.0566
px=lambda lon,lat:((lon-LON0)/(LON1-LON0)*5000,(LAT1-lat)/(LAT1-LAT0)*5000)
SPACE={'parallel':5.5,'diagonal':3.0,'perpendicular':2.5}; OK={'lane','street_side','half_on_kerb','on_kerb'}
veg=np.load('/home/claude/waz/veg.npy')
def seglen(pts): p=np.array(pts); return float(np.hypot(*np.diff(p,axis=0).T).sum())*GSD
def dist_to_line(P,pts):
    p=np.array(pts); best=np.full(len(P),1e9)
    for a,b in zip(p[:-1],p[1:]):
        ab=b-a; t=np.clip(((P-a)@ab)/max(ab@ab,1e-9),0,1); q=a+t[:,None]*ab; best=np.minimum(best,np.hypot(*(P-q).T))
    return best*GSD
def area_share(draw_fn):
    M=Image.new('L',(5000,5000),0); draw_fn(ImageDraw.Draw(M)); m=np.asarray(M)>0
    return float(veg[m].mean()) if m.any() else 0.0, float(m.sum())*GSD*GSD
def run(dets, P, R):
    F=json.load(open('/home/claude/waz/osm.geojson'))['features']; units=[]
    for f in F:
        p=f['properties']; g=f['geometry']
        if 'highway' in p and g['type']=='LineString':
            pts=[px(*c) for c in g['coordinates']]; L=seglen(pts); cap=0; tagged=False
            for side in ('left','right'):
                v=p.get(f'parking:{side}',p.get('parking:both')); o=p.get(f'parking:{side}:orientation',p.get('parking:both:orientation','parallel'))
                if v is None: continue
                tagged=True
                if v in OK: cap+=max(0,int((L-10)/SPACE.get(o,5.5)))
            if not tagged: continue
            units.append(dict(kind='street',name=p.get('name','unnamed street'),pts=pts,cap=cap,L=L))
        elif p.get('amenity')=='parking' and g['type']=='Polygon' and p.get('access') in (None,'yes') and p.get('parking') in ('surface','street_side'):
            pts=[px(*c) for c in g['coordinates'][0]]
            M=Image.new('L',(5000,5000),0); ImageDraw.Draw(M).polygon(pts,fill=1); m=np.asarray(M)>0; A=float(m.sum())*GSD*GSD
            cap=int(p['capacity']) if str(p.get('capacity','')).isdigit() else int(A/25)
            units.append(dict(kind='car park',name=p.get('name','public car park'),pts=pts,cap=cap,mask=m,A=A))
    D=dets[:,:2]; used=np.zeros(len(D),bool)
    for u in units:
        if u['kind']=='car park':
            ins=u['mask'][np.clip(D[:,1].astype(int),0,4999),np.clip(D[:,0].astype(int),0,4999)]&~used
            h=float(veg[u['mask']].mean())
        else:
            ins=(dist_to_line(D,u['pts'])<=7)&~used
            M=Image.new('L',(5000,5000),0); ImageDraw.Draw(M).line(u['pts'],fill=1,width=int(14/GSD)); m=np.asarray(M)>0; h=float(veg[m].mean())
        used|=ins; n=int(ins.sum()); est=n*P/R/(1-min(h,0.9))
        u.update(visible=n,hidden_share=round(h,3),est=round(est,1))
        for k in ('mask','pts'): u.pop(k,None)
    return units
if __name__=='__main__':
    dets=np.load(sys.argv[1]); P=float(sys.argv[2]); R=float(sys.argv[3])
    U=run(dets,P,R); json.dump(U,open('occupancy.json','w'),indent=1)
    st=[u for u in U if u['kind']=='street' and u['cap']>0]; cp=[u for u in U if u['kind']=='car park']
    for lab,G in (('streets',st),('public car parks',cp)):
        cap=sum(u['cap'] for u in G); vis=sum(u['visible'] for u in G); est=sum(u['est'] for u in G)
        print(f'{lab}: units {len(G)}, spaces {cap}, cars visible {vis}, cars estimated {est:.0f}, occupancy {est/max(cap,1):.0%}, free ~{cap-est:.0f}')
