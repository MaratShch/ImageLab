"""Gridline-calibrated vector panel reader for Fuji AF3 datasheets."""
import pymupdf, numpy as np, re

def flat(items, n=28):
    P=[]
    for it in items:
        if it[0]=="c":
            p=[np.array([q.x,q.y]) for q in it[1:5]]
            for t in np.linspace(0,1,n):
                P.append((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t**2*p[2]+t**3*p[3])
        elif it[0]=="l":
            P.append(np.array([it[1].x,it[1].y])); P.append(np.array([it[2].x,it[2].y]))
    return np.array(P)

def rules(pg, box, minlen=20.0):
    x0,y0,x1,y1=box; V,H=[],[]
    for dr in pg.get_drawings():
        for it in dr["items"]:
            if it[0]=="l":
                p,q=it[1],it[2]
                if not (x0<=p.x<=x1 and x0<=q.x<=x1 and y0<=p.y<=y1 and y0<=q.y<=y1): continue
                if abs(p.x-q.x)<0.4 and abs(p.y-q.y)>minlen: V.append((p.x+q.x)/2)
                if abs(p.y-q.y)<0.4 and abs(p.x-q.x)>minlen: H.append((p.y+q.y)/2)
            elif it[0]=="re":
                r=it[1]
                if x0<=r.x0 and r.x1<=x1 and y0<=r.y0 and r.y1<=y1 and r.width>minlen and r.height>minlen:
                    V+= [r.x0,r.x1]; H+=[r.y0,r.y1]
    def dedupe(a, tol=1.2):
        a=sorted(a); out=[]
        for v in a:
            if not out or v-out[-1]>tol: out.append(v)
            else: out[-1]=(out[-1]+v)/2
        return out
    return dedupe(V), dedupe(H)

def labels(pg, box, axis, pat=r"[–-]?\d+(\.\d+)?"):
    x0,y0,x1,y1=box; m={}
    for w in pg.get_text("words"):
        cx,cy=(w[0]+w[2])/2,(w[1]+w[3])/2
        if x0<cx<x1 and y0<cy<y1 and re.fullmatch(pat,w[4]):
            m[float(w[4].replace("–","-"))] = cx if axis=="x" else cy
    return m

def snap(lab, rul, tol=6.0):
    """Attach each printed label to the nearest drawn rule.

    The label carries the VALUE and the rule carries the POSITION; Fuji sets
    labels typographically, so on some sheets they miss their own gridline by
    several points while the rules are exact.  Returns value -> rule position.
    """
    out={}
    for v,p in sorted(lab.items()):
        if not rul: continue
        j=int(np.argmin([abs(r-p) for r in rul]))
        if abs(rul[j]-p)<=tol: out[v]=rul[j]
    return out

def fit(m, log=False):
    k=sorted(m); a=np.array(k,float)
    if log: a=np.log10(a)
    b=np.array([m[v] for v in k])
    c=np.polyfit(a,b,1)
    return c, float(np.abs(np.polyval(c,a)-b).max()), len(k)

def inked(pg, box, inks, tol=0.06):
    R=pymupdf.Rect(*box); out={}
    for dr in pg.get_drawings():
        c=dr.get("color")
        if not c or not any(i[0]=="c" for i in dr["items"]): continue
        if not R.intersects(dr["rect"]): continue
        for k,ref in inks.items():
            if all(abs(c[i]-ref[i])<tol for i in range(3)):
                P=flat(dr["items"])
                out[k]=np.vstack([out[k],P]) if k in out else P
    return out

RGB={"R":(0.92,0.18,0.18),"G":(0.00,0.67,0.31),"B":(0.20,0.23,0.59)}
CMY={"Y":(1.00,0.95,0.00),"M":(0.93,0.00,0.55),"C":(0.00,0.68,0.94)}
GRID=np.arange(380.,681.,10.)
