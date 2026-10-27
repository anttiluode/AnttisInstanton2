"""Two-center lead/lag diagnostics for Antti's nonlinear phi^4 field.

All periodic centers and displacements use a consistent torus convention.
Also record the original *Cartesian* positive-field center for comparison.
A location becomes undefined if its mass or circular concentration vanishes.

This module never evolves the physical field and has no causal/particle claims.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def periodic_delta(a, b, length):
    """Signed shortest separation a-b on a periodic axis."""
    return (np.asarray(a) - np.asarray(b) + length/2) % length - length/2


def center_info(field, *, periodic=True, min_mass=1e-6, min_resultant=0.08):
    """Return (x, y, mass, x/y resultant strength, valid), never fake a center."""
    a=np.asarray(field,float)
    if a.ndim!=2 or not np.isfinite(a).all():
        raise ValueError("Finite 2D field required")
    a=np.maximum(a,0.0)
    mass=float(a.sum())
    if mass<min_mass:
        return dict(x=float("nan"),y=float("nan"),mass=mass,
                    coherence=0.0,valid=False)
    ny,nx=a.shape
    if periodic:
        ex=np.dot(a.sum(axis=0),np.exp(2j*np.pi*np.arange(nx)/nx))/mass
        ey=np.dot(a.sum(axis=1),np.exp(2j*np.pi*np.arange(ny)/ny))/mass
        strength=min(abs(ex),abs(ey))
        valid=bool(strength>=min_resultant)
        if valid:
            x=float((np.angle(ex)%(2*np.pi))*nx/(2*np.pi))
            y=float((np.angle(ey)%(2*np.pi))*ny/(2*np.pi))
        else:
            x=y=float("nan")
    else:
        strength=float("nan")
        xx=np.arange(nx,dtype=float);yy=np.arange(ny,dtype=float)
        x=float(np.dot(a.sum(axis=0),xx)/mass)
        y=float(np.dot(a.sum(axis=1),yy)/mass)
        valid=True
    return dict(x=x,y=y,mass=mass,coherence=float(strength),valid=valid)


def lag_correlations(a,b,limit=12,min_pairs=12):
    """corr[a velocity at t, b velocity at t+lag], positive => a leads b."""
    av=np.asarray(a,float);bv=np.asarray(b,float)
    if av.shape!=bv.shape or av.ndim!=1:
        raise ValueError("Equal 1D arrays required")
    result=[]
    for lag in range(-limit,limit+1):
        if lag>0:
            aa,bb=av[:-lag],bv[lag:]
        elif lag<0:
            aa,bb=av[-lag:],bv[:lag]
        else:
            aa,bb=av,bv
        good=np.isfinite(aa)&np.isfinite(bb)
        n=int(good.sum())
        r=float("nan")
        if n>=min_pairs and np.std(aa[good])>1e-12 and np.std(bb[good])>1e-12:
            r=float(np.corrcoef(aa[good],bb[good])[0,1])
        result.append(dict(lag=lag,correlation=r,pairs=n))
    return result


def best_lag(rows):
    good=[r for r in rows if np.isfinite(r["correlation"])]
    return max(good,key=lambda r:abs(r["correlation"])) if good else None


def derivative(values,times,n):
    v=np.asarray(values,float);t=np.asarray(times,float)
    delta=periodic_delta(v[1:],v[:-1],n)
    dt=np.diff(t)
    out=np.full(len(v),np.nan)
    valid=np.isfinite(v[1:])&np.isfinite(v[:-1])&(dt>0)
    out[1:][valid]=delta[valid]/dt[valid]
    return out


class CenterTracker:
    """Measure original and periodic CoMs plus controls from same snapshots."""
    names=("green","focus","gradient","kinetic")
    def __init__(self,n,confidence=.08,min_green_mass=1e-3):
        self.n=n
        self.confidence=confidence
        self.min_green_mass=min_green_mass
        self.rows=[]

    def record(self,time,phi,phi_prev,dt,focus_map):
        if phi.shape!=(self.n,self.n) or focus_map.shape!=phi.shape:
            raise ValueError("Unexpected map size")
        plus=np.maximum(phi,0)
        # Use central spatial differences, with the original periodic boundary.
        gx=(np.roll(phi,-1,axis=1)-np.roll(phi,1,axis=1))/2
        gy=(np.roll(phi,-1,axis=0)-np.roll(phi,1,axis=0))/2
        grad=gx*gx+gy*gy
        kinetic=((phi-phi_prev)/dt)**2
        parts={
            "green":plus,
            "focus":np.maximum(focus_map,0),
            "gradient":grad,
            "kinetic":kinetic,
        }
        rec=dict(time=float(time))
        for name,weight in parts.items():
            info=center_info(weight,periodic=True,
                             min_mass=self.min_green_mass if name=="green" else 1e-14,
                             min_resultant=self.confidence)
            for key in ("x","y","mass","coherence","valid"):
                rec[f"{name}_{key}"]=info[key]
        original=center_info(plus,periodic=False,min_mass=self.min_green_mass)
        rec["green_cartesian_x"]=original["x"]
        rec["green_cartesian_y"]=original["y"]
        if rec["green_valid"] and rec["focus_valid"]:
            dx=float(periodic_delta(rec["focus_x"],rec["green_x"],self.n))
            dy=float(periodic_delta(rec["focus_y"],rec["green_y"],self.n))
            rec["offset_x"]=dx;rec["offset_y"]=dy
            rec["offset_distance"]=float(np.hypot(dx,dy))
        else:
            rec["offset_x"]=rec["offset_y"]=rec["offset_distance"]=float("nan")
        self.rows.append(rec)
        return rec

    def analysis(self,max_lag=12,min_pairs=12):
        if not self.rows:
            return dict(status="empty")
        t=np.array([r["time"] for r in self.rows])
        results={}
        for axis in ("x","y"):
            green=derivative([r[f"green_{axis}"] for r in self.rows],t,self.n)
            for source in ("focus","gradient","kinetic"):
                observed=derivative([r[f"{source}_{axis}"] for r in self.rows],t,self.n)
                curve=lag_correlations(observed,green,limit=max_lag,
                                       min_pairs=min_pairs)
                results[f"{source}_to_green_{axis}"]={
                    "lag_curve":curve,"best":best_lag(curve)}
        offsets=np.array([r["offset_x"] for r in self.rows],float)
        # A sign change is a *centroid crossing*, not necessarily an overshoot.
        valid=np.isfinite(offsets[1:])&np.isfinite(offsets[:-1])
        crossing=int(np.sum(valid&(offsets[1:]*offsets[:-1]<0)&
                            (abs(offsets[1:]-offsets[:-1])<self.n/2)))
        valid_count=int(sum(r["green_valid"] and r["focus_valid"] for r in self.rows))
        distances=[r["offset_distance"] for r in self.rows
                   if np.isfinite(r["offset_distance"])]
        return dict(
            status="exploratory_single_trajectory",
            samples=len(self.rows),paired_centers=valid_count,
            x_centroid_crossings=crossing,
            median_distance=float(np.median(distances)) if distances else None,
            max_lag_frames=max_lag,
            estimated_frame_dt=float(np.median(np.diff(t))) if len(t)>1 else None,
            velocity_cross_correlations=results,
            caveats=[
                "Positive lag means the named observer's velocity comes before the green centroid velocity.",
                "All velocities use periodic shortest-step differences; unreliable diffuse centroids are NaN.",
                "The original green Cartesian center is recorded separately for historical comparison.",
                "Matching centroids and sign changes do not establish physical transport, anticipation or causation.",
                "A best lag found on one correlated trajectory is exploratory, not a held-out predictive result.",
            ],
        )

    def write(self,folder,max_lag=12):
        path=Path(folder);path.mkdir(parents=True,exist_ok=True)
        if self.rows:
            with (path/"centers.csv").open("w",newline="") as stream:
                fields=list(self.rows[0])
                writer=csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader();writer.writerows(self.rows)
        summary=self.analysis(max_lag=max_lag)
        with (path/"lead_lag.json").open("w") as stream:
            json.dump(summary,stream,indent=2,allow_nan=False)
        return summary
