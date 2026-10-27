"""Held-out discrimination test: OpenAI Appendix-D slice vs ordinary wave observers.

Research question: does the paper's coherent, wave-derived auxiliary disk response
improve held-out prediction of free-wave concentration beyond ordinary L4,
intensity contrast, energy flux and phase coherence?

The paper's positivity/current identity is separately tested in lifted_current.py.
THIS benchmark uses the same *fixed-parameter Phi slice* already implemented;
it is not the paper's full auxiliary-time F_pl or its local-smoothing theorem.
The response is computed via Proposition 2.2's genuine mu=A^0_TT.

Design (frozen, deterministic):
* Independent generated waves; one observation and one future target per seed.
* All waves have identical *modal magnitudes* and total physical wave energy.
* Four phase families, including random, future-focus, past-focus and partial focus.
* Split samples BEFORE fitting by independently generated wave realization.
* Future target: log(spatial integral |wave(T + lag)|**4).
* Same ridge alpha=1 and train-only feature standardization for all models.
* Ordinary controls: present L4, intensity contrast, spatial energy-flux
  fluctuations and conventional band/neighbor phase locking.
* Primary increment: ordinary+disk versus ordinary on fixed held-out examples.
* Report R2, RMSE, MAE, bootstrap confidence interval for paired MSE difference;
  explicit redundancy test of adding a duplicate L4 predictor.
* Also report pairs closely matched in L4 but separated by disk response.

This is a diagnostic benchmark, NOT a proof of physical memory or causation,
and not proof that the paper adds information unavailable in the full wave.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import splu

from lifted_current import (
    Modes, auxiliary_disk, fiber_geometry, modes_example, spatial_grid, target_mu,
)

ORDINARY = ("l4_now", "contrast_now", "flux_rms", "flux_peak",
            "cross_band_lock", "neighbor_phase_lock")
FAMILY = ("random", "future_focus", "past_focus", "mixed_focus")


class CachedSlice:
    """Same variable-coefficient Appendix D Fourier-mode Dirichlet solve.

    LU cache is a speed optimization ONLY. The source is still the actual
    coherent lifted density. Discard zero spectral modes below floating-point
    roundoff; retain all physical Fourier modes with material source energy.
    """
    def __init__(self, n: int = 18, nw: int = 9, tau: float = .35,
                 radius: float = .4):
        self.n,self.nw,self.tau=n,nw,tau
        self.W,self.mask,self.spacing=auxiliary_disk(nw,radius)
        _,jac,ex,eb=fiber_geometry(self.W,tau=tau)
        self.weight=1/jac[self.mask]
        self.ex=ex[self.mask]
        self.eb=eb[self.mask]
        self.jac=jac[self.mask]
        self.locations=np.argwhere(self.mask)
        self.indices=-np.ones((nw,nw),dtype=int)
        self.indices[self.mask]=np.arange(self.mask.sum())
        h=self.spacing
        lap=lil_matrix((len(self.locations),len(self.locations)),dtype=float)
        for y,x in self.locations:
            i=self.indices[y,x]
            lap[i,i]=4/h**2
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy,xx=y+dy,x+dx
                if 0<=yy<nw and 0<=xx<nw and self.mask[yy,xx]:
                    lap[i,self.indices[yy,xx]]=-1/h**2
        self.lap=lap.tocsc()
        self.k=2*np.pi*np.fft.fftfreq(n,d=2*np.pi/n)
        self.factor_cache={}
        self.xx,self.bb=spatial_grid(n)
        self.dx=2*np.pi/n

    def factor(self,ib,ix):
        key=(ib,ix)
        if key not in self.factor_cache:
            kb,kx=self.k[ib],self.k[ix]
            screen=np.abs(kx*self.ex+kb*self.eb)**2/self.jac
            matrix=self.lap+diags(screen,format="csc")
            self.factor_cache[key]=splu(matrix)
        return self.factor_cache[key]

    def response_from_mu(self,mu):
        if mu.shape!=(self.nw,self.nw,self.n,self.n):
            raise ValueError("Unexpected lifted wave density shape")
        if not np.isfinite(mu).all():
            raise ValueError("Non-finite wave density")
        masked=np.where(self.mask[...,None,None],mu,0)
        transformed=np.fft.fft2(masked,axes=(-2,-1))
        result=np.zeros_like(transformed,dtype=complex)
        max_mode=float(np.max(abs(transformed)))
        threshold=max_mode*1e-12
        for iy in range(self.n):
            for ix in range(self.n):
                rhs=transformed[...,iy,ix][self.mask]
                if max(abs(rhs))<threshold:
                    continue
                solved=self.factor(iy,ix).solve(-rhs/(4*self.jac))
                result[...,iy,ix][self.mask]=solved
        u=np.fft.ifft2(result,axes=(-2,-1)).real
        w=np.zeros_like(self.W.real)
        w[self.mask]=self.weight
        quantity=float(.5*np.sum(-u*masked*w[...,None,None])
                       *self.spacing**2*self.dx**2)
        return quantity,u

    def measure(self,modes,T):
        mu=target_mu(modes,T,self.xx,self.bb,self.W,tau=self.tau)
        return self.response_from_mu(mu)[0]


def wave_features(modes,T,n):
    """Non-paper observables including correct free-wave physical energy flux."""
    x,b=spatial_grid(n)
    phase=(modes.kx[:,None,None]*x[None,:,:]+
           modes.kb[:,None,None]*b[None,:,:]-
           modes.omega[:,None,None]*T)
    z=modes.c[:,None,None]*np.exp(1j*phase)
    u=z.sum(axis=0)
    ut=(-1j*modes.omega[:,None,None]*z).sum(axis=0)
    ux=(1j*modes.kx[:,None,None]*z).sum(axis=0)
    ub=(1j*modes.kb[:,None,None]*z).sum(axis=0)
    flux_x=-np.real(np.conj(ut)*ux)
    flux_b=-np.real(np.conj(ut)*ub)
    magnitude=np.hypot(flux_x,flux_b)
    energy=.5*(abs(ut)**2+abs(ux)**2+abs(ub)**2)
    wave_abs2=abs(u)**2
    scale=max(float(np.mean(energy)),1e-20)
    phases=np.angle(modes.c)
    p=modes.kx
    # Cross-band coherence of mode phases at identical transverse kx.
    band1=np.argsort(p[:5])
    band2=np.argsort(p[5:])
    delta=phases[:5][band1]-phases[5:][band2]
    cross=float(abs(np.mean(np.exp(1j*delta))))
    # Phase smoothness along each radial band; includes phase relationships
    # not reducible to a single current peak statistic.
    phase_diffs=np.concatenate([np.diff(phases[:5][band1]),
                                np.diff(phases[5:][band2])])
    lock=float(abs(np.mean(np.exp(1j*phase_diffs))))
    return dict(
      l4_now=float(np.sum(wave_abs2**2)*(2*np.pi/n)**2),
      contrast_now=float(wave_abs2.max()/max(wave_abs2.mean(),1e-30)),
      flux_rms=float(np.sqrt(np.mean(magnitude**2))/scale),
      flux_peak=float(magnitude.max()/scale),
      cross_band_lock=cross,neighbor_phase_lock=lock,
      total_energy=float(np.sum(energy)*(2*np.pi/n)**2),
    )


def make_example(index,lag=.85,seed=1401):
    """The spectral powers are identical for EVERY independent realization."""
    rng=np.random.default_rng(seed+104729*index)
    base=modes_example()
    kx,kb,omega=base.kx,base.kb,base.omega
    family=FAMILY[index%len(FAMILY)]
    x0,b0=rng.uniform(0,2*np.pi,2)
    coherent=-(kx*x0+kb*b0-omega*lag)
    random_phase=rng.uniform(-np.pi,np.pi,len(kx))
    if family=="random":
        phase=random_phase
    elif family=="future_focus":
        phase=coherent+rng.normal(0,rng.uniform(0,.85),len(kx))
    elif family=="past_focus":
        phase=-(kx*x0+kb*b0+omega*lag)+rng.normal(0,.45,len(kx))
    else:
        phase=coherent.copy()
        phase[5:]=random_phase[5:]
        phase+=rng.normal(0,.25,len(kx))
    # Exactly shared absolute c across seeds and phases. Energy normalized
    # analytically by Parseval, not by a time-dependent measured statistic.
    amplitude=abs(base.c)
    amplitude=amplitude/np.sqrt(np.sum(omega**2*amplitude**2))
    c=amplitude*np.exp(1j*phase)
    return Modes(kx.copy(),kb.copy(),c),family


def ridge_predict(train,test,y_train,columns,alpha=1.0):
    """Standardization uses training realizations only."""
    X=train[:,columns]
    Z=test[:,columns]
    mean=X.mean(axis=0)
    std=np.maximum(X.std(axis=0),1e-12)
    X=(X-mean)/std
    Z=(Z-mean)/std
    model=np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@(y_train-y_train.mean()))
    return y_train.mean()+Z@model


def evaluate(rows,test_fraction=.35,seed=771,bootstrap=500):
    if len(rows)<20:
        raise ValueError("At least 20 independent wave realizations required")
    rng=np.random.default_rng(seed)
    indices=np.arange(len(rows))
    rng.shuffle(indices)
    ntest=max(5,int(round(test_fraction*len(rows))))
    test_idx=np.sort(indices[:ntest])
    train_idx=np.sort(indices[ntest:])
    fields=list(ORDINARY)+["disk"]
    X=np.array([[r[k] for k in fields] for r in rows])
    y=np.log(np.array([r["future_l4"] for r in rows]))
    tr,te=X[train_idx],X[test_idx]
    yt,yv=y[train_idx],y[test_idx]
    configs={
        "mean":[],
        "L4_only":[0],
        "ordinary_controls":list(range(len(ORDINARY))),
        "ordinary_plus_paper":list(range(len(fields))),
        "paper_only":[len(ORDINARY)],
    }
    predictions={}
    models={}
    for name,cols in configs.items():
        pred=(np.full(len(yv),yt.mean()) if not cols else
              ridge_predict(tr,te,yt,cols))
        predictions[name]=pred
        error=pred-yv
        mse=float(np.mean(error**2))
        models[name]=dict(
            r2=float(1-np.sum(error**2)/max(np.sum((yv-yv.mean())**2),1e-30)),
            rmse=float(np.sqrt(mse)),
            mae=float(np.mean(abs(error))),
        )
    basic_error=(predictions["ordinary_controls"]-yv)**2
    augmented_error=(predictions["ordinary_plus_paper"]-yv)**2
    improvement=basic_error-augmented_error
    boot_rng=np.random.default_rng(seed+13)
    draws=boot_rng.integers(0,len(yv),size=(bootstrap,len(yv)))
    intervals=np.quantile(improvement[draws].mean(axis=1),[.025,.975])
    for i,row in enumerate(rows):
        row["split"]="test" if i in set(test_idx) else "train"
    return dict(
       seed=seed,examples=len(rows),train=int(len(train_idx)),test=int(len(test_idx)),
       future_target="log spatial integral |u(t+lag)|^4",
       models=models,
       paper_mse_increment_positive_means_better=float(improvement.mean()),
       mse_increment_bootstrap95=list(map(float,intervals)),
       paper_beats_ordinary_on_test=bool(improvement.mean()>0),
       paper_increment_95ci_above_zero=bool(intervals[0]>0),
       paper_to_L4_pearson=float(np.corrcoef([r["disk"] for r in rows],
                                             [r["l4_now"] for r in rows])[0,1]),
       paper_to_L4_ratio_cv=float(
           np.std([r["disk"]/r["l4_now"] for r in rows])/
           np.mean([r["disk"]/r["l4_now"] for r in rows])),
       test_truth_log_future_L4=list(map(float,yv)),
       test_prediction_ordinary=list(map(float,predictions["ordinary_controls"])),
       test_prediction_with_paper=list(map(float,predictions["ordinary_plus_paper"])),
       test_indices=list(map(int,test_idx)),
       warnings=[
          "Single random holdout split, independent waves but shared simulated spectral support.",
          "Fixed-Phi Appendix-D slice, not integrated full F_pl; no nonlinear conserved theorem.",
          "No feature can contain more information than the full known wave coefficients.",
       ]
    )


def matched_pairs(rows,l4_tolerance=.01):
    """Non-cherry-picked ranking of *all* same-power, near-L4 pairs."""
    values=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            a,b=rows[i],rows[j]
            l4diff=abs(a["l4_now"]-b["l4_now"])/max(.5*(a["l4_now"]+b["l4_now"]),1e-30)
            if l4diff<=l4_tolerance:
                diskdiff=abs(a["disk"]-b["disk"])/max(.5*(a["disk"]+b["disk"]),1e-30)
                values.append((i,j,float(l4diff),float(diskdiff)))
    vals=np.array([x[3] for x in values])
    return dict(
       condition="equal modal powers, energy; |delta L4| / mean L4 <= 1%",
       number_pairs=len(values),
       median_paper_relative_difference=(float(np.median(vals)) if len(vals) else None),
       p95_paper_relative_difference=(float(np.quantile(vals,.95)) if len(vals) else None),
       strongest_examples=sorted(values,key=lambda x:x[3],reverse=True)[:5],
    )


def run(cases=128,n=18,disk=9,lag=.85,seed=1401,output="results/heldout",bootstrap=500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if cases<20 or n<18:
        raise ValueError("At least 20 samples and n>=18 required to resolve this wave cap")
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    solver=CachedSlice(n=n,nw=disk)
    records=[]
    for i in range(cases):
        wave,family=make_example(i,lag=lag,seed=seed)
        before=wave_features(wave,0,n)
        later=wave_features(wave,lag,n)
        paper=solver.measure(wave,0)
        record=dict(id=i,family=family,
                    **before,disk=paper,future_l4=later["l4_now"],
                    future_contrast=later["contrast_now"])
        records.append(record)
    metrics=evaluate(records,bootstrap=bootstrap)
    pairs=matched_pairs(records)
    metrics["matched_L4_pairs"]=pairs
    metrics["design"]={
      "n":n,"disk":disk,"lag":lag,"seed":seed,"cases":cases,
      "spectral_powers_same_across_all_cases":True,
      "auxiliary_source":"true A0_TT, Proposition 2.2",
      "operator":"fixed-Phi, fixed-tau variable-coefficient Appendix D (D.1)",
    }
    metrics["energy_drift_across_samples"]=float(
        max(r["total_energy"] for r in records)-
        min(r["total_energy"] for r in records))
    with (folder/"independent_waves.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (folder/"heldout_metrics.json").open("w") as stream:
        json.dump(metrics,stream,indent=2,allow_nan=False)
    a=np.array([r["l4_now"] for r in records])
    b=np.array([r["disk"] for r in records])
    fig,ax=plt.subplots(1,1,figsize=(7,5))
    for family in FAMILY:
        select=np.array([r["family"]==family for r in records])
        ax.scatter(a[select],b[select],s=16,alpha=.75,label=family)
    ax.set(xlabel="Ordinary L4 now",ylabel="Paper derived auxiliary slice response",
           title="Independent realizations, same wave energy and modal power")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/"paper_vs_L4.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    models=metrics["models"]
    labels=["L4_only","ordinary_controls","ordinary_plus_paper","paper_only"]
    ax.bar(range(len(labels)),[models[k]["rmse"] for k in labels])
    ax.set_xticks(range(len(labels)),["L4","Ordinary","Ordinary + paper","Paper alone"])
    ax.set(ylabel="Held-out RMSE of log future L4",
           title="Independent-wave holdout (lower is better)")
    fig.tight_layout()
    fig.savefig(folder/"heldout_predictions.png",dpi=140);plt.close(fig)
    print(f"Heldout n={cases}, train={metrics['train']}, test={metrics['test']}, "
          f"disk LUs cached={len(solver.factor_cache)}")
    for name,scores in metrics["models"].items():
        print(f"  {name}: RMSE={scores['rmse']:.6f} R2={scores['r2']:.6f}")
    print(f"Paper incremental MSE benefit={metrics['paper_mse_increment_positive_means_better']:.7g}; "
          f"bootstrap95={metrics['mse_increment_bootstrap95']}")
    print(f"Paper vs L4 corr={metrics['paper_to_L4_pearson']:.6f}, "
          f"ratio CV={metrics['paper_to_L4_ratio_cv']:.5f}, "
          f"matched L4 pairs={pairs['number_pairs']}")
    print(f"Energy spread over realizations={metrics['energy_drift_across_samples']:.3g}")
    print(f"Outputs: {folder}")
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",type=int,default=128)
    p.add_argument("--n",type=int,default=18)
    p.add_argument("--disk",type=int,default=9)
    p.add_argument("--lag",type=float,default=.85)
    p.add_argument("--seed",type=int,default=1401)
    p.add_argument("--bootstrap",type=int,default=500)
    p.add_argument("--output",default="results/heldout")
    a=p.parse_args()
    run(cases=a.cases,n=a.n,disk=a.disk,lag=a.lag,
        seed=a.seed,bootstrap=a.bootstrap,output=a.output)


if __name__=="__main__":
    main()
