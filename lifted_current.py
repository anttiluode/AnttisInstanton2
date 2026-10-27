"""Paper-derived lifted current and genuine-source planar fiber (n=2).

Reference: OpenAI, Local smoothing for the Euclidean wave equation (2026).
Sections 2.1-2.4, Proposition 2.2 eqs (2.14)-(2.15); Appendix D (D.1).

Two distinct numerical objects are deliberately exposed:
  (1) B = Re sum_l Z_l tensor conjugate(Z_l), Lemma 2.1, in source
      (t,x,s0,w10,w11,w20,w21) coordinates. B is PSD and divergence-free.
  (2) mu = A^0_TT from Proposition 2.2, computed on the target
      (T,X,Y,W,Phi) fiber, and used as the *actual wave-derived source*
      in the variable-coefficient disk equation (D.1).

The disk solve is for one fixed Phi slice, without the X-mollifier or
Phi-cutoff of Section 4.2 and without the full parameter-time energy
inequality. Its quadratic energy is a slice diagnostic, NOT the
paper's integrated F_pl, and not a theorem for nonlinear phi^4.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Modes:
    """Finite Fourier sum of positive-frequency waves in a narrow +b cap.

    Real physical coordinates (T,x,b), with periodic x,b of length 2*pi.
    U(T,x,b) = sum c_j exp(i(kx*x + kb*b - |k|*T)).
    """
    kx: np.ndarray
    kb: np.ndarray
    c: np.ndarray

    def __post_init__(self):
        self.kx = np.asarray(self.kx,dtype=float)
        self.kb = np.asarray(self.kb,dtype=float)
        self.c = np.asarray(self.c,dtype=complex)
        if self.kx.shape != self.kb.shape or self.c.shape != self.kx.shape:
            raise ValueError("Wave arrays must be the same size")
        self.omega = np.hypot(self.kx,self.kb)
        self.kappa = (self.omega + self.kb)/np.sqrt(2)
        self.v = self.kx/self.kappa
        if np.any(self.kappa<=0) or np.any(np.abs(self.v)>=0.5):
            raise ValueError("Choose positive-frequency modes in a small +b cap")

    def evaluate(self,T,x,b):
        phase = (self.kx[:,None,None]*x[None,:,:]+
                 self.kb[:,None,None]*b[None,:,:]-
                 self.omega[:,None,None]*T)
        return np.sum(self.c[:,None,None]*np.exp(1j*phase),axis=0)

    def physical_energy(self,T,x,b):
        phase = (self.kx[:,None,None]*x[None,:,:]+
                 self.kb[:,None,None]*b[None,:,:]-
                 self.omega[:,None,None]*T)
        pieces=self.c[:,None,None]*np.exp(1j*phase)
        dT=np.sum(-1j*self.omega[:,None,None]*pieces,axis=0)
        dx=np.sum(1j*self.kx[:,None,None]*pieces,axis=0)
        db=np.sum(1j*self.kb[:,None,None]*pieces,axis=0)
        return 0.5*(np.abs(dT)**2+np.abs(dx)**2+np.abs(db)**2)


def modes_example(*, seed=0, scrambled=False):
    """Same spectral energy for coherent focal and phase-scrambled control."""
    kx=np.array([-2,-1,0,1,2,-2,-1,0,1,2],dtype=float)
    kb=np.array([7]*5+[8]*5,dtype=float)
    omega=np.hypot(kx,kb)
    envelope=np.exp(-0.5*(kx/1.7)**2)
    t0,x0,b0=0.8,1.3,3.1
    phase=-(kx*x0+kb*b0-omega*t0)
    if scrambled:
        phase=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(kx))
    c=envelope*np.exp(1j*phase)/envelope.sum()
    return Modes(kx,kb,c)


def spatial_grid(n):
    q=2*np.pi*np.arange(n)/n
    return np.meshgrid(q,q,indexing="xy")


def source_current(modes, variables):
    """Lemma 2.1: exact PSD source current B at one 7D lifted point.

    Ordering: [t,x,s0,w10,w11,w20,w21]. Exponent uses
    exp[-kappa/2 * sum_l (w_l1-v*w_l0)^2].
    Quaternion machinery is not needed for B in the original coordinates.
    """
    t,x,s0,w10,w11,w20,w21=np.asarray(variables,dtype=float)
    C=np.array([
        [[2,0],[0,0]],
        [[0,1],[1,0]],
        [[0,0],[0,-1]]
    ],dtype=float)
    w=[np.array([w10,w11]),np.array([w20,w21])]
    v=modes.v
    damp=np.exp(-0.5*modes.kappa*((w11-v*w10)**2+(w21-v*w20)**2))
    u=modes.c*np.exp(1j*modes.kappa*(s0+v*x-0.5*v*v*t))
    amp=np.sqrt(modes.kappa)*damp*u
    G=np.conj(np.array([amp.sum(),(amp*v).sum()]))  # Z_l uses conjugate G (Lemma 2.1)
    B=np.zeros((7,7),dtype=float)
    for l in (0,1):
        z=np.zeros(7,dtype=complex)
        z[:3]=np.array([1j*np.dot(G,Cm@w[l]) for Cm in C])
        z[3+2*l:5+2*l]=G
        B+=np.real(np.outer(z,z.conj()))
    return B


def source_divergence(modes, point, eps=2e-5):
    """Centered finite-difference divergence of each COLUMN of source B."""
    p=np.asarray(point,dtype=float)
    div=np.zeros(7)
    for i in range(7):
        shift=np.zeros(7);shift[i]=eps
        div+=(source_current(modes,p+shift)[i,:]-
              source_current(modes,p-shift)[i,:])/(2*eps)
    return div


def auxiliary_disk(n=13,radius=.4):
    if n < 7 or radius<=0:
        raise ValueError("n>=7 and positive radius required")
    coordinate=np.linspace(-radius,radius,n)
    wa,wb=np.meshgrid(coordinate,coordinate,indexing="xy")
    W=wa+1j*wb  # ordinary Python i represents paper's auxiliary k
    mask=(wa*wa+wb*wb)<(radius**2*(1-1e-10))
    return W,mask,float(coordinate[1]-coordinate[0])


def fiber_geometry(W,tau=.35,Phi=0.0):
    """Exact planar wave chart, (2.9)-(2.10), (4.2), Appendix D.

    W=P/Y, Phi=Y*(1+W^2/2)-sqrt(2)*tau;
    Y=(Phi+sqrt(2)*tau)/(1+W^2/2),  J=(1+W^2/2)/(sqrt(2)*Y).
    The spatial covector e=(dPhi/dZ, -dPhi/dP).
    """
    W=np.asarray(W,dtype=complex)
    Y=(Phi+np.sqrt(2)*tau)/(1+W*W/2)
    if np.any(np.abs(Y)<=1e-12):
        raise ValueError("Fiber chart is singular at Y=0")
    J=(1+W*W/2)/(np.sqrt(2)*Y)
    ex=(1-W*W/2)/np.sqrt(2)
    eb=-W
    return np.abs(Y),np.abs(J)**2,ex,eb


def target_mu(modes,T,x,b,W,tau=.35,Phi=0.0):
    """True A^0_TT from Proposition 2.2, Equation (2.15).

    Chi_T has two orthogonal quaternionic components: S0 and S1.
    Each S is summed coherently over modes BEFORE taking modulus squared.
    Shape: W.shape + x.shape.
    """
    y,_,_,_=fiber_geometry(W,tau=tau,Phi=Phi)
    W=np.asarray(W)
    xx=np.asarray(x)
    bb=np.asarray(b)
    phase=(modes.kx[:,None,None]*xx[None,:,:]+
           modes.kb[:,None,None]*bb[None,:,:]-
           modes.omega[:,None,None]*T)
    packet=modes.c[:,None,None]*np.exp(1j*phase)
    # loop over modes to avoid allocating a large (modes,W,x,b) tensor
    s0=np.zeros(W.shape+xx.shape,dtype=complex)
    s1=np.zeros_like(s0)
    for j in range(len(modes.c)):
        kappa=modes.kappa[j]; v=modes.v[j]
        gauss=np.sqrt(kappa/y)*np.exp(-0.5*kappa*y*np.abs(W-v)**2)
        b0=(1+W.real*v/2)/np.sqrt(2)
        b1=(W.imag*v/2)/np.sqrt(2)
        s0+=(gauss*b0)[...,None,None]*packet[j]
        s1+=(gauss*b1)[...,None,None]*packet[j]
    return np.abs(s0)**2 + np.abs(s1)**2


def source_to_target_mu_identity(modes,T,x,b,Y,W):
    """Validate A^0_TT by explicit source-current pushforward at REAL Y,W.

    Follows (2.8)-(2.11), not a fit or an independent filter.
    J=4|R|^4. At Y>0 and W real, choose R=sqrt(Y), zeta=RW.
    """
    if Y<=0 or not np.isreal(W):
        raise ValueError("This source chart identity is for Y>0, W real")
    R=np.sqrt(Y); w=float(np.real(W))
    t=(T+b)/np.sqrt(2); s0=(b-T)/np.sqrt(2)
    B=source_current(modes,[t,x,s0,R,R*w,0,0])
    jac=4*R**4
    ct=np.array([1/np.sqrt(2),0,-1/np.sqrt(2),0,0,0,0])
    return float(ct@B@ct/jac)


def disk_poisson_source(mu,W,mask,spacing,tau=.35,Phi=0.0,
                        with_potential=True):
    """Solve the ACTUAL variable-coefficient Appendix D disk equation.

    mu(W,b,x) is true A^0_TT, not an invented Gaussian. We Fourier
    transform in both physical spatial coordinates (b,x). At each k:
      [Delta_W - |k.e(W)|^2/|J(W)|^2] u_hat
         = mu_hat/(4|J(W)|^2),  u_hat|disk boundary=0.

    The discrete Laplacian is five-point with zero Dirichlet values.
    FFT in real (b,x) corresponds to the paper's physical X torus.

    Returns u(W,b,x) and a slice integral:
      1/2 integral (-u)mu |J|^-2 dW dX
    This is a fixed-Phi slice, NOT the paper's F_pl (which also
    integrates Phi/Q and includes cutoff and smoothing).
    """
    n=mu.shape[-1]
    nw=W.shape[0]
    if mu.shape != (nw,nw,n,n) or n<3:
        raise ValueError("mu dimensions must be (nw,nw,n,n)")
    y,jac2,ex,eb=fiber_geometry(W,tau=tau,Phi=Phi)
    ids=-np.ones((nw,nw),dtype=int)
    ids[mask]=np.arange(int(mask.sum()))
    cells=np.argwhere(mask)
    nc=len(cells)
    lap=lil_matrix((nc,nc),dtype=float)
    for yidx,xidx in cells:
        i=ids[yidx,xidx]
        lap[i,i]=4/spacing**2
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            yy,xx=yidx+dy,xidx+dx
            if 0<=yy<nw and 0<=xx<nw and mask[yy,xx]:
                lap[i,ids[yy,xx]]=-1/spacing**2
    base=lap.tocsr()
    mu_hat=np.fft.fft2(mu,axes=(-2,-1))
    u_hat=np.zeros_like(mu_hat,dtype=complex)
    coords=2*np.pi*np.fft.fftfreq(n,d=2*np.pi/n)
    # all spatial modes, even though the wave's *source* uses one cap,
    # because intensity cross-terms generate difference frequencies.
    ex_inside=ex[mask]
    eb_inside=eb[mask]
    jac_inside=jac2[mask]
    for ib,kb in enumerate(coords):
        for ix,kx in enumerate(coords):
            screening=(np.abs(kx*ex_inside+kb*eb_inside)**2/jac_inside)
            mat=base+diags(screening,0,format="csr")
            rhs=-mu_hat[...,ib,ix][mask]/(4*jac_inside)
            solved=spsolve(mat,rhs)
            u_hat[...,ib,ix][mask]=solved
    u=np.fft.ifft2(u_hat,axes=(-2,-1)).real
    weights=np.zeros_like(W.real)
    weights[mask]=1/jac2[mask]
    dx=2*np.pi/n
    response=float(.5*np.sum(-u*mu*weights[...,None,None])
                   *spacing**2*dx**2)
    return u,response


def observables(modes,T,n=18,nw=13,tau=.35,w_radius=.4):
    x,b=spatial_grid(n)
    W,mask,step=auxiliary_disk(nw,w_radius)
    # only the interior wave density is used in the PDE; outside is zero
    mu=target_mu(modes,T,x,b,W,tau=tau)
    mu*=mask[...,None,None]
    u,response=disk_poisson_source(mu,W,mask,step,tau=tau)
    wave=modes.evaluate(T,x,b)
    energy=modes.physical_energy(T,x,b)
    dx=2*np.pi/n
    l4=float(np.sum(np.abs(wave)**4)*dx**2)
    energy_integral=float(np.sum(energy)*dx**2)
    intensity_contrast=float(np.max(np.abs(wave)**2)
                              / max(float(np.mean(np.abs(wave)**2)),1e-30))
    weighted_mu=float(np.sum(mu)*step**2*dx**2)
    # squared modulus of the W=0 time component to inspect coherent focusing
    index=nw//2
    local=mu[index,index]
    lifted_peak=float(local.max()/max(float(local.mean()),1e-30))
    return {
      "time":float(T),
      "disk_response_slice":response,
      "wave_L4":l4,
      "physical_energy":energy_integral,
      "physical_intensity_contrast":intensity_contrast,
      "lifted_time_density_peak_to_mean":lifted_peak,
      "raw_lifted_mass":weighted_mu,
    },dict(wave=wave,mu=mu,u=u,W=W,mask=mask)


def experiment(*,n=18,nw=13,samples=9,out="results/paper_lifted",
               tstart=0.0,tend=1.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    records=[]
    images={}
    times=np.linspace(tstart,tend,samples)
    for name,scramble in (("coherent",False),("scrambled",True)):
        modes=modes_example(scrambled=scramble,seed=2026)
        for i,T in enumerate(times):
            row,data=observables(modes,float(T),n=n,nw=nw)
            records.append(dict(case=name,**row))
            if i==len(times)//2:
                images[name]=data

    with (folder/"time_series.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)

    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for row,case in enumerate(("coherent","scrambled")):
        info=images[case]
        mid=nw//2
        axes[row,0].imshow(abs(info["wave"])**2,origin="lower",cmap="magma")
        axes[row,0].set_title(f"{case}: |free wave|²")
        axes[row,1].imshow(info["mu"][mid,mid],origin="lower",cmap="magma")
        axes[row,1].set_title("True lifted density A⁰_TT at W=0")
        axes[row,2].imshow(-info["u"][mid,mid],origin="lower",cmap="viridis")
        axes[row,2].set_title("Response -u(X,W=0), Appendix D")
    fig.suptitle("OpenAI wave paper: exact A⁰_TT feeds Appendix D (fixed-Phi slice)")
    fig.savefig(folder/"wave_and_disk.png",dpi=140)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for case in ("coherent","scrambled"):
        sub=[row for row in records if row["case"]==case]
        for ax,key,title in zip(
            axes,("disk_response_slice","wave_L4","physical_energy"),
            ("Paper slice Dirichlet response","Ordinary free-wave L4",
             "Conserved physical wave energy")):
            ax.plot([r["time"] for r in sub],[r[key] for r in sub],
                    marker="o",label=case)
            ax.set(title=title,xlabel="Physical time T")
            ax.legend()
    fig.savefig(folder/"comparison_curves.png",dpi=140)
    plt.close(fig)
    max_diff=max(abs(r["physical_energy"]-records[0]["physical_energy"])
                 for r in records if r["case"]=="coherent")
    print(f"Lifted-current experiment: {len(records)} observations, "
          f"max physical-energy change {max_diff:.4g}")
    for r in records:
        if abs(r["time"]-times[len(times)//2])<1e-10:
            print(f" {r['case']}: time={r['time']:.3f}, "
                  f"auxiliary response={r['disk_response_slice']:.6g}, "
                  f"wave L4={r['wave_L4']:.6g}, "
                  f"energy={r['physical_energy']:.6g}")
    print(f"Saved outputs to {folder}")
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n",type=int,default=18,help="free-wave torus grid")
    ap.add_argument("--disk",type=int,default=13,help="auxiliary W disk grid")
    ap.add_argument("--samples",type=int,default=9)
    ap.add_argument("--output",default="results/paper_lifted")
    args=ap.parse_args()
    experiment(n=args.n,nw=args.disk,samples=args.samples,out=args.output)


if __name__=="__main__":
    main()
