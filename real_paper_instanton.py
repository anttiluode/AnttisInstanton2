"""Gemini-derived live Appendix-D focus view + two-center motion diagnostics.

Physical dynamics: original Antti phi^4, entirely unchanged.

Important mathematical boundaries:
* Each directional cap forms a Proposition-2.2-inspired coherent density
  from the positive-frequency FREE-WAVE projection of (phi,phi_t).
* Wave propagation is nonlinear; this is an instantaneous diagnostic,
  not the paper's conserved nonlinear current.
* Screened Appendix-D Dirichlet response is a FROZEN-COEFFICIENT
  approximation using mean J² and directional k·e, NOT the exact
  variable-coefficient fiber or the integrated paper F_pl.
* Cap intensities are ADDED, therefore cross-cap interference is absent.

The two centroids are descriptive observables, not interacting particles.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from paper_instanton import Instanton
from center_dynamics import CenterTracker, derivative, periodic_delta


class PlanarAppendixDObserver:
    """Adapted from the fixed fast real_paper_instanton.py supplied via Gemini."""
    def __init__(self,n=128,nw=7,w0=.30,tau=.35,num_caps=6,
                 k_band=(.04,.55)):
        if nw<5 or not nw%2 or num_caps<1:
            raise ValueError("Use odd disk grid >=5 and positive cap count")
        self.n,self.nw,self.num_caps=n,nw,num_caps
        k=2*np.pi*np.fft.fftfreq(n,d=1.)
        self.kx,self.ky=np.meshgrid(k,k,indexing="xy")
        self.k_mag=np.hypot(self.kx,self.ky)
        self.inv_k=np.zeros_like(self.k_mag)
        valid=self.k_mag>1e-8
        self.inv_k[valid]=1/self.k_mag[valid]
        kmin,kmax=k_band
        mid=(kmin+kmax)/2; width=.45*(kmax-kmin)
        self.annulus=np.exp(-.5*((self.k_mag-mid)/width)**2)
        self.annulus[0,0]=0
        c=np.linspace(-w0,w0,nw)
        wa,wb=np.meshgrid(c,c,indexing="xy")
        self.W=wa+1j*wb
        self.dw=float(c[1]-c[0])
        self.mask=(wa**2+wb**2)<w0**2*(1-1e-9)
        self.indices=np.flatnonzero(self.mask)
        if not len(self.indices):
            raise ValueError("No interior auxiliary-disk cells")
        Y=(np.sqrt(2)*tau)/(1+.5*self.W**2)
        self.y_abs=np.abs(Y)
        self.J2=np.abs((1+.5*self.W**2)/(np.sqrt(2)*Y))**2
        self.j2_mean=float(np.mean(self.J2[self.mask]))
        self._disk();self._caps()

    def _disk(self):
        m=self.nw;h=self.dw
        lap=np.zeros((m*m,m*m))
        for iy in range(m):
            for ix in range(m):
                if not self.mask[iy,ix]:
                    continue
                i=iy*m+ix
                lap[i,i]=4/h**2
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy,xx=iy+dy,ix+dx
                    if 0<=yy<m and 0<=xx<m and self.mask[yy,xx]:
                        lap[i,yy*m+xx]=-1/h**2
        self.evals,self.evecs=np.linalg.eigh(lap[np.ix_(self.indices,self.indices)])

    def _caps(self):
        W=self.W.ravel()[self.indices,None,None]
        y=self.y_abs.ravel()[self.indices,None,None]
        self.j2_int=self.J2.ravel()[self.indices,None,None]
        center=int(np.argmin(np.abs(W[:,0,0])))
        self.center_flat_index=center
        self.caps=[]
        for ic in range(self.num_caps):
            theta=ic*2*np.pi/self.num_caps
            kb=self.kx*np.cos(theta)+self.ky*np.sin(theta)
            kp=-self.kx*np.sin(theta)+self.ky*np.cos(theta)
            kappa=np.maximum((self.k_mag+kb)/np.sqrt(2),0)
            valid=(kb>1e-6)&(kappa>1e-6)
            v=np.zeros_like(kb)
            v[valid]=kp[valid]/kappa[valid]
            taper=np.zeros_like(kb)
            taper[valid]=np.exp(-.5*(v[valid]/.45)**4)
            kap=kappa[None,:,:]
            direction=v[None,:,:]
            gaussian=np.sqrt(kap/y)*np.exp(-.5*kap*y*
                                  np.abs(W-direction)**2)*taper[None,:,:]
            b0=(1+.5*W.real*direction)/np.sqrt(2)
            b1=(.5*W.imag*direction)/np.sqrt(2)
            m0=np.asarray(b0*gaussian,dtype=np.complex64)
            m1=np.asarray(b1*gaussian,dtype=np.complex64)
            screening=(.5*kp**2/self.j2_mean)[None,:,:]
            denom=np.asarray(1/(self.evals[:,None,None]+screening),float)
            self.caps.append((m0,m1,denom))

    def observe(self,phi,prev,dt,c_wave=1):
        velocity=(phi-prev)/dt
        phi_hat=np.fft.fft2(phi-phi.mean())
        vt_hat=np.fft.fft2(velocity)
        free_hat=np.asarray(.5*(phi_hat+1j*(self.inv_k/c_wave)*vt_hat)*
                            self.annulus,dtype=np.complex64)
        focus=np.zeros((self.n,self.n))
        at_w0=np.zeros_like(focus)
        for m0,m1,inverse in self.caps:
            s0=np.fft.ifft2(m0*free_hat[None,:,:],axes=(-2,-1))
            s1=np.fft.ifft2(m1*free_hat[None,:,:],axes=(-2,-1))
            mu=np.abs(s0)**2+np.abs(s1)**2
            # Dirichlet solve uses a frozen screening approximation:
            rhs=np.fft.fft2(mu/(4*self.j2_int),axes=(-2,-1))
            proj=np.tensordot(self.evecs.T,rhs,axes=([1],[0]))
            u_hat=np.tensordot(self.evecs,-proj*inverse,axes=([1],[0]))
            potential=np.fft.ifft2(u_hat,axes=(-2,-1)).real
            # Positive source should imply u <= 0, but the approximate
            # FFT-frozen solver can show tiny overshoots.
            potential=np.minimum(potential,0)
            focus+=.5*np.sum((-potential)*mu/self.j2_int,axis=0)*self.dw**2
            at_w0+=mu[self.center_flat_index]
        return focus,at_w0


def _safe_positions(axis,rows,name):
    return [r[f"{name}_{axis}"] for r in rows]


def run_live(*,n=128,nw=7,caps=6,steps_per_frame=10,
             frames=0,output="results/center_dynamics",max_lag=12,
             confidence=.08,show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button

    sim=Instanton(n=n)
    observer=PlanarAppendixDObserver(n=n,nw=nw,num_caps=caps)
    tracker=CenterTracker(n,confidence=confidence)
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    phi_im=ax[0,0].imshow(sim.phi,origin="lower",cmap="RdBu_r")
    mu_im=ax[0,1].imshow(np.zeros((n,n)),origin="lower",cmap="magma")
    pot_im=ax[0,2].imshow(np.zeros((n,n)),origin="lower",cmap="viridis")
    for im,axis in ((phi_im,ax[0,0]),(mu_im,ax[0,1]),(pot_im,ax[0,2])):
        fig.colorbar(im,ax=axis,fraction=.046)
    dot1,=ax[0,0].plot([],[],"go",ms=8,label="Positive-field (periodic) CoM")
    dot2,=ax[0,0].plot([],[],"m*",ms=12,label="Auxiliary focus CoM")
    ax[0,0].legend(fontsize=7)
    stopped=[False]

    def save():
        summary=tracker.write(folder,max_lag=max_lag)
        fig.savefig(folder/"center_dynamics.png",dpi=135)
        print(f"Saved {len(tracker.rows)} frames to {folder}; "
              f"valid paired centers={summary.get('paired_centers',0)}")
        return summary

    def render():
        focus,mu=observer.observe(sim.phi,sim.prev,sim.dt,sim.c)
        r=tracker.record(sim.time,sim.phi,sim.prev,sim.dt,focus)
        phi_im.set_data(sim.phi)
        phi_im.set_clim(-max(1e-3,abs(sim.phi).max()),
                        max(1e-3,abs(sim.phi).max()))
        mu_im.set_data(mu);mu_im.set_clim(0,max(float(mu.max()),1e-12))
        pot_im.set_data(focus);pot_im.set_clim(0,max(float(focus.max()),1e-16))
        if r["green_valid"]:
            dot1.set_data([r["green_x"]],[r["green_y"]])
        else:
            dot1.set_data([],[])
        if r["focus_valid"]:
            dot2.set_data([r["focus_x"]],[r["focus_y"]])
        else:
            dot2.set_data([],[])
        ax[0,0].set_title(f"Antti nonlinear φ⁴ | t={sim.time:.2f}")
        ax[0,1].set_title("Free-wave lifted density at W≈0\n(uses φ and φ̇)")
        ax[0,2].set_title("Approx. Appendix-D fixed-slice response")
        axs=ax[1]
        for a in axs:
            a.clear()
        t=np.array([r["time"] for r in tracker.rows],float)
        for name,color in (("green","green"),("focus","magenta"),
                           ("gradient","tab:orange"),("kinetic","tab:blue")):
            axs[0].plot(t,_safe_positions("x",tracker.rows,name),
                        color=color,label=name)
        axs[0].set_title("Periodic X centers (same coordinate system)")
        axs[0].set_ylabel("x cell")
        axs[0].set_xlabel("Simulation time")
        axs[0].set_ylim(0,n)
        axs[0].legend(fontsize=7)
        distance=np.array([r["offset_distance"] for r in tracker.rows])
        dx=np.array([r["offset_x"] for r in tracker.rows])
        axs[1].plot(t,dx,"m",label="signed focus minus green X")
        axs[1].plot(t,distance,"k",alpha=.6,label="periodic 2D distance")
        axs[1].axhline(0,color="gray",lw=.5)
        axs[1].set_title("Separation and centroid crossings")
        axs[1].set_xlabel("Simulation time")
        axs[1].legend(fontsize=7)
        if len(t)>=max(16,2*max_lag):
            report=tracker.analysis(max_lag=max_lag)
            for name,color in (("focus","m"),("gradient","tab:orange"),
                               ("kinetic","tab:blue")):
                curve=report["velocity_cross_correlations"][f"{name}_to_green_x"]["lag_curve"]
                axs[2].plot([z["lag"] for z in curve],
                            [np.nan if z["correlation"] is None else z["correlation"] for z in curve],
                            color=color,label=name)
        axs[2].axhline(0,color="gray",lw=.5)
        axs[2].axvline(0,color="gray",lw=.5)
        axs[2].set_ylim(-1.05,1.05)
        axs[2].set_title("Velocity lead/lag (positive = observer leads green)")
        axs[2].set_xlabel("Lag (frames)")
        axs[2].legend(fontsize=7) if axs[2].lines else None
        fig.suptitle("Two centers: shared motion, separation and possible overshoot | "
                     "exploratory, not causality")
        return r

    def frame(_):
        if not stopped[0]:
            sim.step(steps_per_frame)
            render()

    if show:
        # Leave Matplotlib's native navigation toolbar intact.
        b_pause=Button(fig.add_axes([.895,.007,.055,.028]),"Pause")
        b_save=Button(fig.add_axes([.952,.007,.045,.028]),"Save")
        def toggle(_):
            stopped[0]=not stopped[0]
            b_pause.label.set_text("Run" if stopped[0] else "Pause")
        b_pause.on_clicked(toggle)
        b_save.on_clicked(lambda _:save())
        render()
        ani=FuncAnimation(fig,frame,interval=60,blit=False,
                          cache_frame_data=False)
        plt.show()
        save()
        plt.close(fig)
        return tracker
    else:
        for i in range(frames):
            sim.step(steps_per_frame)
            render()
        save()
        plt.close(fig)
        return tracker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid",type=int,default=128)
    p.add_argument("--disk",type=int,default=7)
    p.add_argument("--caps",type=int,default=6)
    p.add_argument("--steps-per-frame",type=int,default=10)
    p.add_argument("--frames",type=int,default=0,
                   help="0 = interactive GUI; >0 = headless batch")
    p.add_argument("--confidence",type=float,default=.08)
    p.add_argument("--max-lag",type=int,default=12)
    p.add_argument("--output",default="results/center_dynamics")
    args=p.parse_args()
    run_live(n=args.grid,nw=args.disk,caps=args.caps,
             steps_per_frame=args.steps_per_frame,frames=args.frames,
             confidence=args.confidence,max_lag=args.max_lag,
             output=args.output,show=args.frames==0)


if __name__=="__main__":
    main()
