"""Live, interactive Antti phi^4 + OpenAI imaginary light-ray observer.

The nonlinear field does NOT satisfy the free half-wave theorem.
The right view is the exact paper multiplier applied as a frozen spectral probe
of the current phi snapshot, not the theorem's conserved density.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from paper_instanton import (
    Instanton, circular_center, imaginary_ray, positive_center
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid",type=int,default=128)
    p.add_argument("--steps-per-frame",type=int,default=10)
    p.add_argument("--output",default="results/live_observations.csv")
    args=p.parse_args()

    sim=Instanton(n=args.grid)
    n=args.grid
    out=Path(args.output)
    history=[]
    paused=False

    fig,axes=plt.subplots(1,3,figsize=(16,6))
    plt.subplots_adjust(bottom=.29,wspace=.30)
    original=axes[0].imshow(sim.phi,origin="lower",cmap="RdBu_r")
    axes[0].set_title("Nonlinear phi⁴ substrate")
    com_dot,=axes[0].plot([],[],"go",label="Antti positive CoM")
    active_dot,=axes[0].plot([],[],"co",fillstyle="none",label="activity center")
    axes[0].legend(loc="upper left",fontsize=8)

    focus=axes[1].imshow(np.zeros((n,n)),origin="lower",cmap="magma")
    focus_dot,=axes[1].plot([],[],"c+",markersize=13)
    axes[1].set_title("Paper's complex-ray multiplier\n"
                      "(frozen snapshot; NOT conserved)")
    fig.colorbar(focus,ax=axes[1],fraction=.046)

    ax=axes[2]
    line_com,=ax.plot([],[],color="forestgreen",label="positive CoM x")
    line_focus,=ax.plot([],[],color="darkorange",label="ray peak x")
    line_act,=ax.plot([],[],color="tab:blue",label="activity center x")
    ax.set(xlim=(0,20),ylim=(0,n),xlabel="simulation time",ylabel="X")
    ax.set_title("Three different notions of position")
    ax.legend(fontsize=8,loc="upper right")

    slider_specs=[
        ("rho","ρ: imaginary displacement",0.0,15.0,4.0,.18),
        ("angle","ω: ray angle", -np.pi,np.pi,0.45,.13),
        ("center","annulus |k|",0.25,2.5,1.4,.08),
        ("width","annulus half-width",0.1,1.3,0.6,.03),
    ]
    controls={}
    for key,title,lo,hi,start,y in slider_specs:
        controls[key]=Slider(fig.add_axes([.2,y,.56,.035]),
                             title,lo,hi,valinit=start)
    pause_btn=Button(fig.add_axes([.80,.115,.085,.06]),"Pause / run")
    save_btn=Button(fig.add_axes([.80,.035,.085,.06]),"Save CSV")

    def toggle(_):
        nonlocal paused
        paused=not paused

    def save(_):
        out.parent.mkdir(parents=True,exist_ok=True)
        if not history:
            return
        with out.open("w",newline="") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved {len(history)} samples to {out}")

    pause_btn.on_clicked(toggle)
    save_btn.on_clicked(save)

    def frame(_):
        if not paused:
            sim.step(args.steps_per_frame)
        snap=sim.phi
        spectrum=np.fft.fft2(snap-snap.mean())
        wave,baseline,_=imaginary_ray(
            spectrum, rho=controls["rho"].val,
            angle=controls["angle"].val,time=0,
            center=controls["center"].val,
            half_width=controls["width"].val)
        intensity=np.abs(wave)**2
        y,x=np.unravel_index(intensity.argmax(),intensity.shape)
        com=positive_center(snap)
        activity=circular_center(sim.dynamic_activity())
        original.set_data(snap)
        vmax=max(0.1,float(np.max(abs(snap))))
        original.set_clim(-vmax,vmax)
        focus.set_data(intensity)
        focus.set_clim(0,max(float(intensity.max()),1e-15))
        com_dot.set_data([com[0]],[com[1]])
        active_dot.set_data([activity[0]],[activity[1]])
        focus_dot.set_data([x],[y])
        contrast=float(intensity.max()) / max(baseline,1e-30)
        axes[0].set_title(f"Nonlinear phi⁴ | t={sim.time:.2f}")
        axes[1].set_title(f"Paper free-wave ray on frozen φ\n"
                          f"contrast={contrast:.2f}; peak=({x},{y})")
        if not paused:
            history.append(dict(time=sim.time,com_x=com[0],com_y=com[1],
                    focus_x=x,focus_y=y,activity_x=activity[0],
                    activity_y=activity[1],contrast=contrast,
                    total_energy=float(sim.energy_density().sum())))
            if len(history)>2000:
                history.pop(0)
        ts=[s["time"] for s in history]
        for line,key in ((line_com,"com_x"),(line_focus,"focus_x"),
                         (line_act,"activity_x")):
            line.set_data(ts,[s[key] for s in history])
        if ts:
            ax.set_xlim(max(0,ts[-1]-20),max(20,ts[-1]+.2))
        return original,focus,com_dot,active_dot,focus_dot,line_com,line_focus,line_act

    ani=FuncAnimation(fig,frame,interval=40,blit=False,
                      cache_frame_data=False)
    plt.show()
    # Hold a reference so the animation is not garbage-collected prematurely.
    return ani


if __name__=="__main__":
    main()
