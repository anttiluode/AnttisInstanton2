"""Comparison bridge: original nonlinear phi^4 -> local FREE-wave reference.

IMPORTANT: The paper's theorem does NOT apply to phi^4.
At each nonlinear snapshot we obtain *both* phi and phi_t, extract a
positive-frequency free-wave subspace, and rebuild Proposition 2.2's
current on that independent FREE reference wave. This is a diagnostic
with explicit approximation error and no claimed conservation across
different nonlinear snapshots.

Coordinates: simulator's square has N grid cells per side and dx=1.
Map its coordinates to a 2*pi torus alpha=2*pi*x/N. A unit-speed wave
in original simulation time t has angular time T=2*pi*t/N; therefore
pi_T=(N/(2*pi))*pi_t. For positive frequency exp(-i|k|T):
  c_plus(k) = 1/2 [fft(phi)(k) + i fft(pi_T)(k)/|k|] / N^2.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lifted_current import Modes, modes_example, observables
from paper_instanton import Instanton, positive_center, circular_center


def snapshot_free_reference(sim, template=None, min_amplitude=1e-10):
    """Band-project a nonlinear state AND its time derivative.

    Only 10 positive-frequency Fourier modes are retained by default.
    Their subsequent evolution under this reference is free, not phi^4.
    """
    if template is None:
        template=modes_example()
    n=sim.n
    phi_hat=np.fft.fft2(sim.phi)
    # Velocity is staggered by half a timestep (documented approximation).
    pi=np.asarray((sim.phi-sim.prev)/sim.dt)
    pi_T_hat=np.fft.fft2(pi)*n/(2*np.pi)
    c=[]
    for kx,kb,omega in zip(template.kx,template.kb,template.omega):
        ix=int(kx)%n
        ib=int(kb)%n
        c.append(.5*(phi_hat[ib,ix]+1j*pi_T_hat[ib,ix]/omega)/n**2)
    c=np.asarray(c,dtype=complex)
    raw_norm=float(np.sqrt(np.sum(np.abs(c)**2*template.omega**2)))
    # Equal reference-wave energies allow concentration comparisons
    # across snapshots despite different wave amplitudes.
    if raw_norm>min_amplitude:
        normalized=c/raw_norm
    else:
        normalized=np.zeros_like(c)
    return Modes(template.kx.copy(),template.kb.copy(),normalized),raw_norm


def run_bridge(*,steps=6000,every=600,n=128,probe_grid=12,
               disk=9,output="results/bridge"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if steps<=0 or every<=0:
        raise ValueError("steps and every must be positive")
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    sim=Instanton(n=n)
    template=modes_example()
    records=[]
    for i in range(steps+1):
        if i%every==0 or i==steps:
            modes,rawnorm=snapshot_free_reference(sim,template)
            obs,_=observables(modes,0.0,n=probe_grid,nw=disk)
            positive=np.maximum(sim.phi,0)
            comx,comy=positive_center(sim.phi)
            acx,acy=circular_center(sim.dynamic_activity())
            total_energy=float(sim.energy_density().sum())
            records.append(dict(
                nonlinear_t=float(sim.time),positive_com_x=comx,
                positive_com_y=comy,positive_mass=float(positive.sum()),
                activity_center_x=acx,activity_center_y=acy,
                nonlinear_total_energy=total_energy,
                free_projection_raw_norm=rawnorm,
                disk_response_slice=obs["disk_response_slice"],
                wave_L4=obs["wave_L4"],
                free_wave_energy=obs["physical_energy"],
                intensity_contrast=obs["physical_intensity_contrast"],
                lift_contrast=obs["lifted_time_density_peak_to_mean"],
                ratio_to_L4=obs["disk_response_slice"]/max(obs["wave_L4"],1e-30)
            ))
        if i < steps:
            sim.step()
    # Event flag is future CoM change; time-conditioned predictive
    # evaluation requires many independent runs, not this one trajectory.
    for i,r in enumerate(records):
        if i+1<len(records) and np.isfinite(r["positive_com_x"]) and \
           np.isfinite(records[i+1]["positive_com_x"]):
            r["next_abs_com_dx"]=abs(records[i+1]["positive_com_x"]-r["positive_com_x"])
        else:
            r["next_abs_com_dx"]=float("nan")

    with (folder/"nonlinear_reference.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
    t=[r["nonlinear_t"] for r in records]
    axes[0].plot(t,[r["positive_com_x"] for r in records],
                 label="Original positive-field CoM")
    axes[0].plot(t,[r["activity_center_x"] for r in records],
                 label="Original activity center")
    axes[0].legend()
    axes[0].set_ylabel("Original grid X")
    axes[0].set_title("No particle law added: original phi^4 observers")
    axes[1].plot(t,[r["disk_response_slice"] for r in records],
                 label="Paper lifted current -> auxiliary disk slice")
    axes[1].plot(t,[r["wave_L4"] for r in records],
                 label="Ordinary L4 of the same free-wave projection")
    axes[1].legend()
    axes[1].set_ylabel("Concentration (different units)")
    axes[1].set_title("Per-snapshot free-wave references, energy normalized")
    axes[2].plot(t,[r["ratio_to_L4"] for r in records],
                 label="Auxiliary response / ordinary L4")
    axes[2].legend()
    axes[2].set_xlabel("Original nonlinear simulation time")
    axes[2].set_ylabel("Relative diagnostic response")
    fig.savefig(folder/"nonlinear_paper_bridge.png",dpi=140)
    plt.close(fig)
    print(f"Bridge: {len(records)} nonlinear snapshots -> independently "
          f"reconstructed free-wave references; outputs: {folder}")
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps",type=int,default=6000)
    ap.add_argument("--every",type=int,default=600)
    ap.add_argument("--n",type=int,default=128)
    ap.add_argument("--probe-grid",type=int,default=12)
    ap.add_argument("--disk",type=int,default=9)
    ap.add_argument("--output",default="results/bridge")
    args=ap.parse_args()
    run_bridge(steps=args.steps,every=args.every,n=args.n,
               probe_grid=args.probe_grid,disk=args.disk,output=args.output)


if __name__=="__main__":
    main()
