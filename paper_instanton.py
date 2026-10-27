"""Paper-first wave optics and nonlinear field laboratory.

OpenAI (2026), Local smoothing for the Euclidean wave equation:
* Section 1.4 / 2.1: exact imaginary null-ray multiplier
    exp[-rho |k| (1 - omega_hat dot k_hat)]
* Appendix D, (D.1): frozen-coefficient Fourier-mode disk operator
    4 |J|^2 Delta_W u - 4 |k dot e|^2 u = mu, u|boundary=0.

This does NOT construct the paper's conserved lifted tensor or prove local smoothing.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def wave_grid(n: int, dx: float = 1.0):
    """FFT angular frequencies, with array axes (y, x)."""
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(n, d=dx)[:, None]
    return np.broadcast_to(kx, (n, n)), np.broadcast_to(ky, (n, n))


def annular_window(r: np.ndarray, center: float, half_width: float):
    """Smooth, compactly supported Fourier-annulus multiplier."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    q = np.abs(r - center) / half_width
    return np.where(q < 1, np.cos(0.5 * np.pi * np.minimum(q, 1)) ** 2, 0.0)


def imaginary_ray(
    spectrum: np.ndarray, *,
    rho: float, angle: float, time: float = 0.0,
    center: float = 1.4, half_width: float = 0.6, dx: float = 1.0
):
    """Exactly evaluate free positive-frequency half-wave at a complex null ray.

    A mode exp[i(k.x-|k|t)] evaluated at (t,x)-i*rho*(1,omega)
    gains exp[-rho*(|k|-k.omega)]. Original mode phases are preserved.

    'spectrum' is the complex Fourier coefficient array of a FREE wave.
    Passing fft2(phi(t)) from a nonlinear snapshot is only a diagnostic,
    NOT analytic continuation of its nonlinear evolution.

    Returns (complex_field, spatially_uniform_incoherent_density,
             windowed_spectrum). For the FFT normalization used here,
    mean(|complex_field|**2) equals incoherent_density (Parseval).
    """
    if rho < 0:
        raise ValueError("rho must be >= 0")
    if spectrum.ndim != 2 or spectrum.shape[0] != spectrum.shape[1]:
        raise ValueError("spectrum must be a square 2D array")
    kx, ky = wave_grid(spectrum.shape[0], dx)
    radius = np.hypot(kx, ky)
    direction = kx * np.cos(angle) + ky * np.sin(angle)
    filter_gain = np.exp(-rho * np.maximum(0.0, radius - direction))
    annulus = annular_window(radius, center, half_width)
    coeff = spectrum * annulus * filter_gain * np.exp(-1j * radius * time)
    field = np.fft.ifft2(coeff)
    n = spectrum.shape[0]
    incoherent = float(np.sum(np.abs(coeff)**2) / n**4)
    return field, incoherent, coeff


def coherent_focusing_spectrum(
    n: int = 128, dx: float = 1.0, *,
    focus_x: float = 48.0, focus_y: float = 64.0, focus_time: float = 8.0,
    center: float = 1.4, spread: float = 0.22,
    direction: float = 0.45, angular_spread: float = 0.85
):
    """A calibration wave with engineered constructive focusing.

    This is deliberately synthetic: it demonstrates interference,
    not a discovery that the paper predicts particle jumps.
    """
    kx, ky = wave_grid(n, dx)
    r = np.hypot(kx, ky)
    orientation = np.arctan2(ky, kx)
    delta = np.angle(np.exp(1j * (orientation - direction)))
    envelope = np.exp(-0.5 * ((r - center) / spread)**2)
    envelope *= np.exp(-0.5 * (delta / angular_spread)**2)
    envelope[r == 0] = 0
    phase = -(kx * focus_x + ky * focus_y - r * focus_time)
    spectrum = envelope * np.exp(1j * phase)
    spectrum *= n*n / max(float(np.sum(envelope)), 1e-30)
    return spectrum


def positive_center(phi: np.ndarray):
    """Historical Antti observer: center of mass of max(phi,0), nonperiodic."""
    w = np.maximum(phi, 0.0)
    total = float(w.sum())
    if total <= 1e-14:
        return np.nan, np.nan
    yy, xx = np.indices(phi.shape)
    return float((xx*w).sum()/total), float((yy*w).sum()/total)


def circular_center(weights: np.ndarray):
    """Center on a periodic grid; NaN if spatial first moment cancels."""
    if weights.ndim != 2:
        raise ValueError("weights must be 2D")
    w = np.maximum(weights, 0)
    ny, nx = w.shape
    total = w.sum()
    if total <= 1e-14:
        return np.nan, np.nan
    ex = (w.sum(axis=0) * np.exp(2j*np.pi*np.arange(nx)/nx)).sum()/total
    ey = (w.sum(axis=1) * np.exp(2j*np.pi*np.arange(ny)/ny)).sum()/total
    if min(abs(ex), abs(ey)) < 1e-7:
        return np.nan, np.nan
    return float(np.angle(ex)%(2*np.pi)*nx/(2*np.pi)), float(np.angle(ey)%(2*np.pi)*ny/(2*np.pi))


@dataclass
class Instanton:
    """Original scalar phi^4 field, 2D torus; no independently moving particle."""
    n: int = 128
    dt: float = 0.02
    dx: float = 1.0
    c: float = 1.0
    a: float = 0.1
    b: float = 0.1

    def __post_init__(self):
        yy, xx = np.indices((self.n, self.n))
        radius = np.hypot(xx-self.n*0.25, yy-self.n*0.5)
        self.phi = 1.5 / np.cosh(radius/10.0)
        vx, vy = 0.25, 0.0
        dphidx = (np.roll(self.phi,-1,1)-np.roll(self.phi,1,1))/(2*self.dx)
        dphidy = (np.roll(self.phi,-1,0)-np.roll(self.phi,1,0))/(2*self.dx)
        velocity = -(vx*dphidx+vy*dphidy)
        self.prev = self.phi - velocity*self.dt
        self.time = 0.0

    def laplacian(self, p):
        return (np.roll(p,1,0)+np.roll(p,-1,0)+
                np.roll(p,1,1)+np.roll(p,-1,1)-4*p)/self.dx**2

    def step(self, steps: int = 1):
        for _ in range(steps):
            p = self.phi
            nxt = 2*p - self.prev + self.dt**2*(
                self.c**2*self.laplacian(p)+self.a*p-self.b*p**3)
            self.prev, self.phi = p, nxt
            self.time += self.dt

    def energy_density(self):
        """Nonnegative physical energy with vacuum offset V(min)=0.

        For b>0 and a>0: V(phi)+a^2/(4b) = b/4*(phi^2-a/b)^2.
        Staggered velocity introduces O(dt) drift in this diagnostic.
        """
        vy = (self.phi-self.prev)/self.dt
        gx = (np.roll(self.phi,-1,1)-self.phi)/self.dx
        gy = (np.roll(self.phi,-1,0)-self.phi)/self.dx
        return 0.5*vy**2+0.5*self.c**2*(gx**2+gy**2) + \
            0.25*self.b*(self.phi**2-self.a/self.b)**2

    def dynamic_activity(self):
        """Kinetic + gradient energy (not itself conserved)."""
        vy = (self.phi-self.prev)/self.dt
        gx = (np.roll(self.phi,-1,1)-self.phi)/self.dx
        gy = (np.roll(self.phi,-1,0)-self.phi)/self.dx
        return 0.5*vy**2+0.5*self.c**2*(gx**2+gy**2)


def frozen_disk_dirichlet(
    mu: np.ndarray, *, J: float = 1.0, k_dot_e: float = 0.0
):
    """Finite-difference solution of a FROZEN Appendix D Fourier-mode operator.

    For prescribed nonnegative mu(W), solve
        (4|J|^2 Laplacian_W - 4|k.e|^2)u=mu  inside |W|<1
        u=0 on boundary of the disk.

    W is an AUXILIARY disk, not the physical (x,y) field grid.
    The paper has variable coefficients, a genuine lifted A_TT source,
    and additional fiber geometry; those are NOT represented here.
    """
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 2 or mu.shape[0] != mu.shape[1] or mu.shape[0] < 5:
        raise ValueError("mu must be a square array of size >=5")
    if J == 0 or np.any(mu < 0) or not np.all(np.isfinite(mu)):
        raise ValueError("J must be nonzero and mu finite/nonnegative")
    n = mu.shape[0]
    coord = np.linspace(-1,1,n)
    yy, xx = np.meshgrid(coord,coord,indexing="ij")
    mask = (xx**2+yy**2) < (1.0-1e-12)
    h = coord[1]-coord[0]
    inds = -np.ones((n,n), dtype=int)
    inds[mask] = np.arange(mask.sum())
    mat = lil_matrix((int(mask.sum()),int(mask.sum())))
    rhs = mu[mask] / (4*abs(J)**2)
    screen = (abs(k_dot_e)/abs(J))**2
    for y,x in np.argwhere(mask):
        i = inds[y,x]
        mat[i,i] = -4/h**2 - screen
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx = y+dy,x+dx
            if 0 <= ny < n and 0 <= nx < n and mask[ny,nx]:
                mat[i,inds[ny,nx]] = 1/h**2
    u = np.zeros((n,n))
    u[mask] = spsolve(mat.tocsr(),rhs)
    probe = float(np.sum(-u*mu)*h**2)
    return u, probe, mask


def demo_source(n: int = 37, width: float = 0.21):
    c=np.linspace(-1,1,n)
    yy,xx=np.meshgrid(c,c,indexing="ij")
    mu=np.exp(-(xx**2+yy**2)/(2*width**2))
    mu[(xx**2+yy**2)>=1] = 0.0
    mu /= max(float(mu.sum()),1e-30)
    return mu


def measurement(spectrum, *, t, rho, angle, center, width, dx):
    wave, background, coeff = imaginary_ray(
        spectrum,rho=rho,angle=angle,time=t,center=center,half_width=width,dx=dx)
    intensity = np.abs(wave)**2
    y,x = np.unravel_index(intensity.argmax(),intensity.shape)
    return wave, intensity, background, (float(x),float(y)), float(intensity.max()/max(background,1e-30))


def run(args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    frame_data=[]
    dx=1.0
    if args.mode == "paper":
        spectrum=coherent_focusing_spectrum(
            args.grid,focus_x=args.grid*0.375,focus_y=args.grid*0.5,
            focus_time=args.focus_time,center=args.center)
        t = args.focus_time
        wave,intensity,bg,peak,contrast = measurement(
            spectrum,t=t,rho=args.rho,angle=args.angle,
            center=args.center,width=args.width,dx=dx)
        original = wave.real
        marker=positive_center(original)
        control_phase=np.random.default_rng(args.seed).uniform(-np.pi,np.pi,spectrum.shape)
        control=np.abs(spectrum)*np.exp(1j*control_phase)
        _,control_density,_,_,ctrl_ratio=measurement(
            control,t=t,rho=args.rho,angle=args.angle,
            center=args.center,width=args.width,dx=dx)
        times = np.linspace(t-4,t+4,33)
        for current_t in times:
            _,dens,baseline,p,contrast_t = measurement(
                spectrum,t=float(current_t),rho=args.rho,angle=args.angle,
                center=args.center,width=args.width,dx=dx)
            frame_data.append(dict(t=float(current_t),focus_x=p[0],
                focus_y=p[1],focus_contrast=contrast_t,
                positive_x=np.nan,activity_x=np.nan,
                total_energy=np.nan))
    else:
        sim=Instanton(n=args.grid)
        for j in range(args.steps+1):
            if j % args.sample_every == 0 or j == args.steps:
                snapshot=sim.phi.copy()
                spectrum_now=np.fft.fft2(snapshot-snapshot.mean())
                wave_i,intensity_i,bg_i,peak_i,contrast_i=measurement(
                    spectrum_now,t=0.0,rho=args.rho,angle=args.angle,
                    center=args.center,width=args.width,dx=dx)
                com=positive_center(snapshot)
                activity=com if not np.any(sim.dynamic_activity()) else circular_center(sim.dynamic_activity())
                frame_data.append(dict(t=sim.time,focus_x=peak_i[0],
                    focus_y=peak_i[1],focus_contrast=contrast_i,
                    positive_x=com[0],activity_x=activity[0],
                    total_energy=float(sim.energy_density().sum()*dx**2)))
            if j < args.steps:
                sim.step()
        spectrum=np.fft.fft2(sim.phi-sim.phi.mean())
        t=0.0  # frozen snapshot: free-wave filter, not the nonlinear solution
        wave,intensity,bg,peak,contrast=measurement(
            spectrum,t=t,rho=args.rho,angle=args.angle,
            center=args.center,width=args.width,dx=dx)
        original=sim.phi
        marker=positive_center(original)
        control_phase=np.random.default_rng(args.seed).uniform(-np.pi,np.pi,spectrum.shape)
        control=np.abs(spectrum)*np.exp(1j*control_phase)
        _,control_density,_,_,ctrl_ratio=measurement(
            control,t=t,rho=args.rho,angle=args.angle,
            center=args.center,width=args.width,dx=dx)

    angles=np.linspace(-np.pi,np.pi,72,endpoint=False)
    angular_contrast=[]
    for a in angles:
        _,den,bas,_,q=measurement(
            spectrum,t=t,rho=args.rho,angle=float(a),
            center=args.center,width=args.width,dx=dx)
        angular_contrast.append(q)

    mu=demo_source()
    u,probe,mask=frozen_disk_dirichlet(mu,J=1.0,k_dot_e=args.screen)
    np.savetxt(out/"auxiliary_disk_u.csv",u,delimiter=",")
    with (out/"observations.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(frame_data[0]))
        writer.writeheader()
        writer.writerows(frame_data)

    fig,axs=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    axs[0,0].imshow(original,origin="lower",cmap="RdBu_r")
    if np.isfinite(marker[0]):
        axs[0,0].plot(*marker,"go",label="positive-field CoM")
        axs[0,0].legend(fontsize=8)
    axs[0,0].set_title("Original field / synthetic free wave")

    im=axs[0,1].imshow(intensity,origin="lower",cmap="magma")
    axs[0,1].plot(*peak,"c+",markersize=11)
    axs[0,1].set_title("Paper: coherent ray |U(t-iρ(1,ω)) P_N f|²")
    fig.colorbar(im,ax=axs[0,1],fraction=.046)

    axs[0,2].imshow(control_density,origin="lower",cmap="magma",
                     vmin=0,vmax=max(float(intensity.max()),1e-20))
    axs[0,2].set_title("Same spectral power, scrambled phase")

    axs[1,0].plot(angles,angular_contrast)
    axs[1,0].axvline(args.angle,color="red",alpha=0.5,linestyle="--")
    axs[1,0].set(title="Paper null-ray angle scan",xlabel="ω angle (radians)",
                 ylabel="peak / mean filtered intensity")

    axs[1,1].plot([r["t"] for r in frame_data],
                  [r["focus_contrast"] for r in frame_data],label="ray focus contrast")
    axs[1,1].set(title="Focusing vs physical time",xlabel="time",
                 ylabel="peak / mean")
    if args.mode == "instanton":
        ax2=axs[1,1].twinx()
        ax2.plot([r["t"] for r in frame_data],
                 [r["positive_x"] for r in frame_data],
                 color="tab:orange",alpha=0.65,label="positive CoM X")
        ax2.set_ylabel("positive CoM X",color="tab:orange")

    img=axs[1,2].imshow(u,origin="lower",cmap="viridis",
                         extent=(-1,1,-1,1))
    axs[1,2].set_title("Appendix D frozen disk: u≤0\n"
                       f"∫(-u)μ dW = {probe:.3g}; screening={args.screen:g}")
    fig.colorbar(img,ax=axs[1,2],fraction=.046)
    note=("EXACT free half-wave reference" if args.mode == "paper" else
          "NONLINEAR phi⁴ field: instantaneous free-wave probe only")
    fig.suptitle("Antti's Instanton 2 | OpenAI local smoothing (2026) | "+note,
                 fontsize=13)
    fig.savefig(out/"paper_instanton.png",dpi=140)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"mode={args.mode} coherence contrast={contrast:.4g}; "
          f"phase-scrambled contrast={ctrl_ratio:.4g}; "
          f"disk energy={probe:.6g}; marker={marker}")
    print(f"Wrote {out/'paper_instanton.png'}, {out/'observations.csv'}, "
          f"{out/'auxiliary_disk_u.csv'}")
    return frame_data


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("paper","instanton"),default="paper")
    p.add_argument("--grid",type=int,default=128)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--sample-every",type=int,default=100)
    p.add_argument("--rho",type=float,default=4.0)
    p.add_argument("--angle",type=float,default=0.45,
                   help="direction omega, in radians")
    p.add_argument("--center",type=float,default=1.4,
                   help="frequency annulus center in angular units")
    p.add_argument("--width",type=float,default=0.6,
                   help="frequency annulus half-width")
    p.add_argument("--focus-time",type=float,default=8.0)
    p.add_argument("--screen",type=float,default=0.6,
                   help="absolute |k.e| for frozen Appendix D disk")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--output",default="results")
    p.add_argument("--show",action="store_true",
                   help="open Matplotlib window as well as saving figures")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
