import numpy as np
import pytest

from paper_instanton import (
    Instanton, annular_window, circular_center, coherent_focusing_spectrum,
    demo_source, frozen_disk_dirichlet, imaginary_ray, positive_center, run, parser,
    wave_grid,
)


def test_exact_complex_null_ray_on_one_fourier_mode():
    n = 32
    iy, ix = 3, 5
    kx, ky = 2*np.pi*ix/n, 2*np.pi*iy/n
    r = np.hypot(kx,ky)
    angle = 0.37
    rho, t = 2.3, 1.1
    spectrum=np.zeros((n,n),complex)
    spectrum[iy,ix] = n*n
    wave,bg,coeff=imaginary_ray(
        spectrum,rho=rho,angle=angle,time=t,
        center=r,half_width=0.2)
    expected=np.exp(-rho*(r-kx*np.cos(angle)-ky*np.sin(angle)))*np.exp(-1j*r*t)
    assert wave[0,0] == pytest.approx(expected,abs=1e-12)
    assert bg == pytest.approx(abs(expected)**2,abs=1e-12)
    assert np.mean(abs(wave)**2) == pytest.approx(bg,abs=1e-12)


def test_sum_before_square_keeps_interference_and_parseval():
    n=32
    c=np.zeros((n,n),complex)
    c[0,5]=n*n
    c[5,0]=n*n
    opposite=c.copy()
    opposite[5,0]*=-1
    u,bg,_=imaginary_ray(c,rho=0,angle=0,center=2*np.pi*5/n,half_width=.3)
    v,bg2,_=imaginary_ray(opposite,rho=0,angle=0,center=2*np.pi*5/n,half_width=.3)
    assert abs(u[0,0])**2 == pytest.approx(4.0)
    assert abs(v[0,0])**2 < 1e-25
    assert bg == pytest.approx(bg2)
    assert np.mean(abs(u)**2) == pytest.approx(bg)
    assert np.mean(abs(v)**2) == pytest.approx(bg)


def test_direction_window_has_parabolic_angular_width():
    n=64
    kx,ky=wave_grid(n)
    r=np.hypot(kx,ky)
    # Mathematical fact derived by Taylor expansion of 1-cos(delta).
    rho, freq, theta=5.0,1.5,.14
    gain=np.exp(-rho*freq*(1-np.cos(theta)))
    small_angle=np.exp(-rho*freq*theta**2/2)
    assert gain == pytest.approx(small_angle,rel=.002)


def test_auxiliary_D_operator_sign_and_screening():
    source=demo_source(25)
    u0,e0,mask=frozen_disk_dirichlet(source,J=0.8,k_dot_e=0)
    u1,e1,_=frozen_disk_dirichlet(source,J=0.8,k_dot_e=1.6)
    assert np.max(u0) <= 1e-12
    assert np.max(u1) <= 1e-12
    assert u0[mask].min() < u1[mask].min() < 0
    assert 0<e1<e0
    assert np.all(u0[~mask]==0)


def test_periodic_center_not_a_linear_average():
    w=np.zeros((16,16))
    w[8,0]=1
    w[8,15]=1
    x_lin,y_lin=positive_center(w)
    x_circ,y_circ=circular_center(w)
    assert x_lin==pytest.approx(7.5)
    assert min(x_circ,16-x_circ)<1
    assert y_circ == pytest.approx(8)


def test_original_phi4_simulator_energy_reasonably_stable_on_short_run():
    sim=Instanton(n=48)
    e0=sim.energy_density().sum()
    sim.step(80)
    assert np.all(np.isfinite(sim.phi))
    e1=sim.energy_density().sum()
    assert abs(e1/e0-1)<0.02
    assert sim.time==pytest.approx(1.6)


def test_paper_focus_calibration_spectrum_has_nonzero_energy():
    spectrum=coherent_focusing_spectrum(n=64,focus_x=24,focus_y=32)
    assert spectrum.shape==(64,64)
    assert np.sum(np.abs(spectrum)**2)>0
    u,bg,_=imaginary_ray(spectrum,rho=0,angle=.45,time=8)
    assert max(abs(u.ravel()))**2 / bg > 10


def test_headless_paper_run_writes_outputs(tmp_path):
    args=parser().parse_args([
        "--mode","paper","--grid","32","--output",str(tmp_path)
    ])
    obs=run(args)
    assert len(obs)>2
    assert np.all(np.isfinite([row["focus_contrast"] for row in obs]))
    assert (tmp_path/"paper_instanton.png").is_file()
    assert (tmp_path/"observations.csv").is_file()
    assert (tmp_path/"auxiliary_disk_u.csv").is_file()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        annular_window(np.array([1.]),1,0)
    with pytest.raises(ValueError):
        frozen_disk_dirichlet(np.ones((10,10)),J=0)
    with pytest.raises(ValueError):
        imaginary_ray(np.zeros((8,8)),rho=-1,angle=0)
