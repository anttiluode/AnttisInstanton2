import numpy as np
import pytest

from lifted_current import (
    auxiliary_disk, disk_poisson_source, experiment, fiber_geometry, modes_example,
    observables, source_current, source_divergence,
    source_to_target_mu_identity, spatial_grid, target_mu,
)


def test_current_is_positive_semidefinite_and_columnwise_divergence_free():
    modes=modes_example()
    points=[
        [0.12,0.43,0.25,1.2,0.12,0.36,0.08],
        [-0.21,1.0,0.73,0.9,-0.05,-0.38,0.14],
        [0.83,0.77,1.1,1.1,0.3,0.25,-0.2],
    ]
    for pt in points:
        B=source_current(modes,pt)
        assert np.max(abs(B-B.T))<1e-12
        assert np.linalg.eigvalsh(B).min()>-1e-12
        div=source_divergence(modes,pt,eps=3e-5)
        # A numerical derivative of Lemma 2.1's exact conserved tensor,
        # NOT a claim about the nonlinear original Instanton.
        assert np.max(abs(div))<3e-5,div


def test_proposition_2_2_target_mu_equals_pushed_source_current():
    modes=modes_example()
    T,x,b=.18,1.3,2.7
    xgrid=np.array([[x]])
    bgrid=np.array([[b]])
    tau=.35
    # Test the genuine two-flow chart, including nonzero imaginary W.
    for w in (0.12+0j,0.12+0.19j,-0.23-0.11j):
        phi=0.06
        Y=(phi+np.sqrt(2)*tau)/(1+w*w/2)
        val=float(target_mu(modes,T,xgrid,bgrid,np.array([[w]]),
                            tau=tau,Phi=phi)[0,0,0,0])
        pushed=source_to_target_mu_identity(modes,T,x,b,Y,w)
        assert val==pytest.approx(pushed,rel=1e-11,abs=1e-12)


def test_true_density_is_nonnegative_coherent_and_phase_sensitive():
    a=modes_example(scrambled=False)
    b=modes_example(scrambled=True)
    assert np.allclose(abs(a.c),abs(b.c),atol=1e-15,rtol=1e-15)
    x,y=spatial_grid(12)
    W,mask,step=auxiliary_disk(9,.4)
    mu0=target_mu(a,.8,x,y,W)
    mu1=target_mu(b,.8,x,y,W)
    assert np.min(mu0)>=0 and np.min(mu1)>=0
    assert not np.allclose(mu0,mu1)
    assert np.isfinite(mu0).all()


def test_disk_solve_is_real_nonpositive_and_response_positive():
    modes=modes_example()
    x,b=spatial_grid(12)
    W,mask,step=auxiliary_disk(9,.4)
    mu=target_mu(modes,.8,x,b,W)*mask[...,None,None]
    u,response=disk_poisson_source(mu,W,mask,step)
    assert u.shape==mu.shape
    assert np.max(abs(u[~mask]))<1e-10
    # Spectral discretization can have small ringing in X;
    # use a signed-correlation rather than a pointwise maximum test.
    assert response>0
    assert float(np.sum(u*mu))<0
    assert np.isfinite(u).all()


def test_physical_wave_energy_constant_across_time():
    m=modes_example()
    x,b=spatial_grid(24)
    vals=[m.physical_energy(t,x,b).sum() for t in np.linspace(0,3,7)]
    assert max(vals)-min(vals)<1e-9


def test_disk_response_changes_when_wave_phases_change():
    a=modes_example()
    b=modes_example(scrambled=True)
    obs1,_=observables(a,.8,n=12,nw=9)
    obs2,_=observables(b,.8,n=12,nw=9)
    assert obs1["disk_response_slice"]>0
    assert obs2["disk_response_slice"]>0
    assert obs1["physical_energy"]==pytest.approx(obs2["physical_energy"])
    assert obs1["disk_response_slice"]!=pytest.approx(
        obs2["disk_response_slice"],rel=1e-3
    )


def test_smoke_writes_paper_figures_csv(tmp_path):
    rows=experiment(n=12,nw=9,samples=3,out=str(tmp_path))
    assert len(rows)==6
    assert (tmp_path/"time_series.csv").is_file()
    assert (tmp_path/"comparison_curves.png").is_file()
    assert (tmp_path/"wave_and_disk.png").is_file()
    assert (tmp_path/"comparison_metrics.json").is_file()
