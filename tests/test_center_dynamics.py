import json
import numpy as np
import pytest

from center_dynamics import (CenterTracker,center_info,derivative,
                             lag_correlations,periodic_delta,best_lag)
from real_paper_instanton import PlanarAppendixDObserver,run_live


def test_periodic_center_matches_wrapped_edge_not_cartesian_middle():
    w=np.zeros((16,16))
    w[8,0]=1
    w[8,15]=1
    p=center_info(w,periodic=True)
    c=center_info(w,periodic=False)
    assert c["x"]==pytest.approx(7.5)
    assert min(p["x"],16-p["x"])==pytest.approx(.5)
    assert p["y"]==pytest.approx(8)
    assert periodic_delta(.2,15.8,16)==pytest.approx(.4)


def test_undefined_low_mass_and_diffuse_center_are_rejected():
    zero=center_info(np.zeros((16,16)))
    assert not zero["valid"] and np.isnan(zero["x"])
    uniform=center_info(np.ones((16,16)))
    assert not uniform["valid"] and np.isnan(uniform["x"])
    localized=np.zeros((16,16))
    localized[6,7]=10
    assert center_info(localized)["valid"]


def test_lag_convention_observer_velocity_leads_green():
    t=np.linspace(0,30,160)
    a=np.sin(t)*np.exp(-.003*t)
    b=np.zeros_like(a)
    b[3:]=a[:-3] # source a at t leads target b at t+3
    curves=lag_correlations(a,b,limit=6,min_pairs=25)
    winning=best_lag(curves)
    assert winning["lag"]==3
    assert winning["correlation"]>0.999
    assert any(z["lag"]==0 for z in curves)


def test_tracker_records_controls_and_writes_strict_json(tmp_path):
    n=24
    tracker=CenterTracker(n,confidence=.1)
    p=np.zeros((n,n))
    p[12,7]=2
    p2=p.copy();p2[12,7]=1.99
    f=np.zeros_like(p);f[12,9]=3
    for t in range(5):
        rec=tracker.record(t*.2,p,p2,.02,f)
    assert rec["offset_x"]==pytest.approx(2)
    assert rec["offset_distance"]==pytest.approx(2)
    assert "gradient_x" in rec and "kinetic_x" in rec
    assert rec["green_vx"]==pytest.approx(0)
    assert rec["focus_vx"]==pytest.approx(0)
    assert np.isnan(rec["offset_along_green_motion"])
    summary=tracker.write(tmp_path,max_lag=3)
    assert summary["samples"]==5
    assert (tmp_path/"centers.csv").exists()
    with (tmp_path/"lead_lag.json").open() as fobj:
        data=json.load(fobj)
    assert data["velocity_cross_correlations"]["focus_to_green_x"]["best"] is None


def test_gemini_observer_finite_maps_small_grid():
    n=32
    observer=PlanarAppendixDObserver(n=n,nw=5,num_caps=2)
    xx,yy=np.meshgrid(np.arange(n),np.arange(n))
    p=np.exp(-((xx-9)**2+(yy-16)**2)/28)
    focus,mu=observer.observe(p,p*.995,.02)
    assert focus.shape==(n,n)==mu.shape
    assert np.isfinite(focus).all()
    assert np.isfinite(mu).all()
    assert np.min(focus)>=-1e-16
    assert mu.max()>0


def test_headless_two_center_experiment_saves_outputs(tmp_path):
    tracker=run_live(n=32,nw=5,caps=2,steps_per_frame=8,
                     frames=3,output=str(tmp_path),show=False,max_lag=3)
    assert len(tracker.rows)==3
    for name in ("centers.csv","lead_lag.json","center_dynamics.png"):
        assert (tmp_path/name).is_file()
