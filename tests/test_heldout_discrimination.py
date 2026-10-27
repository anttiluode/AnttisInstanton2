import json

import numpy as np
import pytest

from lifted_current import auxiliary_disk, disk_poisson_source, target_mu
from heldout_discrimination import (
    CachedSlice, ORDINARY, FAMILY, evaluate, make_example,
    matched_pairs, run, wave_features,
)


def test_cached_disk_matches_uncached_variable_coefficient_operator():
    m,_=make_example(4)
    s=CachedSlice(n=18,nw=9)
    mu=target_mu(m,0,s.xx,s.bb,s.W,tau=s.tau)
    expected_u,expected_energy=disk_poisson_source(
        mu*np.where(s.mask[...,None,None],1,0),
        s.W,s.mask,s.spacing,tau=s.tau)
    energy,u=s.response_from_mu(mu)
    assert energy==pytest.approx(expected_energy,rel=1e-9,abs=1e-11)
    assert np.allclose(u,expected_u,atol=1e-10,rtol=1e-9)
    assert energy>0
    assert len(s.factor_cache)>0
    assert all(np.isfinite(u.ravel()))


def test_independent_phase_families_share_exact_spectral_energy():
    x=[]
    seen=set()
    for i in range(20):
        m,fam=make_example(i)
        x.append(wave_features(m,0,18)["total_energy"])
        seen.add(fam)
        assert sum(abs(m.c)**2*m.omega**2)==pytest.approx(1,abs=1e-13)
        assert np.allclose(abs(m.c),abs(make_example(0)[0].c),
                           rtol=1e-14,atol=1e-14)
    assert set(FAMILY)==seen
    assert max(x)-min(x)<1e-12


def test_independent_wave_phases_change_future_outcome():
    a,_=make_example(0)
    b,_=make_example(1)
    now_a=wave_features(a,0,18)
    future_a=wave_features(a,.85,18)
    future_b=wave_features(b,.85,18)
    assert now_a["l4_now"]>0
    assert future_a["l4_now"]!=pytest.approx(future_b["l4_now"],rel=1e-3)


def test_heldout_train_test_split_does_not_overlap_and_is_fixed():
    rows=[]
    for i in range(24):
        m,f=make_example(i)
        features=wave_features(m,0,18)
        rows.append(dict(id=i,family=f,**features,disk=features["l4_now"]*.01+
                         (i%5)*.0005,
                         future_l4=wave_features(m,.85,18)["l4_now"]))
    r=evaluate(rows,bootstrap=20)
    assert r["train"]+r["test"]==24
    assert len(r["test_indices"])==r["test"]
    assert sum(x["split"]=="test" for x in rows)==r["test"]
    assert sum(x["split"]=="train" for x in rows)==r["train"]
    assert r["models"]["ordinary_plus_paper"]["rmse"]>=0
    assert r["mse_increment_bootstrap95"][0]<=r["mse_increment_bootstrap95"][1]


def test_matched_l4_pairs_no_fabricated_matches():
    rows=[{"l4_now":1.0,"disk":.01},
          {"l4_now":1.005,"disk":.015},
          {"l4_now":1.2,"disk":.015}]
    result=matched_pairs(rows)
    assert result["number_pairs"]==1
    assert result["median_paper_relative_difference"]>0.3


def test_full_smoke_writes_control_metrics(tmp_path):
    output=run(cases=24,n=18,disk=9,bootstrap=30,output=str(tmp_path))
    assert output["examples"]==24
    assert output["energy_drift_across_samples"]<1e-10
    for name in ("independent_waves.csv","heldout_metrics.json",
                 "paper_vs_L4.png","heldout_predictions.png"):
        assert (tmp_path/name).is_file()
    with (tmp_path/"heldout_metrics.json").open() as stream:
        stored=json.load(stream)
    assert "mse_increment_bootstrap95" in stored
    assert "ordinary_controls" in stored["models"]
    assert "ordinary_plus_paper" in stored["models"]
