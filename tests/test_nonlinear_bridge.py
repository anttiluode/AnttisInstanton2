import numpy as np
import pytest

from lifted_current import modes_example
from nonlinear_bridge import run_bridge, snapshot_free_reference
from paper_instanton import Instanton


def test_velocity_changes_positive_frequency_projection():
    sim=Instanton(n=32)
    a,norm_a=snapshot_free_reference(sim)
    assert np.isfinite(norm_a)
    sim.prev=sim.phi.copy()  # same spatial field but zero time derivative
    b,norm_b=snapshot_free_reference(sim)
    assert not np.allclose(a.c,b.c)
    assert np.allclose(a.kx,b.kx)


def test_bridge_reconstructs_wave_and_writes_csv(tmp_path):
    rows=run_bridge(steps=12,every=6,n=32,probe_grid=12,disk=9,output=str(tmp_path))
    assert len(rows)==3
    assert (tmp_path/"nonlinear_reference.csv").is_file()
    assert (tmp_path/"nonlinear_paper_bridge.png").is_file()
    assert all(np.isfinite(r["disk_response_slice"]) for r in rows)
    assert all(r["free_wave_energy"]>0 for r in rows)
    assert rows[0]["nonlinear_t"]==pytest.approx(0)
    assert rows[-1]["nonlinear_t"]==pytest.approx(.24)
