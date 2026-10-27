# Paper-first experiment ledger

Source: OpenAI, *Local smoothing for the Euclidean wave equation* (October 25, 2026): §2.1–2.4, Proposition 2.2, Appendix D, Eq. (D.1).

## 2026-10-27: first actual lifted-current reference

Reproduction:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python lifted_current.py --n 12 --disk 9 --samples 3
python nonlinear_bridge.py --n 32 --steps 12 --every 6 --probe-grid 12 --disk 9
```

The GitHub Actions run at [this commit](https://github.com/anttiluode/AnttisInstanton2/actions/runs/40536576441) passed **18 tests** including independent conservation and source-target identity tests, source positivity, and full complex-\(W\) consistency.

### Small-grid finite-wave smoke result, matched spectral power

| Variable (T=0.8) | Coherent engineered focus | Phase-scrambled |
| --- | ---: | ---: |
| Paper's fixed-\(\Phi\) disk response | 0.0499513 | 0.0127898 |
| Ordinary spatial \(\int |u|^4\) | 2.30271 | 0.581495 |
| Total physical free-wave energy | 244.688 | 244.688 |

For this **six-measurement smoke** (three times, two phase arms), correlation of the disk response with ordinary \(L^4\) was **1.00000 when rounded to five decimals**; the relative standard deviation of the ratio (disk response / L4) was **0.00649**. The free-wave physical energy drift across the coherent case was at most **8.53e-14**, within floating point.

**Interpretation:** The actual tensor and auxiliary disk are now wired together and pass their identities, but the first numerical result offers **no evidence** that the paper-derived disk response distinguishes the two phases better than ordinary wave \(L^4\). This is a *negative discrimination result*, not failure to build the operator. The number of points is far too small to generalize beyond this calibration example.

### Nonlinear bridge status

The bridge successfully generated three sample snapshots of the unchanged original \(\phi^4\) equation. It rebuilds independent free-wave references from both \(\phi\) and \(\phi_t\), energy-normalizes those reference waves, and measures their Appendix D response and ordinary L4. This is not a validation of the conservation law on the nonlinear field. Predictive superiority has **not** been established.

### What remains genuinely untested

1. The paper's full auxiliary-time circle mean and the two-circle propagation estimate. We implemented a single fixed-\(\Phi\), fixed-\(\tau\) disk operator, not the integrated \(\mathcal F_{\rm pl}\).
2. Discrimination for ensembles of differently configured free waves, multiple caps, different observation windows, and held-out phase combinations.
3. Prequential/held-out prediction of field domain-wall rearrangements against ordinary energy flux, \(L^4\), phase coherence and strong autoregressive baselines. Do not infer this from a six-point smoke run.
4. A proof or verified numerical balance law for the full auxiliary-tensor field in the nonlinear \(\phi^4\) PDE. None is claimed.

**Kill condition:** If the fully coupled auxiliary quantity continues to be an almost fixed rescaling of \(L^4\) over independent waves and cannot improve held-out prediction beyond conventional controls, report that the paper gives a useful rigorous reference but no new diagnostic power for Antti's Instanton.
