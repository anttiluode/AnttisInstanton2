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


## 2026-10-27: larger held-out wave-discrimination experiment (256 independent waves)

**Question:** Does the implemented true lifted wave-current density plus Appendix D fixed-\(\Phi\) slice offer predictive information beyond ordinary scalar wave concentration, energy flux, and phase summaries?

Run with the fixed, committed protocol:

```powershell
python heldout_discrimination.py --cases 256 --n 18 --disk 9 --bootstrap 1000
```

[Verified GitHub Actions run](https://github.com/anttiluode/AnttisInstanton2/actions/runs/40538711838). All 24 tests passed and both the 24-wave smoke and 256-wave full benchmark completed. The experiment is deterministic: RNG seeds 1401 for wave generation and 771 for holdout split, 166 independent training waves and 90 held-out independent test waves, 4 predeclared phase families. Modes have the same spectral powers across *every* wave; physical wave energy varies by at most \(5.68\times10^{-14}\) numerically.

**Primary target:** \(\log \int |u(T+0.85,X)|^4\,dX\), using measurements at \(T=0\) only.

| Train-fitted feature set | Held-out RMSE | Held-out R² |
|---|---:|---:|
| Sample-mean predictor | 0.380119 | -0.012243 |
| Ordinary \(L^4(T=0)\) only | 0.058256 | 0.976225 |
| Ordinary controls (\(L^4\), peak contrast, RMS/peak energy flux, two phase-locking summaries) | **0.040245** | **0.988653** |
| Ordinary controls + genuine paper-derived auxiliary slice response | 0.040435 | 0.988546 |
| Paper-derived auxiliary slice response alone | 0.058355 | 0.976144 |

Primary paired held-out test: baseline squared error **minus** augmented squared error \(=-1.529\times10^{-5}\); paired bootstrap 95% interval \([-3.462\times10^{-5},\, 6.702\times10^{-6}]\). **No demonstrated improvement.**

**Secondary target:** log future peak-to-mean physical intensity contrast. Ordinary RMSE 0.043420, ordinary+paper 0.043346. Difference in mean squared errors \(=+6.361\times10^{-6}\), bootstrap 95% interval \([-2.169\times10^{-5},\,3.488\times10^{-5}]\). Also **inconclusive**.

**Redundancy control:** Across the 256 independently generated waves, the paper's fixed-\(\Phi\) disk response correlates with ordinary *present* \(L^4\) at **0.999714**. Relative coefficient of variation of disk/L4 is **0.01216**. Among **602** same-power pairs whose present L4 values differ by at most 1%, the implementation also explicitly measures how much auxiliary response changes; see `results/heldout/heldout_metrics.json` for full, reproducible pair statistics. These pairs can establish some mathematical non-equivalence; they do *not* by themselves establish predictive advantage.

**Outcome:** We built the positive conserved source tensor from Lemma 2.1, validated its divergence numerically, used the paper's actual \(A^0_{TT}\) from Proposition 2.2, and solved the variable-coefficient Appendix D fiber problem. Under this benchmark, that specific *single-slice derived scalar* is close to rescaled ordinary \(L^4\) and has **no reliable held-out predictive advantage** over the listed ordinary controls.

**Scope limitations:** The simulated source is a **free wave**, not the nonlinear Instanton; fixed-\(\Phi\) auxiliary slice is **not** the paper's full \(F_{\rm pl}\), and the two-circle maximum-principle argument is not reproduced. The same fixed Fourier powers and intentionally structured phase families constrain the result. Positive discrimination in other ensembles is not excluded. Neither method can add information beyond the full wave coefficients; the meaningful question is whether compact derived observables outperform compact conventional ones, under fair controls.

**Next gate:** Unless a more faithful fiber integration or a carefully motivated independent system produces an improvement against ordinary \(L^4\), flux and phase statistics, this paper-to-Instanton **discriminative-observer hypothesis should be regarded as unsupported**, not as a discovery.
