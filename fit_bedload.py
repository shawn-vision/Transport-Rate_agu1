"""
Fitting and model comparison for the bedload transport manuscript.

Produces every number quoted in Section 5 and in Texts S5-S6, plus fig2.pdf.
Run:  python fit_bedload.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
from scipy.optimize import curve_fit

RNG = np.random.default_rng(0)
THETA_C_LOC = 2.0 / 3.0          # single-grain instantaneous threshold (Bagnold balance)

df = pd.read_csv("data.csv")
T, Q, SRC = df["Gamma"].values, df["Q"].values, df["Sources"].values
y, x = np.log(Q), np.log(T)
n = len(y)
out = {"n": int(n), "theta_min": float(T.min()), "theta_max": float(T.max())}


def gof(resid, k):
    sse = float(np.sum(resid ** 2))
    return dict(R2=1 - sse / np.sum((y - y.mean()) ** 2),
                RMSE=float(np.sqrt(sse / n)),
                AIC=float(n * np.log(sse / n) + 2 * k), k=k)


# ---------------------------------------------------------------- 1. near-threshold asymptote
def asym(lnT, lnAf, gam):
    """ln Q* = ln A_f + (3/2) ln T - gamma/T   (two free parameters)"""
    return lnAf + 1.5 * lnT - gam / np.exp(lnT)


p_as, c_as = curve_fit(asym, x, y, p0=[2.0, 0.3])
se_as = np.sqrt(np.diag(c_as))
out["asymptote"] = dict(A_f=float(np.exp(p_as[0])),
                        A_f_lo=float(np.exp(p_as[0] - se_as[0])),
                        A_f_hi=float(np.exp(p_as[0] + se_as[0])),
                        gamma=float(p_as[1]), gamma_se=float(se_as[1]),
                        sigma0_implied=float(THETA_C_LOC / p_as[1]),
                        **gof(y - asym(x, *p_as), 2))

# ---------------------------------------------------------------- 2. independent high-Theta fit
# Stage one of the two-stage procedure: where suppression is negligible the data
# alone fix the transport capacity, and the exponent can be checked rather than
# assumed.  This is done first because stage two conditions on the result.
out["high_theta"] = {}
for cut in (0.5, 1.0, 2.0):
    m = T > cut
    slope, inter = np.polyfit(x[m], y[m], 1)
    out["high_theta"]["cut_%g" % cut] = dict(
        n=int(m.sum()),
        A_fixed_exponent=float(np.exp(np.mean(y[m] - 1.5 * x[m]))),
        free_exponent=float(slope), A_free=float(np.exp(inter)))
A_CAP = out["high_theta"]["cut_1"]["A_fixed_exponent"]   # capacity, Theta > 1


# ---------------------------------------------------------------- 3. full GEV exceedance
def gev_P(T_, mu0, s0, xi):
    """P(M_T > Theta_c^loc) for GEV block maxima with mu = mu0*T, sigma = s0*T."""
    if abs(xi) < 1e-9:
        return -np.expm1(-np.exp(-((THETA_C_LOC / T_) - mu0) / s0))
    z = np.clip(1.0 + xi * ((THETA_C_LOC / T_) - mu0) / s0, 1e-12, None)
    return -np.expm1(-z ** (-1.0 / xi))


def gev_at(T_, A_, mu0, s0, xi):
    return np.log(A_) + 1.5 * np.log(T_) + np.log(np.clip(gev_P(T_, mu0, s0, xi), 1e-300, None))


def gev(T_, mu0, s0, xi):
    """Stage two: capacity held at the independently determined A_CAP."""
    return gev_at(T_, A_CAP, mu0, s0, xi)


def fit_gev(Tv, yv, A_, p0=(7.2, 0.7, 0.14)):
    try:
        p, _ = curve_fit(lambda tt, m, ss, xx: gev_at(tt, A_, m, ss, xx),
                         Tv, yv, p0=list(p0), maxfev=200000)
        return p
    except Exception:
        return None


p_gev = fit_gev(T, y, A_CAP)
# the bootstrap resamples BOTH stages: the capacity is re-estimated inside each
# resample, so the reported intervals are not conditional on the full-sample A.
boot = []
for _ in range(400):
    i = RNG.integers(0, n, n)
    Ti, yi, xi_ = T[i], y[i], x[i]
    hi = Ti > 1
    if hi.sum() < 15:
        continue
    A_b = np.exp(np.mean(yi[hi] - 1.5 * xi_[hi]))
    start = tuple(p_gev * RNG.uniform(0.7, 1.4, 3))
    p = fit_gev(Ti, yi, A_b, start)
    if p is not None and 0.01 < p[1] < 50 and -1 < p[2] < 1.5:
        boot.append(p)
boot = np.array(boot)
ci = lambda j: [float(v) for v in np.percentile(boot[:, j], [2.5, 50, 97.5])]
out["gev"] = dict(A_capacity_fixed=float(A_CAP), mu0=float(p_gev[0]),
                  sigma0=float(p_gev[1]), xi=float(p_gev[2]),
                  mu0_ci=ci(0), sigma0_ci=ci(1), xi_ci=ci(2),
                  frac_xi_positive=float(np.mean(boot[:, 2] > 0)),
                  n_boot=int(len(boot)),
                  **gof(y - gev(T, *p_gev), 4))

# with the shape forced to the Gumbel limit the full (non-asymptotic) form runs
# away: report this, it is why xi = 0 survives only as an asymptote
def gum_full(T_, lnA, mu0, s0):
    return lnA + 1.5 * np.log(T_) + np.log(np.clip(
        -np.expm1(-np.exp(-((THETA_C_LOC / T_) - mu0) / s0)), 1e-300, None))


# xi <= XI_SMOOTH keeps Theta_c^loc inside the support of the fitted GEV over the
# whole data range, so the curve has no corner; it costs almost nothing in fit.
def _zmin(xi_):
    try:
        q, _ = curve_fit(lambda tt, m, ss: gev(tt, m, ss, xi_), T, y,
                         p0=[p_gev[0], p_gev[1]], maxfev=200000)
        return float(np.min(1 + xi_ * ((THETA_C_LOC / T) - q[0]) / q[1]))
    except Exception:
        return np.nan


grid = np.arange(0.05, 0.30, 0.002)
XI_BIND = float(next((xx for xx in grid if not np.isnan(_zmin(xx)) and _zmin(xx) <= 0), np.nan))
# keep a margin rather than sitting on the endpoint: the plotted curve uses the
# largest shape for which the support boundary stays well clear of the data.
XI_SMOOTH = float(next((xx for xx in grid[::-1]
                        if not np.isnan(_zmin(xx)) and _zmin(xx) >= 0.40), 0.05))
p_sm, _ = curve_fit(gev, T, y, p0=[7.0, 0.8, 0.05], maxfev=300000,
                    bounds=([-50, 0.05, 0.0], [60, 20, XI_SMOOTH]))
zmin = float(np.min(1 + p_sm[2] * ((THETA_C_LOC / T) - p_sm[0]) / p_sm[1]))
out["gev_smooth"] = dict(xi_binding=XI_BIND, mu0=float(p_sm[0]), sigma0=float(p_sm[1]), xi=float(p_sm[2]),
                         min_support_z=zmin, **gof(y - gev(T, *p_sm), 3))

p_ctl, _ = curve_fit(lambda tt, m, ss: gev(tt, m, ss, 0.0), T, y,
                     p0=[1.0, 2.0], maxfev=200000)
out["gumbel_capacity_fixed"] = dict(mu0=float(p_ctl[0]), sigma0=float(p_ctl[1]),
                                    **gof(y - gev(T, p_ctl[0], p_ctl[1], 0.0), 3))

p_gf, _ = curve_fit(gum_full, T, y, p0=[2.5, 3.0, 1.0], maxfev=200000)
out["gumbel_full_runaway"] = dict(A=float(np.exp(p_gf[0])), mu0=float(p_gf[1]),
                                  sigma0=float(p_gf[2]), **gof(y - gum_full(T, *p_gf), 3))

# ---------------------------------------------------------------- 4. competing models
models = {}
Apow = np.exp(np.mean(y - 1.5 * x))
models["power_fixed_1.5"] = (np.log(Apow) + 1.5 * x, 1, dict(A=float(Apow)))
pf = np.polyfit(x, y, 1)
models["power_free"] = (np.polyval(pf, x), 2, dict(exponent=float(pf[0]), A=float(np.exp(pf[1]))))


def mpm(T_, A, Tc):
    return np.log(A) + 1.5 * np.log(np.clip(T_ - Tc, 1e-12, None))


# MPM is given its best possible chance under the SAME objective used to score
# it: Theta_c is chosen on a grid to minimise SSE over the points the form can
# represent, subject to representing all of them (Theta_c < min(Theta)), so the
# comparison is not an artefact of charging it for points it is undefined at.
best = None
for Tc in np.linspace(0.0, T.min() * 0.999, 400):
    lnA = np.mean(y - 1.5 * np.log(T - Tc))
    sse = np.sum((y - lnA - 1.5 * np.log(T - Tc)) ** 2)
    if best is None or sse < best[0]:
        best = (sse, Tc, np.exp(lnA))
_, Tc_mpm, A_mpm = best
p_mpm = np.array([A_mpm, Tc_mpm])
# how much of the data a conventionally-thresholded MPM has to discard
conv = {("%.3f" % tc): int((T <= tc).sum()) for tc in (0.03, 0.047, 0.06)}
models["MPM"] = (mpm(T, *p_mpm), 1, dict(
    A=float(A_mpm), Theta_c=float(Tc_mpm), n_representable=int(n),
    excluded_if_conventional_Tc=conv))

# Head-to-head on the common domain where every candidate applies. Each model is
# REFITTED there, so this compares the forms at their best rather than penalising
# one for parameters chosen to suit a range it is no longer being scored on.
common = T > 0.06
Tc_, yc_, xc_ = T[common], y[common], x[common]
sst = np.sum((yc_ - yc_.mean()) ** 2)
out["common_domain"] = {"theta_min": 0.06, "n": int(common.sum())}


def score_common(resid, extra=None):
    d = dict(R2=float(1 - np.sum(resid ** 2) / sst), RMSE=float(np.sqrt(np.mean(resid ** 2))))
    if extra:
        d.update(extra)
    return d


pa_c, _ = curve_fit(asym, xc_, yc_, p0=[2.0, 0.3])
out["common_domain"]["asymptote"] = score_common(yc_ - asym(xc_, *pa_c))
pm_c, _ = curve_fit(mpm, Tc_, yc_, p0=[10.0, 0.03], maxfev=50000)
out["common_domain"]["MPM"] = score_common(yc_ - mpm(Tc_, *pm_c),
                                           dict(A=float(pm_c[0]), Theta_c=float(pm_c[1])))
pf_c = np.polyfit(xc_, yc_, 1)
out["common_domain"]["power_free"] = score_common(yc_ - np.polyval(pf_c, xc_),
                                                  dict(exponent=float(pf_c[0])))
Ap_c = np.exp(np.mean(yc_ - 1.5 * xc_))
out["common_domain"]["power_fixed_1.5"] = score_common(yc_ - np.log(Ap_c) - 1.5 * xc_)

# mean suppression over the subset used to set the capacity: this, not
# exp(-gamma/1), is what the model predicts for the offset between A and A_f
hi = T > 1
out["capacity_bias"] = dict(
    mean_suppression=float(np.exp(np.mean(-p_as[1] / T[hi]))),
    observed_ratio=float(out["high_theta"]["cut_1"]["A_fixed_exponent"] / np.exp(p_as[0])),
    theta_geomean=float(np.exp(np.mean(np.log(T[hi])))))
out["competing"] = {k: dict(gof(y - pred, k_), **extra) for k, (pred, k_, extra) in models.items()}
out["competing"]["asymptote"] = {kk: out["asymptote"][kk] for kk in ("R2", "RMSE", "AIC", "k")}
out["competing"]["gev"] = {kk: out["gev"][kk] for kk in ("R2", "RMSE", "AIC", "k")}
amin = min(v["AIC"] for v in out["competing"].values())
for v in out["competing"].values():
    v["dAIC"] = v["AIC"] - amin

# ---------------------------------------------------------------- 5. residual diagnostics
bands = [(0, 0.03), (0.03, 0.06), (0.06, 0.2), (0.2, 1), (1, 100)]
out["residuals"] = {}
for lab, pred in (("asymptote", asym(x, *p_as)), ("gev", gev(T, *p_gev))):
    r = y - pred
    out["residuals"][lab] = [dict(lo=lo, hi=hi, n=int(((T >= lo) & (T < hi)).sum()),
                                  bias=float(r[(T >= lo) & (T < hi)].mean()),
                                  rmse=float(np.sqrt(np.mean(r[(T >= lo) & (T < hi)] ** 2))))
                             for lo, hi in bands]
out["by_source"] = [dict(source=str(s), n=int((SRC == s).sum()),
                         theta_lo=float(T[SRC == s].min()), theta_hi=float(T[SRC == s].max()),
                         rmse=float(np.sqrt(np.mean((y - asym(x, *p_as))[SRC == s] ** 2))))
                    for s in np.unique(SRC)]

with open("fit_results.json", "w") as fh:
    json.dump(out, fh, indent=2)

# ---------------------------------------------------------------- 6. Figure 2
Tg = np.logspace(-2.2, 1.05, 500)
cap = A_CAP * Tg ** 1.5
gevc = np.exp(gev(Tg, *p_sm))
asc = np.exp(asym(np.log(Tg), *p_as))
mpmc = np.where(Tg > pm_c[1], pm_c[0] * np.clip(Tg - pm_c[1], 1e-12, None) ** 1.5, np.nan)

plt.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(12.6, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1], hspace=0.42, wspace=0.24)
axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, 1])

marks = ["d", "o", "+", "D", "s"]
cols = ["#2e7d32", "#1565c0", "#212121", "#c62828", "#7b1fa2"]
for ax in (axA, axB, axC):
    for i, s in enumerate(np.unique(SRC)):
        m = SRC == s
        ax.loglog(T[m], Q[m], marks[i % 5], mfc="none", ms=6.5, mew=1.3,
                  color=cols[i % 5], ls="none", label=s if ax is axA else None)
    ax.loglog(Tg, cap, ":", color="#1565c0", lw=2.3,
              label=r"capacity $A\,\Theta^{3/2}$" if ax is axA else None)
    ax.loglog(Tg, mpmc, ":", color="#00838f", lw=2.3,
              label=r"$(\Theta-\Theta_c)^{3/2}$, fitted on $\Theta>0.06$" if ax is axA else None)
    ax.loglog(Tg, gevc, "-", color="#8d6e63", lw=2.1,
              label=r"full GEV closure, $\xi>0$ (Text S6)" if ax is axA else None)
    ax.loglog(Tg, asc, "--", color="#000000", lw=3.2,
              label=r"$A_f\,\Theta^{3/2}e^{-\gamma/\Theta}$, fitted" if ax is axA else None)
    ax.grid(True, which="major", ls="--", lw=0.5, alpha=0.55)

axA.set_xlim(1e-2, 1.1e1); axA.set_ylim(2e-10, 1e4)
axA.set_xlabel(r"$\Theta$"); axA.set_ylabel(r"$Q_*$")
axA.legend(frameon=True, fontsize=10.5, ncol=1, loc="lower right", borderaxespad=0.5,
           handlelength=2.2, labelspacing=0.35, framealpha=0.92, borderpad=0.45)
axA.text(0.03, 0.95, "A", transform=axA.transAxes, fontsize=18, fontweight="bold", va="top")
axB.set_xlim(1e-1, 1.1e1); axB.set_ylim(1e-2, 6e2)
axB.text(0.04, 0.92, "B", transform=axB.transAxes, fontsize=17, fontweight="bold", va="top")
axC.set_xlim(1.2e-2, 1.1e-1); axC.set_ylim(1e-10, 3e0)
axC.xaxis.set_minor_formatter(NullFormatter())
axC.xaxis.set_major_locator(FixedLocator([0.02, 0.04, 0.06, 0.10]))
axC.xaxis.set_major_formatter(FixedFormatter(["0.02", "0.04", "0.06", "0.10"]))
axC.set_xlabel(r"$\Theta$")
axC.text(0.04, 0.92, "C", transform=axC.transAxes, fontsize=17, fontweight="bold", va="top")
for ax in (axB, axC):
    ax.set_ylabel(r"$Q_*$")
fig.savefig("fig2.pdf", bbox_inches="tight")
fig.savefig("fig2.png", dpi=170, bbox_inches="tight")

# ---------------------------------------------------------------- 7. console summary
a, g = out["asymptote"], out["gev"]
print("N = %d,  Theta in [%.4f, %.2f]" % (out["n"], out["theta_min"], out["theta_max"]))
print("\nnear-threshold asymptote  Q* = A_f T^1.5 exp(-gamma/T)")
print("   A_f    = %.2f  (%.2f - %.2f)   [ln A_f = %.3f]" % (a["A_f"], a["A_f_lo"], a["A_f_hi"], np.log(a["A_f"])))
print("   gamma  = %.4f +/- %.4f   -> sigma_0 = Theta_c^loc/gamma = %.2f" % (a["gamma"], a["gamma_se"], a["sigma0_implied"]))
print("   R2 = %.4f  RMSE = %.3f" % (a["R2"], a["RMSE"]))
print("\nfull GEV closure, capacity fixed at A = %.2f ; %d bootstrap resamples" % (g["A_capacity_fixed"], g["n_boot"]))
for k, lab in (("mu0_ci", "mu_0"), ("sigma0_ci", "sigma_0"), ("xi_ci", "xi")):
    lo, med, hi = g[k]
    print("   %-8s %7.3f   95%% CI [%.3f, %.3f]" % (lab, med, lo, hi))
print("   fraction of resamples with xi > 0 : %.0f%%" % (100 * g["frac_xi_positive"]))
print("   R2 = %.4f  RMSE = %.3f   dAIC vs asymptote = %.1f" % (g["R2"], g["RMSE"], g["AIC"] - a["AIC"]))
print("   Gumbel limit xi=0 with A ALSO free: A -> %.3g, mu_0 -> %.1f  (runaway)"
      % (out["gumbel_full_runaway"]["A"], out["gumbel_full_runaway"]["mu0"]))
gc = out["gumbel_capacity_fixed"]
print("   control, xi=0 but capacity FIXED: mu_0=%.2f sigma_0=%.2f R2=%.4f  (identifiable -> the degeneracy is removed by fixing A, not by freeing xi)"
      % (gc["mu0"], gc["sigma0"], gc["R2"]))
sm = out["gev_smooth"]
print("   constrained xi<=%.3f (smooth, used in Fig. 2): mu_0=%.2f sigma_0=%.3f xi=%.3f  R2=%.4f  dAIC=%+.1f  min support z=%+.3f"
      % (XI_SMOOTH, sm["mu0"], sm["sigma0"], sm["xi"], sm["R2"], sm["AIC"] - a["AIC"], sm["min_support_z"]))
print("\nindependent high-Theta fits")
for k, v in out["high_theta"].items():
    print("   %-8s n=%3d   A(3/2 fixed)=%5.2f   free exponent = %.2f" % (k, v["n"], v["A_fixed_exponent"], v["free_exponent"]))
print("\nmodel comparison (log space)")
for k, v in sorted(out["competing"].items(), key=lambda kv: kv[1]["dAIC"]):
    print("   %-18s k=%d  R2=%7.4f  RMSE=%6.3f  dAIC=%7.1f" % (k, v["k"], v["R2"], v["RMSE"], v["dAIC"]))
m = out["competing"]["MPM"]
print("   best full-coverage MPM: A=%.2f, Theta_c=%.4f (below min Theta, so all %d points representable)"
      % (m["A"], m["Theta_c"], m["n_representable"]))
print("   points a conventionally-thresholded MPM must discard: " +
      ", ".join("Theta_c=%s -> %d" % (k, v) for k, v in sorted(m["excluded_if_conventional_Tc"].items())))
print("   dAIC of MPM relative to the asymptote: %+.0f" % (m["dAIC"] - out["competing"]["asymptote"]["dAIC"]))
print("\nresiduals by band (asymptote / GEV)")
for ra, rg in zip(out["residuals"]["asymptote"], out["residuals"]["gev"]):
    print("   %.3g-%-5.3g n=%3d   bias %+5.2f / %+5.2f    rmse %.2f / %.2f"
          % (ra["lo"], ra["hi"], ra["n"], ra["bias"], rg["bias"], ra["rmse"], rg["rmse"]))
print("\nwrote fit_results.json, fig2.pdf, fig2.png")

cb = out["capacity_bias"]
print("\ncapacity offset: model predicts A/A_f = %.3f over Theta>1 (geometric-mean Theta = %.2f); observed %.3f"
      % (cb["mean_suppression"], cb["theta_geomean"], cb["observed_ratio"]))
print("\ncommon-domain head-to-head (Theta > %.2f, n = %d)" % (out["common_domain"]["theta_min"], out["common_domain"]["n"]))
for k in ("asymptote", "MPM", "power_free", "power_fixed_1.5"):
    v = out["common_domain"][k]
    print("   %-18s R2=%7.4f  RMSE=%.3f" % (k, v["R2"], v["RMSE"]))
