"""End-to-end check of scripts/analysis/uncertainty_fair.py on synthetic runs
whose answers are known: an informative sigma_t against an uninformative
inlier count, a covariance overconfident by exactly 4x, and a 10% failure
component the defensive mixture should absorb."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ANALYSIS = os.path.join(HERE, "..", "scripts", "analysis")
sys.path.insert(0, ANALYSIS)
import pose_unc_np as pu  # noqa: E402
import uncertainty_fair as uf  # noqa: E402

TAU_TRUE = 4.0


def _spd(rng, scale):
    A = rng.normal(size=(6, 6))
    S = A @ A.T / 6 + 0.2 * np.eye(6)
    return S * scale ** 2


def _write_runs(root, n=60, seed=0):
    rng = np.random.default_rng(seed)
    ours, ace = root / "ours", root / "ace"
    ours.mkdir()
    ace.mkdir()
    for s in uf.SC:
        recs, arecs = [], []
        for i in range(n):
            scale = 10 ** rng.uniform(-3, -1.5)            # 1 mm .. 3 cm, rad alike
            S_true = _spd(rng, scale)
            fail = rng.random() < 0.10
            e = (np.concatenate([rng.normal(0, 0.3, 3), rng.normal(0, 0.05, 3)]) if fail
                 else rng.multivariate_normal(np.zeros(6), S_true))
            T_gt = pu.se3_exp(rng.normal(0, 0.5, 6))
            T = pu.se3_inverse(pu.se3_exp(e)) @ T_gt          # e = log(T_gt T^-1)
            S_rep = S_true / TAU_TRUE                         # 4x overconfident
            d2 = pu.mahalanobis_sq(pu.relative_tangent(T, T_gt), S_rep)
            recs.append({"seq": "seq-01", "frame_id": i,
                         "t_err": pu.translation_error(T, T_gt), "r_err": 0.0,
                         "d2": d2, "logdet": float(np.linalg.slogdet(S_rep)[1]),
                         "sigma_t": float(np.sqrt(np.trace(S_rep[:3, :3]))),
                         "inliers": int(rng.integers(20, 400)),   # pure noise
                         "max_inliers": int(rng.integers(20, 400)),
                         "T": T.tolist(), "Sigma": S_rep.tolist(), "T_gt": T_gt.tolist()})
            S_a = _spd(rng, 10 ** rng.uniform(-2.5, -2))
            ea = rng.multivariate_normal(np.zeros(6), S_a)
            Ta = pu.se3_inverse(pu.se3_exp(ea)) @ T_gt
            arecs.append({"seq": "seq-01", "frame_id": i, "failed": False,
                          "t_err": pu.translation_error(Ta, T_gt), "r_err": 0.0,
                          "inliers": int(rng.integers(100, 2000)), "T": Ta.tolist(),
                          "Sigma_pnp": (S_a / 9.0).tolist()})
        json.dump({"args": {"width": 640, "height": 480, "corr_matcher": "stage",
                            "corr_stage_warm": "mast3r", "seed": 0},
                   "tau": TAU_TRUE, "records": recs, "corrector_query": "rgb"},
                  open(ours / f"7scenes_{s}.json", "w"))
        json.dump({"scene": s, "records": arecs}, open(ace / f"7scenes_{s}.json", "w"))
    return ours, ace


class _Args:
    scenes = uf.SC
    boot = 40
    mc = 400
    seed = 0
    no_mixture = False


def test_known_answers(tmp_path):
    ours, ace_dir = _write_runs(tmp_path)
    rep = uf.analyze(str(ours), uf.load_ace(str(ace_dir), uf.SC), _Args())

    A = rep["A"]["metrics"]
    # informative sigma_t beats a random inlier count on both ranking metrics
    assert A["sigma_t (ours)"]["ause"] < A["-inliers (dominant hyp.)"]["ause"]
    assert A["sigma_t (ours)"]["eaurc_2cm"] < A["-inliers (dominant hyp.)"]["eaurc_2cm"]
    assert A["sigma_t (ours)"]["spearman"] > 0.3

    C = rep["C"]["ours"]
    # raw covariance is 4x overconfident; the median-rule LOSO temperature
    # must find ~4 and restore the 68% level (failures only hit the tails)
    for t in rep["C"]["loso_tau"].values():
        assert 2.5 < t < 6.5
    assert C["raw"]["cov68.3"] < 45
    assert abs(C["LOSO tau"]["cov68.3"] - 68.3) < 10
    # the defensive mixture absorbs the 10% failures in the 95.4/99.7 tails
    assert abs(C["LOSO mixture"]["cov95.4"] - 95.4) < 6
    assert C["LOSO mixture"]["nll_mean"] < C["LOSO tau"]["nll_mean"]
    for p in rep["C"]["loso_mixture"].values():
        assert 0.03 < p[0] < 0.25
    # conformal is valid by construction on exchangeable scenes
    assert abs(C["LOSO conformal"]["cov95.4"] - 95.4) < 6
    assert rep["verdict"] in ("PASS", "KILL", "INCONCLUSIVE")
    assert "ace" in rep["C"]


def test_cli_writes_json(tmp_path):
    ours, ace_dir = _write_runs(tmp_path, n=25, seed=1)
    out = tmp_path / "fair.json"
    r = subprocess.run([sys.executable, os.path.join(ANALYSIS, "uncertainty_fair.py"),
                        str(ours), "--ace", str(ace_dir), "--boot", "20", "--mc", "200",
                        "--json", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "VERDICT" in r.stdout
    rep = json.load(open(out))
    assert rep[0]["verdict"] in ("PASS", "KILL", "INCONCLUSIVE")


# --------------------------------------------------------------------------
# regressions for the review findings
# --------------------------------------------------------------------------
def _rewrite(run_dir, fn):
    for s in uf.SC:
        p = run_dir / f"7scenes_{s}.json"
        if p.exists():
            j = json.load(open(p))
            fn(s, j)
            json.dump(j, open(p, "w"))


def test_ranks_tempered_sigma_like_the_paper(tmp_path):
    """Per-scene taus differ: the pooled AUSE must be pooled_calib.py's
    (tempered) number, not the raw one."""
    import pooled_calib
    ours, _ = _write_runs(tmp_path)
    taus = dict(zip(uf.SC, (3.0, 6.0, 2.0, 25.0, 10.0, 12.0, 60.0)))
    _rewrite(ours, lambda s, j: j.__setitem__("tau", taus[s]))
    S, T = [], []
    for s in uf.SC:
        j = json.load(open(ours / f"7scenes_{s}.json"))
        for r in j["records"]:
            S.append(r["sigma_t"] * j["tau"] ** 0.5)
            T.append(100 * r["t_err"])
    ref = pooled_calib.ause(np.array(S), np.array(T))
    a = _Args()
    a.no_mixture = True
    rep = uf.analyze(str(ours), {}, a)
    assert abs(rep["A"]["metrics"]["sigma_t (ours)"]["ause"] - ref) < 1e-9


def test_missing_inliers_and_partial_ace_are_not_scored(tmp_path):
    ours, ace_dir = _write_runs(tmp_path, n=30, seed=2)

    def drop(s, j):
        for r in j["records"]:
            r.pop("inliers"), r.pop("max_inliers")
    _rewrite(ours, drop)
    (ace_dir / "7scenes_stairs.json").unlink()            # ACE lacks one scene
    a = _Args()
    a.boot, a.mc = 20, 100
    rep = uf.analyze(str(ours), uf.load_ace(str(ace_dir), uf.SC), a)
    assert not any("inliers" in k for k in rep["A"]["rel"])
    assert rep["C"]["nll_ace_minus_ours"]["incomplete"] == ["stairs"]
    assert rep["verdict"] == "INCOMPLETE"


def test_single_scene_and_no_failures_do_not_crash(tmp_path):
    ours, ace_dir = _write_runs(tmp_path, n=40, seed=3)

    def shrink(s, j):                                      # all errors < 2 cm
        for r in j["records"]:
            r["t_err"] = min(r["t_err"], 0.015)
    _rewrite(ours, shrink)
    a = _Args()
    a.scenes, a.boot, a.mc = ["chess"], 20, 100
    rep = uf.analyze(str(ours), uf.load_ace(str(ace_dir), ["chess"]), a)
    assert np.isnan(rep["C"]["ours"]["LOSO tau"]["cov95.4"])
    assert rep["verdict"] in ("INCOMPLETE", "INCONCLUSIVE", "KILL")


def test_insignificant_advantage_is_not_a_kill():
    """+2 nats with a CI touching 0, 4/7 scene wins and ACE coverage slightly
    closer to nominal is not 'ACE within 10 %': the gate must not KILL."""
    rel = {(b, m): (0.30, 0.2, 0.4) for b in uf.INLIER_BASES for m in ("ause", "eaurc_2cm")}
    C = {"LOSO tau": {"cov95.4": 93.0}}
    nll_cmp = {"incomplete": [], "LOSO tau": (2.0, -0.05, 4.0), "eaurc2_scene_wins": 4,
               "n_scenes": 7, "neaurc2_rel": 0.15}
    Cfull = {"ace": {"LOSO tau": {"cov95.4": 94.0}},
             "ace_ours_paired": {"LOSO tau": {"cov95.4": 93.2}}}
    assert uf.verdict(rel, C, nll_cmp, Cfull) == "INCONCLUSIVE"
    # ...but genuine equivalence with better ACE coverage is a KILL
    nll_cmp.update({"LOSO tau": (0.01, -0.03, 0.05), "neaurc2_rel": 0.02})
    assert uf.verdict(rel, C, nll_cmp, Cfull) == "KILL"
