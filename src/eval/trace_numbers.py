"""Assert every number quoted in paper/main_rewritten.tex resolves to an artifact.

Each entry pairs a value as it is written in the paper with the file that produced
it, and passes only if the paper's figure is the artifact correctly rounded to the
precision it is written at. Run from the repository root:

    python -m src.eval.trace_numbers

Exit status is non-zero on any mismatch, so a number that drifts from its artifact
fails loudly rather than sitting in the prose. Numbers quoted from a superseded
probe configuration are checked against that configuration by name; see
results/probes vs results/probes_pinned.
"""
import json, re, sys, math, statistics as st
import numpy as np

TEX = open('paper/main_rewritten.tex').read()
PROSE = '\n'.join(l for l in TEX.split('\n') if not l.lstrip().startswith('%'))
ok, bad = [], []

def check(label, claimed, actual, source, tol=None):
    """Claimed must be `actual` correctly rounded to the precision it is written at."""
    if tol is not None:
        hit = abs(claimed - actual) <= tol
    else:
        txt = repr(float(claimed))
        dp = len(txt.split('.')[1].rstrip('0')) if '.' in txt else 0
        hit = round(float(actual), dp) == round(float(claimed), dp)
    (ok if hit else bad).append((label, claimed, actual, source))

PROSE_FLAT = ' '.join(PROSE.split())

def instr(needle):
    """The literal string must appear in the shipped prose, ignoring line wrapping."""
    hit = ' '.join(needle.split()) in PROSE_FLAT
    (ok if hit else bad).append((f"literal {needle!r}", None, None, "paper/main_rewritten.tex"))

K = json.load(open('results/hypotheses/kappa_ci.json'))
H = json.load(open('results/hypotheses/hypothesis_report.json'))
row = lambda c, f: next(r for r in K[c] if r['factor'] == f)


for cell, f, k, lo, hi in [
    ('control_strong','floor_hue', 0.784, 0.762, 0.808),
    ('control_strong','wall_hue',  0.900, 0.890, 0.910),
    ('control_strong','object_hue',0.742, 0.718, 0.771),
    ('color_strong','floor_hue',  -0.020,-0.075, 0.041),
    ('color_strong','wall_hue',   -0.064,-0.096,-0.031),
    ('color_strong','object_hue', -0.244,-0.266,-0.225),
    ('position_strong','pos_x',    0.164, 0.141, 0.187),
    ('position_strong','pos_y',    0.147, 0.126, 0.170),
    ('color_weak','object_hue',   -0.272,-0.323,-0.215)]:
    r = row(cell, f)
    check(f"kappa {cell}:{f}", k, r['kappa'], 'kappa_ci.json')
    check(f"  ci_lo {cell}:{f}", lo, r['kappa_ci95'][0], 'kappa_ci.json')
    check(f"  ci_hi {cell}:{f}", hi, r['kappa_ci95'][1], 'kappa_ci.json')


for f, s, lo, hi in [('floor_hue',0.151,0.096,0.215), ('wall_hue',0.076,0.034,0.118),
                     ('object_hue',0.023,0.009,0.038)]:
    r = row('color_strong', f)
    check(f"S_top color:{f}", s, r['S_top'], 'kappa_ci.json')
    check(f"  S_lo color:{f}", lo, r['S_top_ci95'][0], 'kappa_ci.json')
    check(f"  S_hi color:{f}", hi, r['S_top_ci95'][1], 'kappa_ci.json')


for cell, f, lin, top in [('color_strong','floor_hue',0.9708,0.9965),
                          ('color_strong','wall_hue',0.9738,0.9973),
                          ('color_strong','object_hue',0.8837,0.9757),
                          ('color_strong','scale',0.8536,0.9340),
                          ('color_strong','shape',0.7890,0.8241),
                          ('color_strong','orientation',0.9672,0.9882),
                          ('position_strong','pos_x',0.9567,0.9921),
                          ('position_strong','pos_y',0.9564,0.9918)]:
    r = row(cell, f)
    check(f"floor lin {f}", lin, r['R_random_lin'], 'probes_pinned/*/stacks.npz')
    check(f"floor top {f}", top, r['R_random_top'], 'probes_pinned/*/stacks.npz')
check("max targeted headroom at top rung", 0.0243, 1 - row('color_strong','object_hue')['R_random_top'], 'derived')


for cell, f, lin, top in [('control_strong','floor_hue',0.685,0.935),
                          ('control_strong','wall_hue',0.846,0.985),
                          ('control_strong','object_hue',0.608,0.905),
                          ('color_strong','floor_hue',-0.083,-0.079),
                          ('color_strong','wall_hue',-0.095,-0.140),
                          ('color_strong','object_hue',-0.106,-0.255),
                          ('position_strong','pos_x',0.138,0.307),
                          ('position_strong','pos_y',0.158,0.311)]:
    r = row(cell, f)
    check(f"R_trained lin {cell}:{f}", lin, r['R_trained_lin'], 'stacks.npz')
    check(f"R_trained top {cell}:{f}", top, r['R_trained_top'], 'stacks.npz')


pf = {(r['cell'], r['factor']): r for r in H['hypotheses']['H1']['per_factor']}
for cell, f, g, lo, hi in [('control_strong','floor_hue',0.224,0.209,0.241),
                           ('control_strong','wall_hue',0.115,0.101,0.132),
                           ('control_strong','object_hue',0.205,0.179,0.228),
                           ('control_strong','scale',0.204,0.167,0.237),
                           ('control_strong','orientation',0.168,0.139,0.199),
                           ('position_strong','pos_x',0.134,0.116,0.151),
                           ('position_strong','pos_y',0.117,0.101,0.135),
                           ('color_strong','floor_hue',-0.021,-0.079,0.044),
                           ('color_strong','wall_hue',-0.068,-0.102,-0.034),
                           ('color_strong','object_hue',-0.241,-0.258,-0.227)]:
    r = pf[(cell,f)]
    check(f"dG {cell}:{f}", g, r['delta_g'], 'hypothesis_report.json')
    check(f"  dG_lo {cell}:{f}", lo, r['delta_g_ci'][0], 'hypothesis_report.json')
    check(f"  dG_hi {cell}:{f}", hi, r['delta_g_ci'][1], 'hypothesis_report.json')
check("A8(e) bias shapes3d", 0.0099, H['hypotheses']['H1']['a8e_upper_bias_by_cell']['color_strong']['delta_g_upper_bias'],'calibration_shapes3d.json',5e-5)
check("A8(e) bias dsprites", 0.0113, H['hypotheses']['H1']['a8e_upper_bias_by_cell']['position_strong']['delta_g_upper_bias'],'calibration_dsprites.json',5e-5)


check("H1 confirming readouts", 7, len(H['hypotheses']['H1']['confirmed_cells_factors']), 'hypothesis_report.json', 0)
check("H2 significant pairs", 12, len(H['hypotheses']['H2']['significant_pairs']), 'hypothesis_report.json', 0)
check("H2 pairs tested", 23, len(H['hypotheses']['H2']['pairs']), 'hypothesis_report.json', 0)
check("H3 confirmed (two-sided)", 0, len(H['hypotheses']['H3']['confirmed_cells_factors']), 'hypothesis_report.json', 0)
check("H3 invariant-but-saturated", 10, len(H['hypotheses']['H3']['invariant_but_null_saturated']), 'hypothesis_report.json', 0)
check("H4 significant sign cells", 14, len(H['hypotheses']['H4']['sign_component']['significant']), 'hypothesis_report.json', 0)
fc = H['headline_flip_count']
check("flip two-sided", 1, fc['two_sided']['n_flips'], 'hypothesis_report.json', 0)
check("flip saturation-excluded", 1, fc['two_sided_excl_null_saturated']['n_flips'], 'hypothesis_report.json', 0)
check("flip frozen one-sided", 0, fc['primary']['n_flips'], 'hypothesis_report.json', 0)
check("flip fixed 0.05", 0, fc['fixed_0.05']['n_flips'], 'hypothesis_report.json', 0)
check("flip bootstrap mean", 0.196, fc['uncertainty']['two_sided']['n_flips_mean'], 'hypothesis_report.json')
check("kappa undefined, confirmatory", 6,
      sum(1 for c in ('color_strong','control_strong','position_strong') for r in K[c] if not r['kappa_defined']),'kappa_ci.json',0)
check("confirmatory readouts", 17, sum(len(K[c]) for c in ('color_strong','control_strong','position_strong')),'kappa_ci.json',0)


t = {(x['cell'],x['factor'],x['rung']): x for x in H['hypotheses']['H4']['sign_component']['tests']}
check("H4 floor_hue mlp_small", 0.149, t[('color_strong','floor_hue','mlp_small')]['mean_diff'], 'hypothesis_report.json')
check("H4 floor_hue linear", -0.109, t[('color_strong','floor_hue','linear')]['mean_diff'], 'hypothesis_report.json')


def gate(path):
    m = json.load(open(path))['quality_gate']
    a = [s['shape_recoverability'] for s in m['per_seed']]
    return m['n_passed'], m['n_encoders'], st.mean(a), min(a), max(a), (st.stdev(a) if len(a)>1 else 0.0), m['per_seed'][0]['floor']
for name, path, mean, floor, npass in [
    ('color_strong','results/probes_pinned/color_strong/meta.json',1.0000,0.7890,12),
    ('position_strong','results/probes_pinned/position_strong/meta.json',0.9904,0.4690,12),
    ('color_weak','results/probes_pinned/color_weak/meta.json',0.9680,0.7890,12),
    ('control_strong','results/probes_pinned/control_strong/meta.json',0.7370,0.7890,0)]:
    p_, n_, mu, mn, mx, sd, fl = gate(path)
    check(f"anchor {name}", mean, mu, 'meta.json quality_gate')
    check(f"floor {name}", floor, fl, 'meta.json quality_gate')
    check(f"pass {name}", npass, p_, 'meta.json quality_gate', 0)
p_,n_,mu,mn,mx,sd,fl = gate('results/probes_excluded/scale_strong/meta.json')
check("scale anchor mean", 0.4833, mu, 'probes_excluded'); check("scale anchor min", 0.4640, mn, 'probes_excluded')
check("scale anchor max", 0.5066, mx, 'probes_excluded'); check("scale anchor sd", 0.0132, sd, 'probes_excluded')
check("pilot minus scale max", 0.4934, 1.0-mx, 'derived'); check("pilot sd multiples", 37, round((1.0-mx)/sd), 'derived', 0.6)
check("pilot range multiples", 11.6, (1.0-mx)/(mx-mn), 'derived', 0.05)
p_,n_,mu,mn,mx,sd,fl = gate('results/probes_excluded/orientation_strong/meta.json')
check("orientation anchor", 0.677, mu, 'probes_excluded')
check("pre-pin floor", 0.829, fl, 'probes_excluded')

import numpy as _np
z = _np.load('results/probes_excluded/scale_recipe_strong/stacks.npz')
mp = json.load(open('results/probes_excluded/scale_recipe_strong/meta.json'))
fac = [f['name'] for f in mp['factors']]
check("pilot object_hue lin", 0.0217, float(z['trained'][:, fac.index('object_hue'), 0].mean()), 'probes_excluded')
check("pilot object_hue floor", 0.8871, float(z['random'][:, fac.index('object_hue'), 0].mean()), 'probes_excluded')
zo = _np.load('results/probes_excluded/orientation_strong/stacks.npz')
fo = [f['name'] for f in json.load(open('results/probes_excluded/orientation_strong/meta.json'))['factors']]
check("orientation anchor top rung", 0.928, float(zo['trained'][:, fo.index('shape'), -1].mean()), 'probes_excluded')
zs = _np.load('results/probes_excluded/scale_strong/stacks.npz')
fs = [f['name'] for f in json.load(open('results/probes_excluded/scale_strong/meta.json'))['factors']]
check("scale anchor top rung", 0.609, float(zs['trained'][:, fs.index('shape'), -1].mean()), 'probes_excluded')
check("pre-pin floor top rung", 0.986, float(zs['random'][:, fs.index('shape'), -1].mean()), 'probes_excluded')
zc = _np.load('results/probes/control_strong/stacks.npz')
fc2 = [f['name'] for f in json.load(open('results/probes/control_strong/meta.json'))['factors']]
check("pre-pin control anchor", 0.810, float(zc['trained'][:, fc2.index('shape'), 0].mean()), 'results/probes')


def disp(fn, role='trained'):
    d = json.load(open(f'results/displacement/{fn}.json'))
    out = {}
    for f in next(iter(d['roles'][role].values()))['factors']:
        vals = [r['factors'][f]['m_total'] for r in d['roles'][role].values()]
        verd = [r['factors'][f]['verdict'] for r in d['roles'][role].values()]
        out[f] = (st.mean(vals), verd)
    return out, d
cs, meta = disp('color_strong'); ct, _ = disp('control_strong'); ps, _ = disp('position_strong')
nul, _ = disp('color_strong', 'random')
check("m_star", 0.2888319636495567, meta['m_star'], 'threshold_calibration.json')
check("band lo", 0.0799, meta['unadjudicated_band'][0], 'threshold_calibration.json')
check("band hi", 1.0445, meta['unadjudicated_band'][1], 'threshold_calibration.json')
for f, v in [('floor_hue',0.0597),('wall_hue',0.0766),('object_hue',0.0617)]:
    check(f"color m_total {f}", v, cs[f][0], 'displacement/color_strong.json')
for f, v in [('floor_hue',17.63),('wall_hue',23.61),('object_hue',21.10)]:
    check(f"control m_total {f}", v, ct[f][0], 'displacement/control_strong.json')
for f, v in [('floor_hue',4.89),('wall_hue',6.01),('object_hue',4.40)]:
    check(f"null m_total {f}", v, nul[f][0], 'displacement/reference_shapes3d.json')
check("pos_x m_total", 2.411, ps['pos_x'][0], 'displacement/position_strong.json')
check("pos_y m_total", 2.385, ps['pos_y'][0], 'displacement/position_strong.json')
for f, r in [('floor_hue',295),('wall_hue',308),('object_hue',342)]:
    check(f"ratio {f}", r, ct[f][0]/cs[f][0], 'derived', 0.6)
check("log10 ratio min", 2.47, min(math.log10(ct[f][0]/cs[f][0]) for f in ('floor_hue','wall_hue','object_hue')),'derived',5e-3)
check("log10 ratio max", 2.53, max(math.log10(ct[f][0]/cs[f][0]) for f in ('floor_hue','wall_hue','object_hue')),'derived',5e-3)
check("wall_hue destroyed seeds", 7, cs['wall_hue'][1].count('eps_destroyed'), 'displacement/color_strong.json', 0)
check("wall_hue unadjudicated seeds", 5, cs['wall_hue'][1].count('unadjudicated'), 'displacement/color_strong.json', 0)
v95, _ = disp('color_strong_vf0.95'); v999, _ = disp('color_strong_vf0.999')
for f in ('floor_hue','wall_hue','object_hue'):
    check(f"vf0.95 destroyed 12/12 {f}", 12, v95[f][1].count('eps_destroyed'), 'displacement/color_strong_vf0.95.json', 0)
    check(f"vf0.999 unadjudicated 12/12 {f}", 12, v999[f][1].count('unadjudicated'), 'displacement/color_strong_vf0.999.json', 0)

d = json.load(open('results/displacement/color_strong.json'))
re_ = [r['factors'][f]['r_eff'] for r in d['roles']['trained'].values() for f in ('floor_hue','wall_hue','object_hue')]
sf  = [r['factors'][f]['shared_fraction'] for r in d['roles']['trained'].values() for f in ('floor_hue','wall_hue','object_hue')]
mre = {f: st.mean([r['factors'][f]['r_eff'] for r in d['roles']['trained'].values()]) for f in ('floor_hue','wall_hue','object_hue')}
msf = {f: st.mean([r['factors'][f]['shared_fraction'] for r in d['roles']['trained'].values()]) for f in ('floor_hue','wall_hue','object_hue')}
check("r_eff min (mean per factor)", 45.6, min(mre.values()), 'displacement', 5e-2)
check("r_eff max (mean per factor)", 47.0, max(mre.values()), 'displacement', 5e-2)
check("shared frac min", 0.097, min(msf.values()), 'displacement')
check("shared frac max", 0.218, max(msf.values()), 'displacement')
wc = [r['factors'][f]['whitening_check'] for fn in ('color_strong','control_strong','position_strong','reference_shapes3d','reference_dsprites')
      for r in json.load(open(f'results/displacement/{fn}.json'))['roles'].get('trained', json.load(open(f'results/displacement/{fn}.json'))['roles']['random']).values()
      for f in r['factors']]
allwc = []
for fn in ('color_strong','control_strong','position_strong','reference_shapes3d','reference_dsprites'):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    for role in dd['roles']:
        for r in dd['roles'][role].values():
            allwc += [x['whitening_check'] for x in r['factors'].values()]
check("whitening check min", 0.84, min(allwc), 'displacement')
check("whitening check max", 1.01, max(allwc), 'displacement')
mono = []
for fn in ('color_strong','control_strong','position_strong','reference_shapes3d','reference_dsprites'):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    for role in dd['roles']:
        for r in dd['roles'][role].values():
            mono += [x['monotonicity']['monotone_fraction'] for x in r['factors'].values()]
_mono_mean = {}
for fn in ('color_strong','control_strong','position_strong','reference_shapes3d','reference_dsprites'):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    for role in ('trained', 'random'):
        for r in dd['roles'].get(role, {}).values():
            for f, x in r['factors'].items():
                _mono_mean.setdefault((fn, role, f), []).append(
                    x['monotonicity']['monotone_fraction'])
check("max per-readout mean monotone fraction, encoder roles", 0.294,
      max(st.mean(v) for v in _mono_mean.values()), 'displacement')

for vf, fn, exp in [(0.95,'color_strong_vf0.95',12), (0.99,'color_strong',12), (0.999,'color_strong_vf0.999',11)]:
    dd = json.load(open(f'results/displacement/{fn}.json'))
    lo_, hi_ = dd['unadjudicated_band']
    cnt = min(sum(1 for r in dd['roles']['trained'].values()
                  if r['factors'][f]['m_shared'] < lo_) for f in ('floor_hue','wall_hue','object_hue'))
    check(f"m_shared destroyed at vf {vf} (min over hues)", exp, cnt, 'displacement', 0)

cal = json.load(open('results/displacement/threshold_calibration.json'))
dev = max(abs(r['m_shared']/r['m_total'] - 1) for r in cal['rows'] if r['m_total'] > 0)
check("calibration m_shared/m_total deviation", 0.0, dev, 'threshold_calibration.json', 1e-4)
check("calibration contexts", 256, cal['n_contexts'], 'threshold_calibration.json', 0)


mm = json.load(open('results/probes_pinned/color_strong/meta.json'))
for i, v in enumerate([513, 32897, 131585, 164353]):
    check(f"rung params {i}", v, mm['rung_params_h']['floor_hue'][i], 'meta.json', 0)
check("projector dim", 128, mm['projector_dim'], 'meta.json', 0)
check("probe_train pin", 2000, mm['probe_train_size'], 'meta.json', 0)
check("probe_steps pin", 1000, mm['probe_steps'], 'meta.json', 0)
import yaml
b = yaml.safe_load(open('configs/base.yaml'))
for k, v in [('encoder_train',400000),('probe_train',40000),('probe_val',10000),('probe_test',30000)]:
    check(f"split {k}", v, b['split']['sizes'][k], 'configs/base.yaml', 0)
check("shapes3d n_total", 480000, b['data']['n_total'], 'configs/base.yaml', 0)
check("NT-Xent temperature", 0.5, b['simclr']['temperature'], 'configs/base.yaml', 0)
check("projector out_dim", 128, b['projector']['out_dim'], 'configs/base.yaml', 0)
cfg = yaml.safe_load(open('configs/run/color_strong.yaml'))
check("encoder subset", 150000, cfg['data']['subset'], 'configs/run/*.yaml', 0)
check("encoder epochs", 50, cfg['run']['epochs'], 'configs/run/*.yaml', 0)
check("encoder batch", 512, cfg['run']['batch_size'], 'configs/run/*.yaml', 0)

aug = open('src/encoders/augmentations.py').read()
for lit in ["ColorJitter(0.8, 0.8, 0.8, 0.5)", "ColorJitter(0.2, 0.2, 0.2, 0.1)",
            "ColorJitter(0.4, 0.4, 0.4, 0.1)", "RandomGrayscale(p=0.2)",
            "scale_range = (0.9, 1.1) if weak else (0.75, 1.25)"]:
    (ok if lit in aug else bad).append((f"aug literal {lit}", None, None, 'src/encoders/augmentations.py'))
lad = open('src/probes/ladder.py').read()
for lit in ['Rung("mlp_small", (64,))', 'Rung("mlp_large", (256,))', 'Rung("mlp_deep", (256, 128))',
            'WEIGHT_DECAY_GRID: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)',
            'DEFAULT_BATCH = 4096', 'DEFAULT_LR = 1e-3']:
    (ok if lit in lad else bad).append((f"ladder literal {lit}", None, None, 'src/probes/ladder.py'))
gq = open('src/eval/quality_gate.py').read()
(ok if '0.9' in gq and '0.02' in gq else bad).append(("gate thresholds 0.90 / 0.02", None, None, 'src/eval/quality_gate.py'))
pre = open('preregistration/prereg.md').read()
for lit in ['17.2% at 0.95', '13.4% at 0.99', '6.7% at 0.999', '0.25% at full rank']:
    (ok if lit in pre else bad).append((f"prereg literal {lit}", None, None, 'preregistration/prereg.md'))


_hues = ('floor_hue', 'wall_hue', 'object_hue')
_kc = [row('control_strong', f)['kappa'] for f in _hues]
_kk = [row('color_strong', f)['kappa'] for f in _hues]
check("abstract control kappa low",  0.74, min(_kc), 'kappa_ci.json')
check("abstract control kappa high", 0.90, max(_kc), 'kappa_ci.json')
check("abstract color kappa high", -0.02, max(_kk), 'kappa_ci.json')
check("abstract color kappa low",  -0.24, min(_kk), 'kappa_ci.json')
_targeted = [('color_strong', f) for f in _hues] + [('position_strong','pos_x'), ('position_strong','pos_y')]
_tf = [row(c, f)['R_random_top'] for c, f in _targeted]
check("abstract targeted floor low",  0.976, min(_tf), 'probes_pinned/*/stacks.npz')
check("abstract targeted floor high", 0.997, max(_tf), 'probes_pinned/*/stacks.npz')


check("band factor", 13.1, meta['unadjudicated_band'][1] / meta['unadjudicated_band'][0],
      'threshold_calibration.json')





COMP = json.load(open('results/displacement/scale_companion.json'))
_cc = COMP['cells']['color_strong']['factors']
_ct = COMP['cells']['control_strong']['factors']
_rs = COMP['rank_scaling']
_nr = COMP['null_ratio_stability']

def _st(cell):
    z = np.load(f'results/probes_pinned/{cell}/stacks.npz')
    m = json.load(open(f'results/probes_pinned/{cell}/meta.json'))
    return z, [f['name'] for f in m['factors']], m

_zc, _fc, _mc = _st('control_strong')
_zk, _fk, _mk = _st('color_strong')


_gate = _mc['quality_gate']
check("control gate passes", 0, _gate['n_passed'], 'probes_pinned/control_strong/meta.json')
check("control gate n", 12, _gate['n_encoders'], 'probes_pinned/control_strong/meta.json')
check("control anchor", 0.7370, st.mean(p['shape_recoverability'] for p in _gate['per_seed']),
      'probes_pinned/control_strong/meta.json')
check("control anchor floor", 0.7890, st.mean(p['floor'] for p in _gate['per_seed']),
      'probes_pinned/control_strong/meta.json')
_shape_i = _fc.index('shape')
_dshape = (_zc['random'] - _zc['trained']).mean(0)[_shape_i]
for r, claimed in zip(range(4), (0.052, 0.055, 0.054, 0.058)):
    check(f"control shape deficit rung{r}", claimed, _dshape[r], 'probes_pinned/control_strong')


_D = (_zk['random'] - _zk['trained']).mean(0)
_hidx = [_fk.index(f) for f in _hues]
_Dh = [_D[i][r] for i in _hidx for r in range(4)]
check("color hue deficit low", 0.99, min(_Dh), 'probes_pinned/color_strong')
check("color hue deficit high", 1.23, max(_Dh), 'probes_pinned/color_strong')

_unt = [_zk['trained'].mean(0)[_fk.index(f)][r] for f in ('scale','shape','orientation')
        for r in range(4)]
check("color untargeted low", 0.999, min(_unt), 'probes_pinned/color_strong')
check("color untargeted high", 1.000, max(_unt), 'probes_pinned/color_strong')


for f, k0, k1, k2 in [('floor_hue', 56, 106, 210)]:
    for i, claimed in enumerate((k0, k1, k2)):
        check(f"retained rank {f}[{i}]", claimed, _rs[f'trained/{f}']['rank'][i], 'scale_companion.json')
    for i, claimed in enumerate((0.0212, 0.0597, 0.2571)):
        check(f"m_total {f}[{i}]", claimed, _rs[f'trained/{f}']['m_total'][i], 'scale_companion.json')
for f, a in (('floor_hue', 1.90), ('wall_hue', 1.89), ('object_hue', 1.84)):
    check(f"rank exponent {f}", a, _rs[f'trained/{f}']['exponent_alpha'], 'scale_companion.json')
_ua = [_rs[f'trained/{f}']['exponent_alpha'] for f in ('scale', 'shape', 'orientation')]
check("untargeted exponent low", 0.54, min(_ua), 'scale_companion.json')
check("untargeted exponent high", 0.74, max(_ua), 'scale_companion.json')
check("rank change over sweep", 3.7,
      _rs['trained/floor_hue']['rank'][2] / _rs['trained/floor_hue']['rank'][0],
      'scale_companion.json')
check("m_total spread over sweep", 12,
      max(_rs['trained/floor_hue']['m_total']) / min(_rs['trained/floor_hue']['m_total']),
      'scale_companion.json', tol=0.2)


for vf, vals in (('0.95', (0.0113, 0.0108, 0.0194)), ('0.99', (0.0122, 0.0127, 0.0140)),
                 ('0.999', (0.0218, 0.0229, 0.0210))):
    for f, claimed in zip(_hues, vals):
        check(f"null ratio {f}@{vf}", claimed, _nr[f][vf], 'scale_companion.json')
_sp = [_nr[f]['spread_max_over_min'] for f in _hues]
check("null ratio spread low", 1.5, min(_sp), 'scale_companion.json')
check("null ratio spread high", 2.1, max(_sp), 'scale_companion.json')
_allr = [_nr[f][vf] for f in _hues for vf in ('0.95', '0.99', '0.999')]
check("null ratio pct low", 1.1, 100 * min(_allr), 'scale_companion.json')
check("null ratio pct high", 2.3, 100 * max(_allr), 'scale_companion.json')



_nc = COMP['null_ratio_ci']
for f, r, lo, hi in (('floor_hue', 0.0122, 0.0108, 0.0139),
                     ('wall_hue', 0.0127, 0.0114, 0.0144),
                     ('object_hue', 0.0140, 0.0123, 0.0164),
                     ('shape', 2.65, 2.28, 3.13),
                     ('scale', 3.72, 3.11, 4.61),
                     ('orientation', 4.05, 3.42, 4.95)):
    d = _nc[f]['0.99']
    check(f"null ratio ci {f}", r, d['ratio'], 'scale_companion.json')
    check(f"  lo {f}", lo, d['ci95'][0], 'scale_companion.json')
    check(f"  hi {f}", hi, d['ci95'][1], 'scale_companion.json')
    check(f"  n seeds {f}", 12, d['n_seeds'], 'scale_companion.json')
check("hue ratio intervals stay under", 0.017,
      max(_nc[f]['0.99']['ci95'][1] for f in _hues), 'scale_companion.json', tol=0.0031)
check("untargeted ratio low", 2.65,
      min(_nc[f]['0.99']['ratio'] for f in ('shape', 'scale', 'orientation')),
      'scale_companion.json')
check("untargeted ratio high", 4.05,
      max(_nc[f]['0.99']['ratio'] for f in ('shape', 'scale', 'orientation')),
      'scale_companion.json')

for f, vals in (('floor_hue', (0.0113, 0.0122, 0.0218)),
                ('wall_hue', (0.0108, 0.0127, 0.0229)),
                ('object_hue', (0.0194, 0.0140, 0.0210))):
    for vf, claimed in zip(('0.95', '0.99', '0.999'), vals):
        check(f"sweep triple {f}@{vf}", claimed, _nr[f][vf], 'scale_companion.json')
instr("each across variance fractions $0.95$, $0.99$ and\n$0.999$")
instr("this within-arm contrast does not\ndepend on the cross-arm rank comparison")


check("control k trained", 435.5, _ct['floor_hue']['trained']['rank'], 'scale_companion.json')
check("control k random", 56.8, _ct['floor_hue']['random']['rank'], 'scale_companion.json')
check("control sqrt2k trained", 29.5, _ct['floor_hue']['trained']['attainable_scale_sqrt_2k'],
      'scale_companion.json')
check("control sqrt2k random", 10.7, _ct['floor_hue']['random']['attainable_scale_sqrt_2k'],
      'scale_companion.json')
check("control m/mctx trained", 0.655, _ct['floor_hue']['trained']['m_over_ctx'], 'scale_companion.json')
check("control m/mctx random", 0.518, _ct['floor_hue']['random']['m_over_ctx'], 'scale_companion.json')
_norm = [_ct[f]['trained']['m_over_ctx'] / _cc[f]['trained']['m_over_ctx'] for f in _hues]
check("normalized separation low", 154, min(_norm), 'scale_companion.json', tol=0.5)
check("normalized separation high", 182, max(_norm), 'scale_companion.json', tol=0.5)
check("normalized separation orders", 2.2, math.log10(min(_norm)), 'scale_companion.json', tol=0.05)
for f, claimed in zip(_hues, (0.75, 0.959, 0.77)):
    check(f"band margin {f}", claimed, _cc[f]['margin_to_band_lower'], 'scale_companion.json')
check("wall hue margin pct", 4.1, 100 * (1 - _cc['wall_hue']['margin_to_band_lower']),
      'scale_companion.json')
check("sweep coverage control", 3, len(COMP['sweep_coverage']['control_strong']),
      'scale_companion.json')
check("sweep coverage position", 3, len(COMP['sweep_coverage']['position_strong']),
      'scale_companion.json')


import glob
def _loss(cell):
    v = [json.load(open(p))['final_loss'] for p in
         glob.glob(f'results/encoders/{cell}/**/*.json', recursive=True)
         if 'final_loss' in json.load(open(p))]
    return st.mean(v)
check("control NT-Xent", 4.949, _loss('control'), 'results/encoders/control')
check("color NT-Xent", 4.953, _loss('color'), 'results/encoders/color')
check("control shape level", 0.737, _zc['trained'].mean(0)[_fc.index('shape')][0],
      'probes_pinned/control_strong')
check("control scale level", 0.321, _zc['trained'].mean(0)[_fc.index('scale')][0],
      'probes_pinned/control_strong')
check("color shape level", 1.000, _zk['trained'].mean(0)[_fk.index('shape')][0],
      'probes_pinned/color_strong')
check("color scale level", 0.999, _zk['trained'].mean(0)[_fk.index('scale')][0],
      'probes_pinned/color_strong')


for cell, z, f, vals in (('control_strong', _zc, _fc, (0.817, 0.982, 0.847)),
                         ('color_strong',  _zk, _fk, (-0.230, -0.217, -0.278))):
    for name, claimed in zip(_hues, vals):
        check(f"R_perm top {cell}:{name}", claimed, z['perm'].mean(0)[f.index(name)][3],
              f'probes_pinned/{cell}/stacks.npz')

for name, sv, lo, hi in (('floor_hue', 0.118, 0.086, 0.150),
                         ('object_hue', 0.058, 0.041, 0.074),
                         ('wall_hue', 0.003, 0.002, 0.004)):
    i = _fc.index(name)
    _S = _zc['trained'][:, i, 3] - _zc['perm'][:, i, 3]
    _rng = np.random.default_rng(20260906)
    _b = _S[_rng.integers(0, _S.shape[0], (4000, _S.shape[0]))].mean(1)
    check(f"S_top control:{name}", sv, _S.mean(), 'probes_pinned/control_strong')
    check(f"  S_lo control:{name}", lo, np.percentile(_b, 2.5), 'probes_pinned/control_strong',
          tol=0.004)
    check(f"  S_hi control:{name}", hi, np.percentile(_b, 97.5), 'probes_pinned/control_strong',
          tol=0.004)
check("control wall true level", 0.985, _zc['trained'].mean(0)[_fc.index('wall_hue')][3],
      'probes_pinned/control_strong')
check("untrained floor low targeted", 0.976, min(_tf), 'probes_pinned/*/stacks.npz')
_rlo = min(_zk['random'].mean(0)[_fk.index(f)][3] for f in _hues)
check("color arm random top low", 0.976, _rlo, 'probes_pinned/color_strong')

_cperm = [_zc['perm'].mean(0)[_fc.index(f)][3] for f in _hues]
check("abstract control task low",  0.82, min(_cperm), 'probes_pinned/control_strong')
check("abstract control task high", 0.98, max(_cperm), 'probes_pinned/control_strong')
_kperm = [_zk['perm'].mean(0)[_fk.index(f)][r] for f in _hues for r in range(4)]
(ok if max(_kperm) < 0 else bad).append(
    ("color control task unrecoverable at every rung", "< 0", max(_kperm),
     'probes_pinned/color_strong/stacks.npz'))


_z40 = np.load('results/probes/color_strong/stacks.npz')
_f40 = [f['name'] for f in json.load(open('results/probes/color_strong/meta.json'))['factors']]
check("pre-A10 scale linear (40k root)", -1.653,
      _z40['trained'].mean(0)[_f40.index('scale')][0], 'results/probes/color_strong')
check("pinned scale linear", 0.999, _zk['trained'].mean(0)[_fk.index('scale')][0],
      'probes_pinned/color_strong')
instr("Amendment A10 invalidated the earlier continuous readouts through target standardization")





NULL = json.load(open('results/hypotheses/kappa_null.json'))
_nrow = lambda c, f: next(r for r in NULL['cells'][c] if r['factor'] == f)

check("kappa null probe train", 2000, NULL['probe_train_size'], 'kappa_null.json', 0)
check("kappa null probe steps", 1000, NULL['probe_steps'], 'kappa_null.json', 0)
check("kappa null bootstrap draws", 4000, NULL['n_boot'], 'kappa_null.json', 0)
check("kappa null seeds", 12, len(NULL['seeds']), 'kappa_null.json', 0)
check("kappa null device dev", 3.31e-5, NULL['device_check']['linear_rung_max_dev'],
      'kappa_null.json', tol=5e-8)
(ok if NULL['device_check']['linear_rung_max_dev'] < 5e-3 else bad).append(
    ("kappa null device dev under 5e-3 threshold", "< 5e-3",
     NULL['device_check']['linear_rung_max_dev'], 'kappa_null.json'))


_rep = [abs(_nrow(c, f)['kappa'] - row(c, f)['kappa'])
        for c in ('color_strong', 'control_strong')
        for f in [r['factor'] for r in NULL['cells'][c] if r['kappa_defined']]]
check("kappa null defined rows reproduced", 9, len(_rep), 'kappa_null.json', 0)
(ok if max(_rep) <= 0.004 else bad).append(
    ("kappa null reproduces shipped kappa to 0.004", "<= 0.004", max(_rep), 'kappa_null.json'))


for cell, f, kp, lo, hi in [
        ('control_strong', 'floor_hue',   0.641,  0.598,  0.680),
        ('control_strong', 'wall_hue',    0.892,  0.869,  0.909),
        ('control_strong', 'object_hue',  0.573,  0.438,  0.669),
        ('color_strong',   'floor_hue',  -0.198, -0.257, -0.129),
        ('color_strong',   'wall_hue',   -0.145, -0.184, -0.107),
        ('color_strong',   'object_hue', -0.365, -0.432, -0.302),
        ('control_strong', 'scale',      -0.205, -0.347, -0.055),
        ('control_strong', 'orientation',-0.405, -0.493, -0.281)]:
    r = _nrow(cell, f)
    check(f"kappa_perm {cell}:{f}", kp, r['kappa_perm'], 'kappa_null.json')
    check(f"  perm ci_lo {cell}:{f}", lo, r['kappa_perm_ci95'][0], 'kappa_null.json')
    check(f"  perm ci_hi {cell}:{f}", hi, r['kappa_perm_ci95'][1], 'kappa_null.json')


for cell, f, k in [('control_strong', 'floor_hue', 0.785), ('control_strong', 'wall_hue', 0.900),
                   ('control_strong', 'object_hue', 0.739), ('color_strong', 'floor_hue', -0.020),
                   ('color_strong', 'wall_hue', -0.064), ('color_strong', 'object_hue', -0.244),
                   ('control_strong', 'scale', 0.383), ('control_strong', 'orientation', 0.779)]:
    check(f"kappa (null run) {cell}:{f}", k, _nrow(cell, f)['kappa'], 'kappa_null.json')
for cell, f, lo, hi in [('control_strong', 'scale', 0.306, 0.459),
                        ('control_strong', 'orientation', 0.694, 0.885)]:
    check(f"kappa ci_lo (null run) {cell}:{f}", lo, _nrow(cell, f)['kappa_ci95'][0], 'kappa_null.json')
    check(f"kappa ci_hi (null run) {cell}:{f}", hi, _nrow(cell, f)['kappa_ci95'][1], 'kappa_null.json')

_wn = _nrow('control_strong', 'wall_hue')
check("flagship null reproduction pct", 99.1, 100 * _wn['kappa_perm'] / _wn['kappa'], 'kappa_null.json')


_drop = {f: a[0] - b[0] for f, a, b in
         zip(NULL['factors'], NULL['random_real_mean'], NULL['random_perm_mean'])}
for f, v in [('wall_hue', 0.024), ('floor_hue', 0.057), ('object_hue', 0.110),
             ('scale', 0.665), ('orientation', 0.529), ('shape', -0.0002)]:
    check(f"control-task drop {f}", v, _drop[f], 'kappa_null.json')


def _sweep(arm, vf):
    fn = f"{arm}_strong" + ("" if vf == 0.99 else f"_vf{vf}")
    dd = json.load(open(f'results/displacement/{fn}.json'))
    rows = [(f, x['m_total'], x['verdict'])
            for r in dd['roles']['trained'].values() for f, x in r['factors'].items()]
    return dd, rows

_ctlv = {vf: _sweep('control', vf) for vf in (0.95, 0.99, 0.999)}
_posv = {vf: _sweep('position', vf) for vf in (0.95, 0.99, 0.999)}
_EDGE = _ctlv[0.99][0]['unadjudicated_band'][1]

check("control sweep verdicts total", 216, sum(len(v[1]) for v in _ctlv.values()), 'displacement', 0)
check("control sweep all moves", 216,
      sum(1 for v in _ctlv.values() for _, _, w in v[1] if w == 'moves'), 'displacement', 0)
for vf, margin in [(0.95, 2.34), (0.99, 3.25), (0.999, 3.66)]:
    check(f"control tightest band margin vf{vf}", margin,
          min(m for _, m, _ in _ctlv[vf][1]) / _EDGE, 'displacement')

check("position sweep moves vf0.95", 59,
      sum(1 for _, _, w in _posv[0.95][1] if w == 'moves'), 'displacement', 0)
check("position sweep total vf0.95", 60, len(_posv[0.95][1]), 'displacement', 0)
for vf in (0.99, 0.999):
    check(f"position sweep all moves vf{vf}", 60,
          sum(1 for _, _, w in _posv[vf][1] if w == 'moves'), 'displacement', 0)
for vf, px, py in [(0.99, 1.89, 1.87), (0.999, 4.64, 4.64)]:
    for f, claimed in (('pos_x', px), ('pos_y', py)):
        check(f"position {f} band margin vf{vf}", claimed,
              min(m for g, m, _ in _posv[vf][1] if g == f) / _EDGE, 'displacement')

_s4 = _posv[0.95][0]['roles']['trained']['position_strong_seed4']['factors']
check("band upper edge", 1.0445, _EDGE, 'displacement')
check("seed4 pos_x m_total", 1.0407, _s4['pos_x']['m_total'], 'displacement')
check("seed4 pos_y m_total", 1.0445, _s4['pos_y']['m_total'], 'displacement')
check("seed4 pos_y over edge", 3.3e-5, _s4['pos_y']['m_total'] - _EDGE, 'displacement', tol=5e-7)
for f, want in (('pos_x', 'unadjudicated'), ('pos_y', 'moves')):
    (ok if _s4[f]['verdict'] == want else bad).append(
        (f"seed4 {f} verdict", want, _s4[f]['verdict'], 'displacement'))
    (ok if _s4[f]['ci']['m'][0] < _EDGE < _s4[f]['ci']['m'][1] else bad).append(
        (f"seed4 {f} ci brackets edge", True, _s4[f]['ci']['m'], 'displacement'))


def _mean_by_factor(fn, role):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    o = {}
    for r in dd['roles'][role].values():
        for f, x in r['factors'].items():
            o.setdefault(f, []).append(x['m_total'])
    return {f: st.mean(v) for f, v in o.items()}

def _spreads(arm, factors):
    fns = [f"{arm}_strong_vf0.95", f"{arm}_strong", f"{arm}_strong_vf0.999"]
    tr = [_mean_by_factor(fn, 'trained') for fn in fns]
    rn = [_mean_by_factor(fn, 'random') for fn in fns]
    ms = [max(x[f] for x in tr) / min(x[f] for x in tr) for f in factors]
    rt = [[t[f] / r[f] for t, r in zip(tr, rn)] for f in factors]
    rs = [max(v) / min(v) for v in rt]
    return ms, rs, dict(zip(factors, rt))

_HU = ['floor_hue', 'wall_hue', 'object_hue']
_CF = _HU + ['scale', 'shape', 'orientation']
_PF = ['pos_x', 'pos_y', 'shape', 'scale', 'orientation']
_cms, _crs, _crt = _spreads('color', _HU)
_tms, _trs, _trt = _spreads('control', _CF)
_pms, _prs, _prt = _spreads('position', _PF)
for lbl, lo, hi, vals in [("color m_total spread", 11.2, 12.1, _cms),
                          ("color ratio spread", 1.5, 2.1, _crs),
                          ("control m_total spread", 1.4, 1.6, _tms),
                          ("control ratio spread", 4.0, 19.4, _trs),
                          ("position m_total spread", 2.1, 4.4, _pms),
                          ("position ratio spread", 14.6, 105.3, _prs)]:
    check(f"{lbl} low", lo, min(vals), 'displacement')
    check(f"{lbl} high", hi, max(vals), 'displacement')
for f, vals in [('scale', (10.35, 1.64, 0.53)), ('orientation', (11.40, 1.95, 0.63))]:
    for i, v in enumerate(vals):
        check(f"control ratio {f} vf{i}", v, _trt[f][i], 'displacement')
for f, vals in [('pos_x', (11.00, 0.47, 0.28)), ('pos_y', (11.19, 0.46, 0.28))]:
    for i, v in enumerate(vals):
        check(f"position ratio {f} vf{i}", v, _prt[f][i], 'displacement')


_sh = []
for fn in ('control_strong_vf0.95', 'control_strong', 'control_strong_vf0.999'):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    g = lambda role: st.mean([r['factors']['floor_hue']['m_shared'] for r in dd['roles'][role].values()])
    _sh.append(g('trained') / g('random'))
for i, v in enumerate((1.62, 0.88, 0.78)):
    check(f"control m_shared ratio floor_hue vf{i}", v, _sh[i], 'displacement')


def _rank(fn, role):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    return st.mean([r['whitening']['rank'] for r in dd['roles'][role].values()])
def _cond(fn, role):
    dd = json.load(open(f'results/displacement/{fn}.json'))
    return st.mean([r['whitening']['condition_number'] for r in dd['roles'][role].values()])

_pfn = ('position_strong_vf0.95', 'position_strong', 'position_strong_vf0.999')
_cfn = ('control_strong_vf0.95', 'control_strong', 'control_strong_vf0.999')
for i, v in enumerate((2.3, 44.7, 316.5)):
    check(f"dsprites random rank vf{i}", v, _rank(_pfn[i], 'random'), 'displacement')
for i, v in enumerate((92.8, 183.5, 377.2)):
    check(f"dsprites trained rank vf{i}", v, _rank(_pfn[i], 'trained'), 'displacement')
for i, v in enumerate((9.8, 56.8, 260.9)):
    check(f"shapes3d random rank vf{i}", v, _rank(_cfn[i], 'random'), 'displacement')
for i, v in enumerate((283.5, 435.5, 502.4)):
    check(f"control trained rank vf{i}", v, _rank(_cfn[i], 'trained'), 'displacement')
check("dsprites random rank swing", 136, _rank(_pfn[2], 'random') / _rank(_pfn[0], 'random'),
      'displacement', tol=0.5)
check("dsprites trained rank swing", 4, _rank(_pfn[2], 'trained') / _rank(_pfn[0], 'trained'),
      'displacement', tol=0.1)
check("dsprites trained cond vf0.999", 5984, _cond(_pfn[2], 'trained'), 'displacement', tol=1.0)
check("control trained cond vf0.999", 410, _cond(_cfn[2], 'trained'), 'displacement', tol=1.0)


_pr = [_prt[f][i] for f in ('pos_x', 'pos_y') for i in (1, 2)]
check("position ratio to null low pct", 28, 100 * min(_pr), 'displacement', tol=0.5)
check("position ratio to null high pct", 47, 100 * max(_pr), 'displacement', tol=0.5)


_zs = np.load('results/probes_exploratory/standard_strong/stacks.npz', allow_pickle=True)
_ms = json.load(open('results/probes_exploratory/standard_strong/meta.json'))
_fs = [f['name'] for f in _ms['factors']]
check("standard arm trained seeds", 1, _zs['trained'].shape[0], 'probes_exploratory', 0)
check("standard arm random seeds", 12, _zs['random'].shape[0], 'probes_exploratory', 0)

def _arm(z, names):
    o = {}
    for i, f in enumerate(names):
        dl = z['random'][:, i, 0].mean() - z['trained'][:, i, 0].mean()
        dt = z['random'][:, i, 3].mean() - z['trained'][:, i, 3].mean()
        o[f] = dict(dl=dl, dt=dt, kappa=(1 - dt / dl) if dl > 0 else float('nan'),
                    deep=z['trained'][:, i, 3].mean(), perm=z['perm'][:, i, 3].mean())
    return o

_A_std = _arm(_zs, _fs)
_A_ctl = _arm(_zc, _fc)
_A_col = _arm(_zk, _fk)
_zw = np.load('results/probes_pinned/color_weak/stacks.npz')
_fw = [f['name'] for f in json.load(open('results/probes_pinned/color_weak/meta.json'))['factors']]
_A_wk = _arm(_zw, _fw)

for f, k, deep, perm in [('floor_hue', 0.426, 0.810, 0.633),
                         ('wall_hue', 0.566, 0.911, 0.914),
                         ('object_hue', 0.215, 0.705, 0.360)]:
    check(f"standard kappa {f}", k, _A_std[f]['kappa'], 'probes_exploratory/standard_strong')
    check(f"standard R deep {f}", deep, _A_std[f]['deep'], 'probes_exploratory/standard_strong')
    check(f"standard R_perm deep {f}", perm, _A_std[f]['perm'], 'probes_exploratory/standard_strong')




def _perseed(z, names, f):
    i = names.index(f)
    dl = z['random'][:, i, 0].mean() - z['trained'][:, i, 0]
    dt = z['random'][:, i, 3].mean() - z['trained'][:, i, 3]
    return 1 - dt / dl

for f in _HU:
    _kc, _kk = _perseed(_zc, _fc, f), _perseed(_zk, _fk, f)
    _s = _A_std[f]['kappa']
    (ok if _s < _kc.min() and _s > _kk.max() else bad).append(
        (f"standard kappa outside both arms {f}", "outside",
         (_s, _kc.min(), _kk.max()), 'probes_exploratory/standard_strong'))

_fk_ = [(_A_ctl[f]['kappa'] - _A_std[f]['kappa']) / (_A_ctl[f]['kappa'] - _A_col[f]['kappa']) for f in _HU]
_fl_ = [(_A_std[f]['dl'] - _A_ctl[f]['dl']) / (_A_col[f]['dl'] - _A_ctl[f]['dl']) for f in _HU]
check("standard kappa fraction low", 34.6, 100 * min(_fk_), 'probes_exploratory/standard_strong')
check("standard kappa fraction high", 53.5, 100 * max(_fk_), 'probes_exploratory/standard_strong')
check("standard D_lin fraction low", 5.0, 100 * min(_fl_), 'probes_exploratory/standard_strong')
check("standard D_lin fraction high", 9.6, 100 * max(_fl_), 'probes_exploratory/standard_strong')


for f, deep, k in [('floor_hue', -0.104, -0.161), ('wall_hue', -0.114, -0.205),
                   ('object_hue', -0.189, -0.272)]:
    check(f"color_weak R deep {f}", deep, _A_wk[f]['deep'], 'probes_pinned/color_weak')
    check(f"color_weak kappa {f}", k, _A_wk[f]['kappa'], 'probes_pinned/color_weak')


for nm, arm, nonhue, hue, gap in [("color_weak", _A_wk, 0.984, -0.136, 1.120),
                                  ("standard", _A_std, 0.959, 0.809, 0.150)]:
    _nh = st.mean([arm[f]['deep'] for f in ('scale', 'shape', 'orientation')])
    _hh = st.mean([arm[f]['deep'] for f in _HU])
    check(f"{nm} non-hue mean", nonhue, _nh, 'probes')
    check(f"{nm} hue mean", hue, _hh, 'probes')
    check(f"{nm} hue gap", gap, _nh - _hh, 'probes')


_SD = json.load(open('results/displacement/standard_strong.json'))
_sdm = _mean_by_factor('standard_strong', 'trained')
for f, v in [('floor_hue', 2.23), ('wall_hue', 4.97), ('object_hue', 3.59)]:
    check(f"standard m_total {f}", v, _sdm[f], 'displacement/standard_strong.json')
_sdv = [x['verdict'] for r in _SD['roles']['trained'].values()
        for f, x in r['factors'].items() if f in _HU]
(ok if set(_sdv) == {'moves'} else bad).append(
    ("standard hue verdicts all moves", "moves", set(_sdv), 'displacement/standard_strong.json'))
check("standard arm var fraction", 0.99, _SD['var_fraction'], 'displacement/standard_strong.json')


_aug = open('src/encoders/augmentations.py').read()
for lit in ('T.RandomResizedCrop(image_size, scale=(0.3, 1.0)', 'T.RandomHorizontalFlip()',
            'T.ColorJitter(0.4, 0.4, 0.4, 0.1)', 'T.ColorJitter(0.2, 0.2, 0.2, 0.1)',
            'T.ColorJitter(0.8, 0.8, 0.8, 0.5)'):
    (ok if lit in _aug else bad).append((f"augmentation literal {lit!r}", lit, None,
                                         'src/encoders/augmentations.py'))
_ep = lambda p: 1 + max(json.loads(l)['epoch'] for l in open(p) if 'epoch' in l)
check("standard arm epochs", 100, _ep('results/encoders/standard_strong_seed1/train_log.jsonl'),
      'results/encoders/standard_strong_seed1', 0)
check("color_weak epochs", 50,
      _ep('results/encoders/color_weak/color_weak_seed0/train_log - 2026-08-30T203803.472.jsonl'),
      'results/encoders/color_weak', 0)


for nm, fn, v in [("control", 'control_strong', 435.5), ("color", 'color_strong', 105.8),
                  ("standard", 'standard_strong', 64.0)]:
    check(f"retained rank {nm}", v, _rank(fn, 'trained'), 'displacement')
check("retained rank untrained", 56.8, _rank('control_strong', 'random'), 'displacement')
_alpha = COMP['rank_scaling']['trained/floor_hue']['exponent_alpha']
check("hue rank exponent", 1.9, _alpha, 'scale_companion.json', tol=0.05)
_rr = _rank('control_strong', 'trained') / _rank('standard_strong', 'trained')
check("control over standard rank ratio", 6.8, _rr, 'displacement', tol=0.05)
check("rank ratio worth", 38, _rr ** 1.9, 'displacement', tol=0.5)
_flag = _mean_by_factor('control_strong', 'trained')['floor_hue'] / \
        _mean_by_factor('color_strong', 'trained')['floor_hue']
_frr = _rank('control_strong', 'trained') / _rank('color_strong', 'trained')
check("flagship raw separation", 295, _flag, 'displacement', tol=1.0)
check("flagship rank ratio worth", 15, _frr ** 1.9, 'displacement', tol=0.5)
check("flagship after rank correction", 20, _flag / _frr ** 1.9, 'displacement', tol=0.5)
_srr = _rank('color_strong', 'trained') / _rank('standard_strong', 'trained')
check("color over standard rank worth", 2.6, _srr ** 1.9, 'displacement', tol=0.05)
_sc = [_sdm[f] / _mean_by_factor('color_strong', 'trained')[f] for f in _HU]
check("standard over color raw low", 37, min(_sc), 'displacement', tol=0.5)
check("standard over color raw high", 65, max(_sc), 'displacement', tol=0.5)








import glob as _glob, os as _os

def _ulp(v):
    """Half a unit in the last place of a two-significant-figure quotation."""
    return 0.05 * 10 ** math.floor(math.log10(abs(v)))

_ARM_DIRS = {'color': 'color', 'control': 'control', 'position': 'position',
             'color_weak': 'color_weak', 'standard': 'standard_strong_seed1'}
def _arm_metrics(top):
    rows = []
    for p in sorted(_glob.glob(f'results/encoders/{top}/**/metrics*.json', recursive=True)) \
             + sorted(_glob.glob(f'results/encoders/{top}/metrics*.json')):
        if p in [r[0] for r in rows]: continue
        m = json.load(open(p)); d = m.get('diagnostics') or {}
        rows.append((p, m['final_loss'], d['alignment'], d['uniformity'],
                     m.get('epochs_run'), m.get('seed')))


    seen = {_os.path.dirname(r[0]) for r in rows}
    for d_ in sorted(_glob.glob(f'results/encoders/{top}/*/')):
        if d_.rstrip('/') in seen:
            continue
        logs = sorted(_glob.glob(_os.path.join(d_, 'train_log*.jsonl')))
        if not logs:
            continue
        last = json.loads(open(logs[0]).read().strip().split('\n')[-1])
        rows.append((logs[0], last['loss'], last['alignment'], last['uniformity'],
                     last['epoch'] + 1, None))
    return rows
_TRAIN = {k: _arm_metrics(v) for k, v in _ARM_DIRS.items()}
for _arm, _n, _lo, _hi, _alo, _ahi, _ulo, _uhi, _ep in [
        ('color',      12, 4.9527, 4.9529, 5.5e-5, 4.2e-3, -3.834, -3.815, 50),
        ('control',    12, 4.9488, 4.9490, 9.3e-5, 1.2e-4, -3.885, -3.880, 50),
        ('position',   12, 4.9638, 4.9652, 2.6e-3, 1.1e-2, -3.761, -3.728, 50),
        ('color_weak', 12, 4.9490, 4.9492, 4.8e-5, 7.7e-5, -3.884, -3.879, 50),
        ('standard',    1, 4.9875, 4.9875, 1.1e-2, 1.1e-2, -3.691, -3.691, 100)]:
    _r = _TRAIN[_arm]
    _src = f'results/encoders/{_ARM_DIRS[_arm]}/metrics*.json'
    check(f"M16 {_arm} n_seeds", _n, len(_r), _src, 0)
    check(f"M16 {_arm} epochs", _ep, max(x[4] for x in _r), _src, 0)
    check(f"M16 {_arm} loss lo", _lo, min(x[1] for x in _r), _src)
    check(f"M16 {_arm} loss hi", _hi, max(x[1] for x in _r), _src)
    check(f"M16 {_arm} align lo", _alo, min(x[2] for x in _r), _src, tol=_ulp(_alo))
    check(f"M16 {_arm} align hi", _ahi, max(x[2] for x in _r), _src, tol=_ulp(_ahi))
    check(f"M16 {_arm} unif lo", _ulo, min(x[3] for x in _r), _src)
    check(f"M16 {_arm} unif hi", _uhi, max(x[3] for x in _r), _src)



_pairs = _agree = 0
for _d in sorted(_glob.glob('results/encoders/position/*/')):
    _mf = sorted(_glob.glob(_os.path.join(_d, 'metrics*.json')))
    _lf = sorted(_glob.glob(_os.path.join(_d, 'train_log*.jsonl')))
    if not _mf or not _lf:
        continue
    _mj = json.load(open(_mf[0])); _dg = _mj['diagnostics']
    _pairs += 1
    _agree += any(
        json.loads(open(_f).read().strip().split('\n')[-1])['loss'] == _mj['final_loss']
        and json.loads(open(_f).read().strip().split('\n')[-1])['alignment'] == _dg['alignment']
        and json.loads(open(_f).read().strip().split('\n')[-1])['uniformity'] == _dg['uniformity']
        for _f in _lf)
check("M16 position seeds with both metrics and log", 11, _pairs,
      'results/encoders/position/*', 0)
check("M16 position train_log reproduces metrics", 11, _agree,
      'results/encoders/position/*/train_log*.jsonl', 0)
_p5 = [r for r in _TRAIN['position'] if r[0].endswith('.jsonl')]
check("M16 position log-sourced seeds", 1, len(_p5), 'results/encoders/position/*', 0)
for _lbl, _idx, _lo, _hi in [('loss', 1, 4.9638, 4.9652), ('align', 2, 2.6e-3, 1.1e-2),
                             ('unif', 3, -3.761, -3.728)]:
    check(f"M16 position log-sourced {_lbl} inside printed range", 1,
          1 if _lo <= _p5[0][_idx] <= _hi else 0, _p5[0][0], 0)


check("M16 color median alignment", 6.9e-5, st.median([x[2] for x in _TRAIN['color']]),
      'results/encoders/color', tol=_ulp(6.9e-5))
check("M16 control median alignment", 1.0e-4, st.median([x[2] for x in _TRAIN['control']]),
      'results/encoders/control', tol=_ulp(1.0e-4))
check("M16 color median uniformity", -3.822, st.median([x[3] for x in _TRAIN['color']]),
      'results/encoders/color')
check("M16 control median uniformity", -3.883, st.median([x[3] for x in _TRAIN['control']]),
      'results/encoders/control')
_hi_align = sorted(x[2] for x in _TRAIN['color'] if x[2] > 1e-3)
check("M16 color seeds near 1e-4", 10, sum(1 for x in _TRAIN['color'] if x[2] < 1e-3),
      'results/encoders/color', 0)
check("M16 color outlier alignment lo", 2.5e-3, _hi_align[0], 'results/encoders/color', tol=5e-5)
check("M16 color outlier alignment hi", 4.2e-3, _hi_align[1], 'results/encoders/color', tol=5e-5)
check("M16 standard is highest loss", 1,
      int(max(x[1] for x in _TRAIN['standard']) >
          max(max(x[1] for x in _TRAIN[a]) for a in ['color','control','position','color_weak'])),
      'results/encoders', 0)
instr("The crop-plus-color arm is the one exception and ran 100 epochs")


for _ds in ['shapes3d', 'dsprites']:
    _rows = json.load(open(f'results/calibration/calibration_{_ds}.json'))['results']
    _comp = [r for r in _rows if r['regime'] == 'composition']
    _src = f'results/calibration/calibration_{_ds}.json'
    check(f"M17 {_ds} composition rows", 8, len(_comp), _src, 0)
    check(f"M17 {_ds} distinct probe_train_size", 4,
          len({r['probe_train_size'] for r in _comp}), _src, 0)
    check(f"M17 {_ds} distinct probe_steps", 2, len({r['probe_steps'] for r in _comp}), _src, 0)
    for _sz in (2000, 5000, 10000, 40000):
        check(f"M17 {_ds} size {_sz} present", 1,
              int(_sz in {r['probe_train_size'] for r in _comp}), _src, 0)
    for _stp in (1000, 4000):
        check(f"M17 {_ds} steps {_stp} present", 1,
              int(_stp in {r['probe_steps'] for r in _comp}), _src, 0)
    check(f"M17 {_ds} none clears the gate", 0,
          sum(1 for r in _comp if r['all_factor_headroom_ok']), _src, 0)
check("M17 sixteen across both datasets", 16,
      sum(len([r for r in json.load(open(f'results/calibration/calibration_{d}.json'))['results']
               if r['regime'] == 'composition']) for d in ['shapes3d', 'dsprites']),
      'results/calibration', 0)
instr("eight compositional split configurations per dataset")


_DIAG = {('dsprites', 'orientation')}
_CONF = ['color_strong', 'control_strong', 'position_strong']
_n_conf = 0
_undef_conf, _undef_diag = [], []
for _c in _CONF:
    _m = json.load(open(f'results/probes_pinned/{_c}/meta.json'))
    for _f in _m['factors']:
        _isdiag = (_m['dataset'], _f['name']) in _DIAG
        if not _isdiag: _n_conf += 1
        if not row(_c, _f['name'])['kappa_defined']:
            (_undef_diag if _isdiag else _undef_conf).append((_c, _f['name']))
check("m13 confirmatory readouts", 16, _n_conf, 'results/probes_pinned/*/meta.json', 0)
check("m13 kappa-undefined confirmatory", 5, len(_undef_conf), 'kappa_ci.json', 0)
check("m13 kappa-undefined diagnostic", 1, len(_undef_diag), 'kappa_ci.json', 0)
instr("Five of sixteen confirmatory readouts have nonpositive linear deficits")


_TC = json.load(open('results/displacement/threshold_calibration.json'))
check("m5 fixture is rank one", 2e-14,
      max(abs(r['m_total'] - r['m_shared']) for r in _TC['rows']),
      'threshold_calibration.json', tol=2e-14)
def _sweep_exponents(cell):
    _pts = []
    for _suf in ('_vf0.95', '', '_vf0.999'):
        _d = json.load(open(f'results/displacement/{cell}{_suf}.json'))
        _tr = _d['roles']['trained']
        _k = np.mean([s['whitening']['rank'] for s in _tr.values()])
        _mt = {}
        for _s in _tr.values():
            for _fn, _fv in _s['factors'].items(): _mt.setdefault(_fn, []).append(_fv['m_total'])
        _pts.append((_k, {f: float(np.mean(v)) for f, v in _mt.items()}))
    _out = {}
    for _f in _pts[0][1]:
        _ms = [p[1][_f] for p in _pts]
        if min(_ms) <= 0: continue
        _out[_f] = float(np.polyfit(np.log([p[0] for p in _pts]), np.log(_ms), 1)[0])
    return _out, [p[0] for p in _pts]
_EXP, _KS = {}, None
for _c in ['color_strong', 'control_strong', 'position_strong']:
    _e, _ks = _sweep_exponents(_c)
    _EXP.update({(_c, f): v for f, v in _e.items()})
    if _c == 'color_strong': _KS = _ks
check("m5 exponent low", 0.54, min(_EXP.values()), 'displacement sweep', tol=0.005)
check("m5 exponent high", 1.90, max(_EXP.values()), 'displacement sweep', tol=0.005)
check("m5 every exponent positive", 0, sum(1 for v in _EXP.values() if v <= 0),
      'displacement sweep', 0)
check("m5 sweep rank low", 56, _KS[0], 'displacement sweep', tol=0.5)
check("m5 sweep rank high", 210, _KS[2], 'displacement sweep', tol=0.5)
_cs = json.load(open('results/displacement/color_strong.json'))['roles']['trained']
_sf = [f['shared_fraction'] for s in _cs.values() for f in s['factors'].values()]
check("m5 color shared fraction lo", 0.09, min(_sf), 'color_strong.json', tol=0.005)
check("m5 color shared fraction hi", 0.76, max(_sf), 'color_strong.json', tol=0.005)
instr("The observed $\\epsilon$-destruction is conservative in that direction")


from scipy.stats import wilcoxon as _wcx
def _holm(p):
    p = np.asarray(p, float); m = p.size; adj = np.empty(m); run = 0.0
    for _rk, _ix in enumerate(np.argsort(p, kind='stable')):
        run = max(run, (m - _rk) * p[_ix]); adj[_ix] = min(1.0, run)
    return adj
_H2DIR = {'color_strong': 'results/probes_pinned/color_strong',
          'control_strong': 'results/probes_pinned/control_strong',
          'position_strong': 'results/probes_pinned/position_strong',
          'orientation_strong': 'results/probes_excluded/orientation_strong',
          'scale_strong': 'results/probes_excluded/scale_strong'}
def _h2_pairs(cell):
    _d = _H2DIR[cell]; _m = json.load(open(f'{_d}/meta.json'))
    _fs = _m['factors']; _nm = [f['name'] for f in _fs]
    _bar = {'orientation'} if _m['dataset'] == 'dsprites' else set()
    _z = np.load(f'{_d}/stacks.npz')
    _s = min(_z['trained'].shape[0], _z['random'].shape[0])
    _g = _z['trained'][:_s].astype(float) - _z['random'][:_s].astype(float)
    _dg = _g[:, :, -1] - _g[:, :, 0]
    _out = []
    for _i in range(len(_nm)):
        for _j in range(_i + 1, len(_nm)):
            if _fs[_i]['kind'] != _fs[_j]['kind']: continue
            if _nm[_i] in _bar or _nm[_j] in _bar: continue
            _x = _dg[:, _i] - _dg[:, _j]
            _p = 1.0 if np.allclose(_x, 0) else float(_wcx(_x, alternative='two-sided').pvalue)
            _out.append(((cell, _nm[_i], _nm[_j]), _p))
    return _out
_REAL = [p for c in ['color_strong','control_strong','position_strong'] for p in _h2_pairs(c)]
_DES = [p for c in ['color_strong','position_strong','orientation_strong','scale_strong',
                    'control_strong'] for p in _h2_pairs(c)]
_realh = dict(zip([k for k, _ in _REAL], _holm([p for _, p in _REAL])))
_desh = dict(zip([k for k, _ in _DES], _holm([p for _, p in _DES])))
_sig_real = {k for k, v in _realh.items() if v < 0.05}
_sig_des = {k for k, v in _desh.items() if v < 0.05}

_shipped = {tuple(p) for p in H['hypotheses']['H2']['significant_pairs']}
check("M20 realized pairs", 23, len(_REAL), 'probes_pinned stacks', 0)
check("M20 realized survivors", 12, len(_sig_real), 'probes_pinned stacks', 0)
check("M20 realized survivors match shipped", 0, len(_sig_real ^ _shipped),
      'hypothesis_report.json', 0)
check("M20 designed pairs", 43, len(_DES), 'probes_pinned + probes_excluded stacks', 0)
check("M20 designed survivors", 29, len(_sig_des), 'probes_pinned + probes_excluded stacks', 0)
check("M20 realized survivors lost", 0, len(_sig_real - _sig_des),
      'probes_pinned + probes_excluded stacks', 0)
check("M20 tightest surviving p_holm", 0.0415, max(_desh[k] for k in _sig_real),
      'probes_pinned + probes_excluded stacks', tol=5e-5)
check("M20 wilcoxon floor n=12", 4.88e-4, 2 / 2 ** 12, 'exact two-sided Wilcoxon', tol=5e-7)
check("M20 family size defeating the floor", 103, math.floor(0.05 / (2 / 2 ** 12)) + 1,
      'exact two-sided Wilcoxon', 0)




_FLOOR = 2 / 2 ** 12
_praw = dict(_REAL)
_at_floor = {k for k in _sig_real if abs(_praw[k] - _FLOOR) < 1e-9}
check("M20 realized survivors at the exact floor", 10, len(_at_floor), 'probes_pinned stacks', 0)
check("M20 realized survivors above the floor", 2, len(_sig_real - _at_floor),
      'probes_pinned stacks', 0)
check("M20 family size the floor survives", 102, math.floor(0.05 / _FLOOR),
      'exact two-sided Wilcoxon', 0)
_added = {k for k, _ in _DES} - set(_praw)
check("M20 added pairs", 20, len(_added), 'probes_excluded stacks', 0)
_cfh = dict(zip([k for k, _ in _DES],
                _holm([1.0 if k in _added else p for k, p in _DES])))
check("M20 above-floor survivors without the added pairs", 0.0806,
      max(_cfh[k] for k in _sig_real - _at_floor),
      'probes_pinned + probes_excluded stacks', tol=5e-5)
check("M20 floor survivors lost without the added pairs", 0,
      sum(1 for k in _at_floor if _cfh[k] >= 0.05),
      'probes_pinned + probes_excluded stacks', 0)



_H2ART = {
    'description': 'Holm over the maximal designed H2 family: the five strong cells as '
                   'specified before any exclusion.',
    'alpha': 0.05,
    'n_seeds': 12,
    'wilcoxon_floor_two_sided': _FLOOR,
    'family_size_the_floor_survives': math.floor(0.05 / _FLOOR),
    'source_dirs': dict(_H2DIR),
    'continuous_readouts_voided_by_A10': ['orientation_strong', 'scale_strong'],
    'realized': {'pairs': len(_REAL), 'survivors': len(_sig_real)},
    'designed': {'pairs': len(_DES), 'survivors': len(_sig_des)},
    'realized_survivors_lost_under_designed': sorted('/'.join(k) for k in _sig_real - _sig_des),
    'tests': [{'cell': k[0], 'factor_a': k[1], 'factor_b': k[2],
               'p_raw': float(p),
               'p_holm_designed': float(_desh[k]),
               'p_holm_realized': (float(_realh[k]) if k in _realh else None),
               'in_realized_family': k in _praw,
               'at_wilcoxon_floor': bool(abs(p - _FLOOR) < 1e-9),
               'survives_designed': k in _sig_des}
              for k, p in _DES],
}
with open('results/hypotheses/h2_designed_family.json', 'w') as _fh:
    json.dump(_H2ART, _fh, indent=2, sort_keys=True)
    _fh.write('\n')

instr("Of its forty-three pairs, twenty-nine survive")
instr("survive any family of up to $102$ tests")
instr("The other two have Holm-adjusted $p=0.0415$ at $\\alpha=0.05$ after the twenty comparisons")


from scipy.stats import norm as _norm
_BCA = {}
_rng21 = np.random.default_rng(20260824)
for _cell in ['color_strong', 'control_strong', 'position_strong', 'color_weak']:
    _meta = json.load(open(f'results/probes_pinned/{_cell}/meta.json'))
    _z = np.load(f'results/probes_pinned/{_cell}/stacks.npz')
    _D = (_z['random'] - _z['trained']).astype(float)
    _n = _D.shape[0]
    for _i, _f in enumerate([f['name'] for f in _meta['factors']]):
        _dl, _dt = _D[:, _i, 0], _D[:, _i, -1]
        _ix = _rng21.integers(0, _n, size=(4000, _n))
        if _dl.mean() <= 0: continue
        _kh = 1 - _dt.mean() / _dl.mean()
        _kb = 1 - _dt[_ix].mean(1) / _dl[_ix].mean(1)
        _kb = _kb[np.isfinite(_kb)]
        _plo, _phi = np.percentile(_kb, [2.5, 97.5])
        _pr = min(max(float((_kb < _kh).mean()), 1 / (len(_kb) + 1)), 1 - 1 / (len(_kb) + 1))
        _z0 = _norm.ppf(_pr)
        _jk = np.array([1 - np.delete(_dt, s).mean() / np.delete(_dl, s).mean()
                        for s in range(_n)])
        _dev = _jk.mean() - _jk
        _den = 6.0 * (_dev ** 2).sum() ** 1.5
        _a = (_dev ** 3).sum() / _den if _den > 0 else 0.0
        _ends = []
        for _zq in (_norm.ppf(0.025), _norm.ppf(0.975)):
            _adj = _z0 + (_z0 + _zq) / (1.0 - _a * (_z0 + _zq))
            _ends.append(float(np.percentile(_kb, 100.0 * _norm.cdf(_adj))))
        _BCA[(_cell, _f)] = (_plo, _phi, _ends[0], _ends[1])

        _sr = row(_cell, _f)
        check(f"M21 percentile repro {_cell}:{_f} lo", _plo, _sr['kappa_ci95'][0],
              'kappa_ci.json', tol=1e-6)
        check(f"M21 percentile repro {_cell}:{_f} hi", _phi, _sr['kappa_ci95'][1],
              'kappa_ci.json', tol=1e-6)
check("M21 max endpoint move", 0.0195,
      max(max(abs(v[2] - v[0]), abs(v[3] - v[1])) for v in _BCA.values()),
      'probes_pinned stacks', tol=5e-5)
check("M21 max width ratio", 1.062,
      max((v[3] - v[2]) / (v[1] - v[0]) for v in _BCA.values()),
      'probes_pinned stacks', tol=5e-4)
_cs21 = _BCA[('control_strong', 'shape')]
check("M21 control shape percentile hi", 0.0022, _cs21[1], 'probes_pinned stacks', tol=5e-5)
check("M21 control shape BCa hi", -2e-5, _cs21[3], 'probes_pinned stacks', tol=5e-5)
instr("A bias-corrected and accelerated recomputation changes each endpoint")


_BUD = {}
for _ds in ['shapes3d', 'dsprites']:
    _BUD[_ds] = json.load(open(f'results/calibration/calibration_{_ds}.json'))['results']
def _budget_at(dataset, pin):
    _r = [r for r in _BUD[dataset]
          if (r['regime'], r['probe_train_size'], r['probe_steps']) == pin
          and r['encoder_role'] == 'random']
    assert len(_r) == 1, (dataset, pin, len(_r))
    return float(_r[0]['linear_rung_gap'])

STUDY_PIN = ('interpolation', 2000, 1000)

def _budget(meta):
    return _budget_at(meta['dataset'],
                      (meta['probe_regime'], meta['probe_train_size'], meta['probe_steps']))
_M15 = {}
for _cell in ['color_strong', 'control_strong', 'position_strong', 'color_weak']:
    _meta = json.load(open(f'results/probes_pinned/{_cell}/meta.json'))
    _g = _budget(_meta)
    _z = np.load(f'results/probes_pinned/{_cell}/stacks.npz')
    _D = (_z['random'] - _z['trained']).astype(float)
    for _i, _f in enumerate([f['name'] for f in _meta['factors']]):
        _dl, _dt = _D[:, _i, 0].mean(), _D[:, _i, -1].mean()
        if _dl <= 0: continue
        _k = 1 - _dt / _dl
        _kp, _km = 1 - _dt / (_dl + _g), 1 - _dt / (_dl - _g)
        _M15[(_cell, _f)] = (_dl, _k, min(_kp, _km), max(_kp, _km),
                             max(abs(_kp - _k), abs(_km - _k)), _g)
_ex = ('control_strong', 'shape')
_others = {k: v for k, v in _M15.items() if k != _ex}
check("M15 defined readouts", 14, len(_M15), 'probes_pinned stacks', 0)
check("M15 defined readouts confirmatory", 11,
      sum(1 for k in _M15 if k[0] != 'color_weak'), 'probes_pinned stacks', 0)
check("M15 defined readouts exploratory weak color", 3,
      sum(1 for k in _M15 if k[0] == 'color_weak'), 'probes_pinned stacks', 0)
check("M15 shift excluding control shape", 0.014, max(v[4] for v in _others.values()),
      'probes_pinned stacks', tol=5e-4)
check("M15 color hue shift", 0.013,
      max(v[4] for k, v in _M15.items() if k[0] == 'color_strong'),
      'probes_pinned stacks', tol=5e-4)
check("M15 control wall hue lo", 0.892, _M15[('control_strong', 'wall_hue')][2],
      'probes_pinned stacks', tol=5e-4)
check("M15 control wall hue hi", 0.908, _M15[('control_strong', 'wall_hue')][3],
      'probes_pinned stacks', tol=5e-4)
check("M15 control shape D_linear", 0.0520, _M15[_ex][0], 'probes_pinned stacks', tol=5e-5)
check("M15 control shape budget share", 19, 100 * _M15[_ex][5] / _M15[_ex][0],
      'probes_pinned stacks', tol=0.5)
check("M15 control shape span lo", -0.385, _M15[_ex][2], 'probes_pinned stacks', tol=5e-4)
check("M15 control shape span hi", 0.059, _M15[_ex][3], 'probes_pinned stacks', tol=5e-4)
check("M15 control shape point", -0.120, _M15[_ex][1], 'probes_pinned stacks', tol=5e-4)





_M2DIRS = [f'results/probes_pinned/{_c}/'
           for _c in ('color_strong', 'control_strong', 'position_strong')]
_steps = _drops = _deep = 0
_tsteps = _tdrops = _tdeep = 0
_seen_random = set()
for _d in _M2DIRS:
    _meta = json.load(open(f'{_d}/meta.json'))
    _b = _budget_at(_meta['dataset'], STUDY_PIN)
    _z = np.load(f'{_d}/stacks.npz')
    for _role in ('trained', 'random'):
        if _role == 'random':
            if _meta['dataset'] in _seen_random: continue
            _seen_random.add(_meta['dataset'])
        _df = np.diff(_z[_role].astype(float), axis=2)
        _steps += _df.size; _drops += int((_df < 0).sum()); _deep += int((_df < -_b).sum())
        if _role == 'trained':
            _tsteps += _df.size; _tdrops += int((_df < 0).sum()); _tdeep += int((_df < -_b).sum())
check("M2 rung-to-rung steps", 1008, _steps, 'confirmatory probe stacks', 0)
check("M2 decreasing steps", 207, _drops, 'confirmatory probe stacks', 0)
check("M2 decreases past the budget", 114, _deep, 'confirmatory probe stacks', 0)
check("M2 trained-role steps", 612, _tsteps, 'confirmatory probe stacks', 0)
check("M2 trained-role decreasing steps", 143, _tdrops, 'confirmatory probe stacks', 0)
_cz = np.diff(np.load('results/probes_pinned/color_strong/stacks.npz')['trained'].astype(float), axis=2)
_cf = [f['name'] for f in json.load(open('results/probes_pinned/color_strong/meta.json'))['factors']]
for _f, _want in [('floor_hue', 20), ('wall_hue', 21), ('object_hue', 28)]:
    check(f"M2 color {_f} decreasing", _want, int((_cz[:, _cf.index(_f), :] < 0).sum()),
          'color_strong stacks', 0)
check("M2 color steps per readout", 36, _cz[:, 0, :].size, 'color_strong stacks', 0)
check("M2 color worst drop", 0.157, float(-_cz.min()), 'color_strong stacks', tol=5e-4)
_ctz = np.diff(np.load('results/probes_pinned/control_strong/stacks.npz')['trained'].astype(float), axis=2)
check("M2 control decreasing", 4, int((_ctz < 0).sum()), 'control_strong stacks', 0)
check("M2 control steps", 216, _ctz.size, 'control_strong stacks', 0)
instr("The selected values are unavailable in the released artifacts")
instr("with the shared\nuntrained array counted once per dataset")



_Z = {c: np.load(f'results/probes_pinned/{c}/stacks.npz') for c in
      ['color_strong', 'control_strong', 'color_weak']}
check("M6 shared untrained array control", 1,
      int(np.array_equal(_Z['color_strong']['random'], _Z['control_strong']['random'])),
      'probes_pinned stacks', 0)
check("M6 shared untrained array weak", 1,
      int(np.array_equal(_Z['color_strong']['random'], _Z['color_weak']['random'])),
      'probes_pinned stacks', 0)
_HUE = ['floor_hue', 'wall_hue', 'object_hue']
_fc = [f['name'] for f in json.load(open('results/probes_pinned/color_strong/meta.json'))['factors']]
_fk = [f['name'] for f in json.load(open('results/probes_pinned/control_strong/meta.json'))['factors']]
for _f, _cv, _kv in [('floor_hue', -0.079, 0.935), ('wall_hue', -0.140, 0.985),
                     ('object_hue', -0.255, 0.905)]:
    _tc = _Z['color_strong']['trained'].astype(float)[:, _fc.index(_f), -1]
    _tk = _Z['control_strong']['trained'].astype(float)[:, _fk.index(_f), -1]
    check(f"M6 color top {_f}", _cv, _tc.mean(), 'color_strong stacks')
    check(f"M6 control top {_f}", _kv, _tk.mean(), 'control_strong stacks')
    check(f"M6 seed ranges disjoint {_f}", 1, int(_tc.max() < _tk.min()),
          'probes_pinned stacks', 0)
_gc = _Z['color_strong']['trained'].astype(float) - _Z['color_strong']['random'].astype(float)
_gk = _Z['control_strong']['trained'].astype(float) - _Z['control_strong']['random'].astype(float)
_rng6 = np.random.default_rng(0)
for _f, _m, _lo, _hi in [('floor_hue', 0.246, 0.170, 0.314), ('wall_hue', 0.184, 0.147, 0.221),
                         ('object_hue', 0.446, 0.429, 0.461)]:
    _i, _j = _fc.index(_f), _fk.index(_f)
    _d = (_gk[:, _j, -1] - _gk[:, _j, 0]) - (_gc[:, _i, -1] - _gc[:, _i, 0])
    _bs = np.array([_d[_rng6.integers(0, len(_d), len(_d))].mean() for _ in range(4000)])
    check(f"M6 dDelta_G {_f}", _m, _d.mean(), 'probes_pinned stacks')
    check(f"M6 dDelta_G lo {_f}", _lo, np.percentile(_bs, 2.5), 'probes_pinned stacks', tol=5e-3)
    check(f"M6 dDelta_G hi {_f}", _hi, np.percentile(_bs, 97.5), 'probes_pinned stacks', tol=5e-3)

    _tro = ((_Z['control_strong']['trained'].astype(float)[:, _j, -1]
             - _Z['control_strong']['trained'].astype(float)[:, _j, 0])
            - (_Z['color_strong']['trained'].astype(float)[:, _i, -1]
               - _Z['color_strong']['trained'].astype(float)[:, _i, 0]))
    check(f"M6 floor cancels exactly {_f}", 1, int(np.allclose(_d, _tro)),
          'probes_pinned stacks', 0)

_cap, _ratios = [], []
for _c in ['color_strong', 'control_strong', 'position_strong', 'color_weak']:
    _mm = json.load(open(f'results/probes_pinned/{_c}/meta.json'))
    _zz = np.load(f'results/probes_pinned/{_c}/stacks.npz')
    _DD = (_zz['random'] - _zz['trained']).astype(float)
    _rr = _zz['random'].astype(float)
    for _i, _f in enumerate([f['name'] for f in _mm['factors']]):
        _dl, _dt = _DD[:, _i, 0].mean(), _DD[:, _i, -1].mean()
        if _dl <= 0: continue
        _ratio = _rr[:, _i, 0].mean() / _rr[:, _i, -1].mean()
        _ratios.append(_ratio)
        if _c == 'control_strong' and _f in _HUE:
            _cap.append(abs((1 - (1 - _dt / _dl)) * (1 - _ratio)))
check("M6 floor ratio lo", 0.906, min(_ratios), 'probes_pinned stacks', tol=5e-4)
check("M6 floor ratio hi", 0.979, max(_ratios), 'probes_pinned stacks', tol=5e-4)
check("M6 control hue reparam cap", 0.025, max(_cap), 'probes_pinned stacks', tol=1e-3)
_cD = [(_Z['color_strong']['random'] - _Z['color_strong']['trained']).astype(float)[:, _fc.index(f), 0].mean()
       for f in _HUE]
_cR = [_Z['color_strong']['random'].astype(float)[:, _fc.index(f), 0].mean() for f in _HUE]
check("M6 color D_lin lo", 0.99, min(_cD), 'color_strong stacks', tol=5e-3)
check("M6 color D_lin hi", 1.07, max(_cD), 'color_strong stacks', tol=5e-3)
check("M6 color floor lo", 0.88, min(_cR), 'color_strong stacks', tol=5e-3)
check("M6 color floor hi", 0.97, max(_cR), 'color_strong stacks', tol=5e-3)
check("M6 deficit exceeds its floor", 3, sum(1 for a, b in zip(_cD, _cR) if a > b),
      'color_strong stacks', 0)
_ht = [1 - _Z['control_strong']['random'].astype(float)[:, _fk.index(f), -1] for f in _HUE]
check("M6 A8c denominator lo", 0.0027, min(h.mean() for h in _ht), 'control_strong stacks', tol=5e-5)
check("M6 A8c denominator hi", 0.0243, max(h.mean() for h in _ht), 'control_strong stacks', tol=5e-5)
check("M6 A8c denominator cv lo", 0.30, min(h.std(ddof=1) / abs(h.mean()) for h in _ht),
      'control_strong stacks', tol=5e-3)
check("M6 A8c denominator cv hi", 0.42, max(h.std(ddof=1) / abs(h.mean()) for h in _ht),
      'control_strong stacks', tol=5e-3)
instr("The deficit has no analogous upper bound because negative $R^2$ is retained")
instr("it does not test the concern it appears to answer")








_R3_HUE = ('floor_hue', 'wall_hue', 'object_hue')
_r3_mono, _r3_reff = [], []
for _cell, _facs in (('color_strong', _R3_HUE), ('control_strong', _R3_HUE),
                     ('position_strong', ('pos_x', 'pos_y'))):
    _d = json.load(open(f'results/displacement/{_cell}.json'))
    for _role, _seeds in _d['roles'].items():
        _v = {_f: [s['factors'][_f] for s in _seeds.values() if _f in s['factors']]
              for _f in _facs}
        for _f, _rows in _v.items():
            _r3_mono += [x['monotonicity']['monotone_fraction'] for x in _rows]
            if _role == 'trained':
                _r3_reff.append(st.mean(x['r_eff'] for x in _rows))
check("M7 monotone fraction on every hue and position readout", 0.0, max(_r3_mono),
      'displacement', 0)
check("M7 effective rank low", 45.6, min(_r3_reff), 'displacement', tol=0.05)
check("M7 effective rank high", 353.3, max(_r3_reff), 'displacement', tol=0.05)


_prereg = open('preregistration/prereg.md').read()
_apos = {_a: _prereg.find(f'### {_a} ') for _a in ('A6', 'A9', 'A10')}
check("M8 A6 found in prereg", 1, int(min(_apos.values()) >= 0), 'prereg.md', 0)
check("M8 A6 sits after A9", 1, int(_apos['A9'] < _apos['A6']), 'prereg.md', 0)
check("M8 A6 sits before A10", 1, int(_apos['A6'] < _apos['A10']), 'prereg.md', 0)


instr("The ladder and permuted-label task are both probe based")
instr("This gate is strongest on a complete factorial grid")
instr("whole-scene affine zoom in place of the preregistered random-resized crop")
instr("on the shape gate as it then stood")
instr("Pairing trained and untrained encoders controls architecture and data splits, but it does not cancel probe memorization")


_r3_ct = _mean_by_factor('control_strong', 'trained')
_r3_cc = _mean_by_factor('color_strong', 'trained')
_r3_rr = (_rank('control_strong', 'trained') / _rank('color_strong', 'trained')) ** _alpha
check("M3 exponent-corrected separation, min over hues", 20,
      min(_r3_ct[f] / _r3_cc[f] / _r3_rr for f in _HU), 'displacement', tol=0.5)
check("M3 exponent-corrected separation, max over hues", 23,
      max(_r3_ct[f] / _r3_cc[f] / _r3_rr for f in _HU), 'displacement', tol=0.5)
check("M3 raw separation, min over hues", 295,
      min(_r3_ct[f] / _r3_cc[f] for f in _HU), 'displacement', tol=1.0)
instr("Exclusion instead follows from the preregistered analytic argument")







_r4col = json.load(open('results/displacement/color_strong.json'))
_r4band = _r4col['unadjudicated_band']
for _f, _ndest in (('floor_hue', 12), ('wall_hue', 7), ('object_hue', 12)):
    _ms = [s['factors'][_f]['m_total'] for s in _r4col['roles']['trained'].values()]
    check(f"m1 {_f} seeds", 12, len(_ms), 'displacement/color_strong.json', 0)
    check(f"m1 {_f} epsilon-destroyed seeds", _ndest,
          sum(1 for m in _ms if m < _r4band[0]), 'displacement/color_strong.json', 0)
    check(f"m1 {_f} moving seeds", 0,
          sum(1 for m in _ms if m > _r4band[1]), 'displacement/color_strong.json', 0)
instr("Floor and object hue are $\\epsilon$-destroyed on all twelve seeds")


check("m2 color-arm intervals containing zero", 1,
      sum(1 for _f in ('floor_hue', 'wall_hue', 'object_hue')
          if row('color_strong', _f)['kappa_ci95'][0] <= 0 <= row('color_strong', _f)['kappa_ci95'][1]),
      'kappa_ci.json', 0)
check("m2 control-arm intervals containing zero", 0,
      sum(1 for _f in ('floor_hue', 'wall_hue', 'object_hue')
          if row('control_strong', _f)['kappa_ci95'][0] <= 0 <= row('control_strong', _f)['kappa_ci95'][1]),
      'kappa_ci.json', 0)
instr("For floor hue, the control interval is positive while the color interval includes zero")



check("m4 calibration fixture width", 512,
      json.load(open('results/displacement/threshold_calibration.json'))['dims'],
      'threshold_calibration.json', 0)
instr("uses the single embedding width 512. Width independence is therefore derived")


_m6margin = (_s4['pos_y']['m_total'] - _EDGE) * 1e5
_m6dev = NULL['device_check']['linear_rung_max_dev'] * 1e5
check("m6 pos_y margin x1e5", 3.30, _m6margin, 'displacement', tol=5e-3)
check("m6 null device deviation x1e5", 3.31, _m6dev, 'kappa_null.json', tol=5e-3)
check("m6 the two differ at three significant figures", 1,
      int(round(_m6margin, 2) != round(_m6dev, 2)), 'derived', 0)

check("m6 no ambiguous bare 3.3e-5 remains", 0,
      PROSE_FLAT.count('3.3 \\times 10^{-5}'), 'paper/main_rewritten.tex', 0)
check("m6 the margin is quoted at three s.f. everywhere it appears", 2,
      PROSE_FLAT.count('3.30 \\times 10^{-5}'), 'paper/main_rewritten.tex', 0)


instr("against the recomputed run's own $\\kappa$ of $+0.785$, $+0.900$ and $+0.739$")
instr("These are descriptive distances")


instr("attributing head-induced invariance to metric degeneracy")
instr("The capacity effect in our minimal-augmentation arm resembles linear concept erasure in reverse")
instr("We reverse that operation by changing the target factor at a fixed context")
instr("EditCLEVR is one possible setting")
instr("We tested this property for displacement only")



check("m3 the old over-count is gone", 0, TEX.count('four cells, two datasets'), 'paper/main_rewritten.tex', 0)
check("m12 gate exemption on kappa, plane and ladder captions", 3,
      sum(int(_s in PROSE_FLAT) for _s in (
          'The Control-aug baseline is exempt from the quality gate under amendment A2.',
          'Control-aug is exempt from the quality gate under amendment A2.',
          'The Control-aug points use the A2 baseline exemption.')),
      'paper/main_rewritten.tex', 0)


for _fn in ('color_strong', 'control_strong', 'position_strong',
            'control_strong_vf0.95', 'position_strong_vf0.95'):
    _dd = json.load(open(f'results/displacement/{_fn}.json'))
    for _role in ('trained', 'random'):
        check(f"m14 {_fn}/{_role} rank is averaged over twelve seeds", 12,
              len(_dd['roles'][_role]), 'displacement', 0)
check("m14 a quoted rank really is non-integer", 1,
      int(any(_rank(_f, _r) % 1 for _f in ('control_strong', 'position_strong_vf0.95')
              for _r in ('trained', 'random'))), 'displacement', 0)
instr("every reported rank is a mean over twelve seeds")


check("m15 exploratory rows in the conditions table", 3,
      TEX.split('label{tab:conditions}')[1].split('bottomrule')[0].count('\\textsuperscript{e}'),
      'paper/main_rewritten.tex', 0)
instr("Three exploratory arms are reported separately")


_BIB = open('paper/references.bib').read()
for _lit in ('@inproceedings{foster2021probing,',
             'booktitle = {Proceedings of ICLR}',
             '@inproceedings{balestriero2022contrastive,',
             'note      = {arXiv:2205.11508}',
             '@misc{cosentino2022geometry,',
             'Toward a Geometrical Understanding of Self-supervised Contrastive Learning',
             'Sengupta, Anirvan'):
    check(f"m16 bib literal {_lit[:44]!r}", 1, int(_lit in _BIB), 'paper/references.bib', 0)
check("m16 no arXiv-typed journal fields left", 0,
      _BIB.count('journal = {arXiv preprint}'), 'paper/references.bib', 0)



_ARM18 = {'Color, strong': 'color_strong', 'Control-aug': 'control_strong'}
_FAC18 = {'Floor hue': 'floor_hue', 'Wall hue': 'wall_hue', 'Object hue': 'object_hue',
          'Scale': 'scale', 'Shape': 'shape', 'Orientation': 'orientation'}
_t18 = TEX.split('label{tab:kappanull}')[1].split('\\midrule', 1)[1]
_t18 = _t18.split('\\bottomrule')[0].replace('\\midrule', '')
_n18 = lambda s: [float(x) for x in re.findall(r'[-+]?\d*\.\d+', s)]
_arm18, _rows18 = None, 0
for _line in _t18.split('\\\\'):
    _cells = [c.strip() for c in _line.split('&')]
    if len(_cells) != 6:
        continue
    if _cells[0]:
        _arm18 = _ARM18[_cells[0]]
    _r18 = next(x for x in NULL['cells'][_arm18] if x['factor'] == _FAC18[_cells[1]])
    _lbl = f"m18 {_arm18}:{_r18['factor']}"
    check(f"{_lbl} kappa", _n18(_cells[2])[0], _r18['kappa'], 'kappa_null.json')
    check(f"{_lbl} kappa ci lo", _n18(_cells[3])[0], _r18['kappa_ci95'][0], 'kappa_null.json')
    check(f"{_lbl} kappa ci hi", _n18(_cells[3])[1], _r18['kappa_ci95'][1], 'kappa_null.json')
    check(f"{_lbl} kappa_perm", _n18(_cells[4])[0], _r18['kappa_perm'], 'kappa_null.json')
    check(f"{_lbl} perm ci lo", _n18(_cells[5])[0], _r18['kappa_perm_ci95'][0], 'kappa_null.json')
    check(f"{_lbl} perm ci hi", _n18(_cells[5])[1], _r18['kappa_perm_ci95'][1], 'kappa_null.json')
    _rows18 += 1
check("m18 table lists every readout where both are defined", _rows18,
      sum(1 for c in ('color_strong', 'control_strong')
          for x in NULL['cells'][c] if x['kappa_defined']), 'kappa_null.json', 0)



_h18 = [(c, x) for c in ('color_strong', 'control_strong')
        for x in NULL['cells'][c] if 'hue' in x['factor']]
check("m18 null keeps kappa's sign on every hue readout", 6,
      sum(1 for _, x in _h18 if x['kappa'] * x['kappa_perm'] > 0), 'kappa_null.json', 0)
check("m18 null keeps the arms apart", 1,
      int(max(x['kappa_perm_ci95'][1] for c, x in _h18 if c == 'color_strong')
          < min(x['kappa_perm_ci95'][0] for c, x in _h18 if c == 'control_strong')),
      'kappa_null.json', 0)
_f18 = next(x for c, x in _h18 if c == 'color_strong' and x['factor'] == 'floor_hue')
check("m18 color floor hue null over point estimate", 9.8,
      _f18['kappa_perm'] / _f18['kappa'], 'kappa_null.json', tol=0.05)
instr("On color-arm floor hue, its magnitude is $9.8$ times the observed point estimate")



_i19 = lambda f: NULL['factors'].index(f)
for _f, _v in (('wall_hue', 0.024), ('floor_hue', 0.057), ('object_hue', 0.110),
               ('scale', 0.665), ('orientation', 0.529), ('shape', -0.0002)):
    check(f"m19 control-task cost {_f}", _v,
          NULL['random_real_mean'][_i19(_f)][0] - NULL['random_perm_mean'][_i19(_f)][0],
          'kappa_null.json')
instr("permutation reduces recoverability by $+0.024$ on wall hue")


for _cell, _vals in (('control_strong', ((0.286, 0.062), (0.128, 0.013), (0.276, 0.071))),
                     ('color_strong', ((1.054, 1.076), (1.069, 1.138), (0.989, 1.230)))):
    for _f, (_dl, _dt) in zip(('floor_hue', 'wall_hue', 'object_hue'), _vals):
        _r20 = row(_cell, _f)
        check(f"m20 {_cell}:{_f} D_linear", _dl, _r20['D_linear'], 'kappa_ci.json')
        check(f"m20 {_cell}:{_f} D_top", _dt, _r20['D_top'], 'kappa_ci.json')


for _a, _d in (('A11', '2026-08-13'), ('A12', '2026-08-13'), ('A15', '2026-08-19')):
    check(f"m22 {_a} is dated {_d}", 1,
          int(_prereg.split(f'### {_a} ')[1][:60].startswith(f'\u2014 {_d} ')), 'prereg.md', 0)
check("m22 six days between the first and last", 6, 19 - 13, 'derived', 0)
instr("went through three versions in six days, A11 and A12 on 2026-08-13 and A15 on 2026-08-19")



import os
check("R4 no weak-arm displacement artifact exists", 0,
      sum(1 for _f in os.listdir('results/displacement') if 'weak' in _f),
      'results/displacement', 0)
instr("The crop-free weak-color arm also widens")



instr("its encoders are not part of the realized grid and we do\nnot release them")




_s1 = _mean_by_factor('control_strong', 'trained')
check("S1 control hue m_total low", 17.6, min(_s1[_f] for _f in _HU), 'displacement', tol=0.05)
check("S1 control hue m_total high", 23.6, max(_s1[_f] for _f in _HU), 'displacement', tol=0.05)



_s1b = []
for _suf in ('_vf0.95', '', '_vf0.999'):
    _d = json.load(open(f'results/displacement/color_strong{_suf}.json'))
    for _f in _HU:
        _t = st.mean([r['factors'][_f]['m_total'] for r in _d['roles']['trained'].values()])
        _r = st.mean([r['factors'][_f]['m_total'] for r in _d['roles']['random'].values()])
        _s1b.append(-math.log10(_t / _r))
check("S1 orders below parity, least", 1.64, min(_s1b), 'displacement sweep', tol=0.02)
check("S1 orders below parity, most", 1.97, max(_s1b), 'displacement sweep', tol=0.02)
check("S1 orders below parity stay inside the quoted band", 0,
      sum(1 for _v in _s1b if not 1.6 <= _v <= 2.0), 'displacement sweep', 0)


_kn = json.load(open('results/hypotheses/kappa_null.json'))
check("S4 null cells", 2, len(_kn['cells']), 'kappa_null.json', 0)
check("S4 null random encoders", 12, len(_kn['seeds']), 'kappa_null.json', 0)
check("S4 null trained encoders", 24, len(_kn['seeds']) * len(_kn['cells']), 'kappa_null.json', 0)
check("S4 null readouts where kappa is defined", 9,
      sum(1 for _c in _kn['cells'].values() for _r in _c
          if _r.get('kappa') is not None and _r['kappa'] == _r['kappa']), 'kappa_null.json', 0)
check("S4 null is shapes3d only", 1, int(_kn['dataset'] == 'shapes3d'), 'kappa_null.json', 0)




_s9_ctrl = {_f: v for (_c, _f), v in _EXP.items() if _c == 'control_strong' and _f in _HU}
check("S9 control-arm hue exponent low", 0.60, min(_s9_ctrl.values()), 'displacement sweep', tol=0.01)
check("S9 control-arm hue exponent high", 0.67, max(_s9_ctrl.values()), 'displacement sweep', tol=0.01)
_s9_rr = _rank('control_strong', 'trained') / _rank('color_strong', 'trained')
check("S9 multiple at a rounded 0.7 exponent", 5.5, _s9_rr ** (_alpha - 0.7),
      'displacement', tol=0.05)
check("S9 quoted five is not an overstatement", 0,
      sum(1 for _f in _HU if _s9_rr ** (_alpha - _s9_ctrl[_f]) < 5.0), 'displacement', 0)
instr("Using the Control-aug exponent leaves a gap about five times larger")








instr("Those arms also apply disjoint augmentation sets")
instr("we exempt the minimal-augmentation baseline from that gate, which it "
      "fails on all twelve seeds")
instr("The two arms apply disjoint augmentation sets and differ in gate status")
instr("across two arms that apply disjoint augmentation sets and differ in gate status")


instr("where exploratory measurements suggest a color shortcut in its "
      "single-augmentation recipe")
instr("two exploratory measurements, one of them a single-seed pilot, point at its "
      "single-transform recipe")


instr("more than two orders of magnitude before rank correction and by roughly twentyfold")
instr("about twenty to twenty-three times apart once we correct for their unlike "
      "whitening rank")


_v5c = [row('control_strong', _f) for _f in _HU]
check("V5 M4 control top-rung deficit low", 0.013, min(_r['D_top'] for _r in _v5c),
      'kappa_ci.json')
check("V5 M4 control top-rung deficit high", 0.071, max(_r['D_top'] for _r in _v5c),
      'kappa_ci.json')
check("V5 M4 control hue is below its untrained floor at the top rung", 3,
      sum(1 for _r in _v5c if _r['R_trained_top'] < _r['R_random_top']), 'kappa_ci.json', 0)
instr("so hue recoverability ends within $0.013$ to $0.071$ of the untrained floor")



instr("We chose this paper's organization after unblinding")
instr("which amendment A14 recorded before the blind broke")


instr("and wall hue on seven of them, at variance fractions at or below $0.99$")
instr("on all twelve seeds for floor and object hue and on seven for wall hue")


instr("treating all twenty as uninformative moves the two to a Holm-adjusted $0.0806$")


instr("We excluded orientation under the gate as it then stood")
instr("Under the two-valued gate in force when we removed that arm")



for _lbl in ('fig:kappa', 'fig:disp', 'fig:disprank', 'tab:notation', 'eq:delta'):
    (ok if f"ref{{{_lbl}}}" in PROSE_FLAT else bad).append(
        (f"m1 {_lbl} is referenced from the prose", None, None, 'paper/main_rewritten.tex'))


instr("The probe-side instruments come from prior work")
instr("the observation that a nonlinear probe recovers what a linear one misses is "
      "established across domains")



for _f, _ctrl, _col in (('floor_hue', 0.6848, -0.0835), ('wall_hue', 0.8458, -0.0952),
                        ('object_hue', 0.6082, -0.1056)):
    check(f"lin R control {_f}", _ctrl, row('control_strong', _f)['R_trained_lin'], 'kappa_ci.json')
    check(f"lin R color {_f}", _col, row('color_strong', _f)['R_trained_lin'], 'kappa_ci.json')
_gaps = [row('control_strong', _f)['R_trained_lin'] - row('color_strong', _f)['R_trained_lin']
         for _f in _hues]
check("linear-rung separation low", 0.714, min(_gaps), 'kappa_ci.json')
check("linear-rung separation high", 0.941, max(_gaps), 'kappa_ci.json')
instr("a closure statistic separates two augmentation regimes the thresholded "
      "linear-probe verdict conflates")
instr("the linear rung's scores separate those arms by $0.714$ to $0.941$ of the "
      "normalized scale")




_m = re.search(r'checks (\d+) assertions', PROSE_FLAT)
_total = None
if _m is None:
    bad.append(("trace assertion count sentence missing", None, None, 'paper/main_rewritten.tex'))
else:
    _total = len(ok) + len(bad) + 1
    check("trace assertion count", int(_m.group(1)), _total,
          'src/eval/trace_numbers.py', 0)

print(f"\nPASS {len(ok)}   FAIL {len(bad)}\n")
for lbl, c, a, src in bad:
    print(f"  FAIL {lbl}: paper says {c}, artifact says {a}   [{src}]")
if _total is not None and len(ok) + len(bad) != _total:
    print(f"  FAIL assertion count self-check ran against {_total} assertions but "
          f"{len(ok) + len(bad)} ran in total. A check was added below it; move it back "
          f"to the end of src/eval/trace_numbers.py.")
    sys.exit(1)
sys.exit(1 if bad else 0)
