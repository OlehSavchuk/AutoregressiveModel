"""
make_figures.py - figures for the README. Needs only numpy and matplotlib (no TensorFlow).

  fig_made_structure.png   which inputs each output site can see (MADE masks)
  fig_sampling_cost.png    original vs incremental sampling: measured time (NumPy reference) and operation counts
  fig_phi4_reference.png   reference Metropolis simulation of the same lattice phi^4 model the network is trained on
                           (shows the double-well structure and why local updates mix slowly)

    python make_figures.py            # all figures (about 1-2 minutes)
    python make_figures.py --quick    # shorter MCMC and timing
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUICK = '--quick' in sys.argv
plt.rcParams.update({'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.minor.visible': True, 'ytick.minor.visible': True, 'font.size': 10,
                     'axes.spines.top': True, 'axes.spines.right': True})
C_OLD, C_NEW, C_GRAY = '#c0392b', '#1f77b4', '0.45'

# ---------------------------------------------------------------- the model (NumPy mirror of the notebook's MADE layer)
D, H, P = 100, 2048, 5
def make_masks(D, H, P):
    m = 1 + np.arange(H) % (D - 1)                                      # hidden degrees; input j has degree j+1, output site i degree i+1
    mask1 = (m[None, :] >= np.arange(1, D + 1)[:, None]).astype(np.float32)                              # [D, H]
    mask2 = np.broadcast_to(np.arange(1, D + 1)[None, :, None] > m[:, None, None], (H, D, P)).astype(np.float32)   # [H, D, P]
    return m, mask1, mask2

def post(o):
    l1, l2, s1, s2, a = [o[..., k] for k in range(5)]
    sig = lambda z: 1 / (1 + np.exp(-z))
    return np.stack([4 * np.tanh(l1) + 0.4, 4 * np.tanh(l2) - 0.4, sig(s1), sig(s2), sig(a)], -1)

class MADE:
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        _, self.m1, self.m2 = make_masks(D, H, P)
        self.W1 = (r.standard_normal((D, H)) * np.sqrt(2 / (D + H)) * 3).astype(np.float32); self.b1 = (r.standard_normal(H) * 0.1).astype(np.float32)
        self.W2 = (r.standard_normal((H, D, P)) * np.sqrt(2 / (H + D * P)) * 3).astype(np.float32); self.b2 = (r.standard_normal((D, P)) * 0.1).astype(np.float32)
        self.W1m, self.W2m = self.W1 * self.m1, self.W2 * self.m2
    def full(self, x):                                                   # all sites at once
        h = np.maximum((0.5 * x) @ self.W1m + self.b1, 0)
        return post(np.einsum('nh,hdp->ndp', h, self.W2m) + self.b2)
    def sample_original(self, u, z):                                     # full network at every site (the original scheme)
        n = u.shape[0]; cfg = np.zeros((n, D), np.float32)
        for i in range(D):
            l1, l2, s1, s2, a = [self.full(cfg)[:, i, k] for k in range(5)]
            first = u[:, i] < a
            cfg[:, i] = np.where(first, l1, l2) + np.where(first, s1, s2) * z[:, i]
        return cfg
    def sample_incremental(self, u, z):                                  # reuse the hidden pre-activation (the new scheme)
        n = u.shape[0]; h_pre = np.tile(self.b1, (n, 1)); xs = np.zeros((n, D), np.float32)
        for i in range(D):
            l1, l2, s1, s2, a = [post(np.maximum(h_pre, 0) @ self.W2m[:, i, :] + self.b2[i])[:, k] for k in range(5)]
            first = u[:, i] < a
            xs[:, i] = np.where(first, l1, l2) + np.where(first, s1, s2) * z[:, i]
            h_pre += 0.5 * xs[:, i][:, None] * self.W1m[i][None, :]
        return xs

# ---------------------------------------------------------------- figure 1: structure
def fig_structure():
    Ds, Hs = 10, 20
    ms, m1s, m2s = make_masks(Ds, Hs, 1)
    _, m1, m2 = make_masks(D, H, P)
    dep = ((m1.astype(np.float64) @ m2[:, :, 0].astype(np.float64)) > 0)                 # [input j, output site i]: can i see j?
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={'width_ratios': [1, 1.3, 1.2]})
    ax[0].imshow(m1s.T, cmap='Blues', aspect='auto', vmin=0, vmax=1.2, origin='upper'); ax[0].set_title('input -> hidden mask (D=10, H=20)')
    ax[0].set_xlabel('input site j'); ax[0].set_ylabel('hidden unit k'); ax[0].set_xticks(range(0, Ds, 3))
    ax[1].imshow(m2s[:, :, 0].T, cmap='Oranges', aspect='auto', vmin=0, vmax=1.2, origin='upper'); ax[1].set_title('hidden -> output mask')
    ax[1].set_xlabel('hidden unit k'); ax[1].set_ylabel('output site i')
    for a in ax[:2]: a.tick_params(top=True, right=True)
    ax[2].imshow(dep.T, cmap='Greens', aspect='equal', vmin=0, vmax=1.2, origin='upper')
    ax[2].plot([0, D - 1], [0, D - 1], color='k', lw=0.8, ls='--')
    ax[2].set_title('what output site i can see (D=100, H=2048)'); ax[2].set_xlabel('input site j'); ax[2].set_ylabel('output site i')
    ax[2].text(70, 25, 'blind (j >= i)', ha='center', color='k'); ax[2].text(25, 75, 'j < i : visible', ha='center', color='white')
    plt.tight_layout(); plt.savefig('figures/fig_made_structure.png', dpi=140); plt.close()
    print('structure: output i is blind to inputs j >= i:', not dep[np.tril_indices(D)].any(), '| sees every input j < i:', bool(dep[np.triu_indices(D, 1)].all()))

# ---------------------------------------------------------------- figure 2: sampling cost
def fig_cost():
    net = MADE(0); rng = np.random.default_rng(1)
    ns = [8, 16, 32, 64] if QUICK else [8, 16, 32, 64, 128]
    t_old, t_new = [], []
    for n in ns:
        u = rng.random((n, D)).astype(np.float32); z = rng.standard_normal((n, D)).astype(np.float32)
        t0 = time.time(); a = net.sample_original(u, z); t_old.append(time.time() - t0)
        t0 = time.time(); b = net.sample_incremental(u, z); t_new.append(time.time() - t0)
        assert np.abs(a - b).max() < 1e-3, 'schemes disagree'
        print(f'  n={n:4d}: original {t_old[-1]:6.2f} s   incremental {t_new[-1]:6.3f} s   x{t_old[-1] / t_new[-1]:.0f}   max|diff| {np.abs(a - b).max():.1e}')
    fl_old = lambda n: D * (2 * n * D * H + 2 * n * H * D * P) / 1e9
    fl_new = lambda n: D * (2 * n * H * P + 3 * n * H) / 1e9
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(ns, t_old, 'o-', color=C_OLD, label='original: full network at every site')
    ax[0].loglog(ns, t_new, 's-', color=C_NEW, label='new: incremental hidden layer')
    ax[0].set_xlabel('batch size N'); ax[0].set_ylabel('time per batch [s]'); ax[0].legend(frameon=False)
    ax[0].set_title(f'measured (NumPy reference, D={D}, H={H})'); ax[0].text(0.03, 0.06, f'x{t_old[-1] / t_new[-1]:.0f} at N={ns[-1]}', transform=ax[0].transAxes, color='k')
    bars = ax[1].bar(['original', 'new'], [fl_old(1024), fl_new(1024)], color=[C_OLD, C_NEW])
    ax[1].set_yscale('log'); ax[1].set_ylabel('GFLOP per batch'); ax[1].set_title('operation count, N = 1024')
    for b_, v in zip(bars, [fl_old(1024), fl_new(1024)]): ax[1].text(b_.get_x() + b_.get_width() / 2, v * 1.15, f'{v:.1f}', ha='center')
    ax[1].set_ylim(1, 1000)
    plt.tight_layout(); plt.savefig('figures/fig_sampling_cost.png', dpi=140); plt.close()
    print(f'  flops N=1024: original {fl_old(1024):.0f} GFLOP, new {fl_new(1024):.1f} GFLOP (x{fl_old(1024) / fl_new(1024):.0f})')

# ---------------------------------------------------------------- figure 3: reference Metropolis
LAM, F2, A, M = 1.0, 2.0, 0.1, 0.5
def energy(y):                                                           # same as Energy() in the notebook
    return (M / (2 * A) * (y - np.roll(y, -1, axis=1)) ** 2).sum(-1) + A * (LAM * (y ** 2 - F2) ** 2).sum(-1)

def metropolis(n_chains, n_sweeps, delta=1.0, seed=0):
    r = np.random.default_rng(seed)
    y = r.standard_normal((n_chains, D)) * 1.5
    Mag = np.zeros((n_sweeps, n_chains)); acc = 0; tot = 0; snaps = []; pool = []
    idx = np.arange(D)
    for s in range(n_sweeps):
        for parity in (0, 1):
            sites = idx[idx % 2 == parity]
            old = y[:, sites]; new = old + delta * (2 * r.random(old.shape) - 1)
            L, R = y[:, (sites - 1) % D], y[:, (sites + 1) % D]
            loc = lambda p: M / (2 * A) * ((p - L) ** 2 + (p - R) ** 2) + A * LAM * (p ** 2 - F2) ** 2
            ok = r.random(old.shape) < np.exp(-(loc(new) - loc(old)))
            y[:, sites] = np.where(ok, new, old); acc += ok.sum(); tot += ok.size
        Mag[s] = y.mean(1)
        if s > n_sweeps // 10 and s % 200 == 0: pool.append(y.copy().ravel())
        if s in (n_sweeps // 4, n_sweeps // 2, n_sweeps - 1): snaps.append(y[0].copy())
    return np.concatenate(pool), Mag, acc / tot, snaps

def fig_phi4():
    n_sweeps = 20000 if QUICK else 100000
    t0 = time.time(); y, Mag, acc, snaps = metropolis(8, n_sweeps); print(f'  metropolis: {n_sweeps} sweeps x 8 chains in {time.time() - t0:.0f} s, acceptance {acc:.2f}')
    burn = n_sweeps // 10; Mb = Mag[burn:]
    flips = int(np.sum(np.diff(np.sign(Mb), axis=0) != 0))
    # autocorrelation of the magnetisation (|M|-independent: use M itself)
    def acf(x, maxlag):
        x = x - x.mean(); f = np.fft.rfft(x, 2 * len(x)); a = np.fft.irfft(f * np.conj(f))[:maxlag]; return a / a[0]
    lags = 3000 if not QUICK else 1000
    A_ = np.mean([acf(Mb[:, c], lags) for c in range(Mb.shape[1])], axis=0)
    # histograms
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
    xs = np.arange(D)
    for k, (s, lab) in enumerate(zip(snaps, ['early', 'middle', 'end of run'])):
        ax[0, 0].plot(xs, s, lw=1.2, label=lab)
    ax[0, 0].axhline(np.sqrt(2), color='k', lw=0.6, ls=':'); ax[0, 0].axhline(-np.sqrt(2), color='k', lw=0.6, ls=':')
    ax[0, 0].set_xlabel('lattice site'); ax[0, 0].set_ylabel(r'$\phi$'); ax[0, 0].set_title('configurations of one chain'); ax[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc='lower center')
    ax[0, 0].set_ylim(-3, 3)
    ax[0, 1].hist(y, bins=80, density=True, color=C_NEW, alpha=0.8)
    ax[0, 1].axvline(np.sqrt(2), color='k', lw=0.6, ls=':'); ax[0, 1].axvline(-np.sqrt(2), color='k', lw=0.6, ls=':')
    ax[0, 1].set_xlabel(r'$\phi$'); ax[0, 1].set_ylabel('density'); ax[0, 1].set_title(r'field values (all sites, thinned chains; dotted: $\pm\sqrt{2}$)')
    t = np.arange(0, n_sweeps, max(1, n_sweeps // 4000))
    for c in range(2): ax[1, 0].plot(t, Mag[t, c], lw=0.7)
    ax[1, 0].set_xlabel('Metropolis sweep'); ax[1, 0].set_ylabel(r'magnetisation $\langle\phi\rangle$'); ax[1, 0].set_title('magnetisation of two chains (jumps = tunnelling between wells)')
    ax[1, 1].plot(np.arange(lags), A_, color=C_OLD); ax[1, 1].axhline(0, color='k', lw=0.5)
    tau = 1 + 2 * A_[1:int(np.argmax(A_ < 0)) if (A_ < 0).any() else lags].sum()
    ax[1, 1].set_xlabel('lag [sweeps]'); ax[1, 1].set_ylabel('autocorrelation'); ax[1, 1].set_title(rf'autocorrelation of the magnetisation ($\tau_{{int}}\gtrsim${tau:.0f} sweeps)')
    plt.suptitle(r'Reference: single-site Metropolis for the same lattice $\phi^4$ model ($D=100$, $a=0.1$, $m=0.5$, $\lambda=1$, $f^2=2$)', y=1.0, fontsize=11)
    plt.tight_layout(); plt.savefig('figures/fig_phi4_reference.png', dpi=140); plt.close()
    print(f'  <phi^2> = {np.mean(y**2):.3f}   sign changes of M: {flips}   tau_int ~ {tau:.0f} sweeps')

if __name__ == '__main__':
    print('structure'); fig_structure()
    print('sampling cost'); fig_cost()
    print('reference metropolis'); fig_phi4()
