"""
density_comparison.py - how well can an autoregressive mixture model reproduce the lattice phi^4 density,
compared with unimodal models?   NumPy + matplotlib only (hand-written backprop), no TensorFlow.

Target samples: single-site Metropolis for the same action as in the notebooks (D=100, a=0.1, m=0.5, lambda=1, f^2=2).
Models, all fitted by maximum likelihood on the same training chains:
  * multivariate Gaussian   - the best unimodal model (sample mean and covariance)
  * AR, 1 Gaussian per site - MADE-masked network, p(phi_i | phi_<i) = N(mu_i, sigma_i)
  * AR, 2 Gaussians per site- MADE-masked network, mixture of two Gaussians per site (as in the notebook)

NOTE: this is a self-contained NumPy illustration trained by maximum likelihood on MCMC samples. The notebooks train the
same architecture with the reverse KL against the action (no samples needed), so absolute numbers will differ.

    python density_comparison.py            # ~10 min
    python density_comparison.py --quick    # ~2 min
"""
import sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUICK = '--quick' in sys.argv
D, LAM, F2, A, M = 100, 1.0, 2.0, 0.1, 0.5
plt.rcParams.update({'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.minor.visible': True, 'ytick.minor.visible': True, 'font.size': 10})

# ------------------------------------------------------------------ target samples (Metropolis)
def metropolis(n_chains, n_sweeps, thin, burn, delta=1.0, seed=0):
    r = np.random.default_rng(seed); y = r.standard_normal((n_chains, D)) * 1.5; idx = np.arange(D); out = []
    for s in range(n_sweeps):
        for par in (0, 1):
            st = idx[idx % 2 == par]; old = y[:, st]; new = old + delta * (2 * r.random(old.shape) - 1)
            L, R = y[:, (st - 1) % D], y[:, (st + 1) % D]
            loc = lambda p: M / (2 * A) * ((p - L) ** 2 + (p - R) ** 2) + A * LAM * (p ** 2 - F2) ** 2
            ok = r.random(old.shape) < np.exp(-(loc(new) - loc(old))); y[:, st] = np.where(ok, new, old)
        if s >= burn and (s - burn) % thin == 0: out.append(y.copy())
    return np.stack(out, 1)                                                         # [chain, time, D]

# ------------------------------------------------------------------ MADE with K Gaussians per site, hand-written backprop
sig = lambda z: 1 / (1 + np.exp(-z))
sp = lambda z: np.logaddexp(0, z)

class ARMixture:
    def __init__(self, K, H=192, seed=0):
        r = np.random.default_rng(seed); self.K, self.H = K, H; self.P = 3 * K - 1
        m = 1 + np.arange(H) % (D - 1)
        self.m1 = (m[None, :] >= np.arange(1, D + 1)[:, None]).astype(np.float64)
        self.m2 = np.broadcast_to(np.arange(1, D + 1)[None, :, None] > m[:, None, None], (H, D, self.P)).astype(np.float64)
        self.p = {'W1': r.standard_normal((D, H)) * np.sqrt(2 / (D + H)) * 2, 'b1': np.zeros(H),
                  'W2': r.standard_normal((H, D, self.P)) * 0.02, 'b2': np.zeros((D, self.P))}
        b2 = self.p['b2']
        if K == 2: b2[:, 0], b2[:, 1] = 1.0, -1.0          # initial means +-1, break the symmetry
        b2[:, K:2 * K] = 0.0                                # sigma = softplus(0) ~ 0.7
        self.adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.p.items()}; self.t = 0
    def forward(self, x):
        p = self.p; hp = (0.5 * x) @ (p['W1'] * self.m1) + p['b1']; h = np.maximum(hp, 0)
        o = np.einsum('nh,hdp->ndp', h, p['W2'] * self.m2) + p['b2']; return hp, h, o
    def unpack(self, o):
        K = self.K; mu = o[..., :K]; s = sp(o[..., K:2 * K]) + 1e-2
        w = np.ones_like(mu) if K == 1 else np.stack([sig(o[..., 2 * K]), 1 - sig(o[..., 2 * K])], -1)
        return mu, s, w
    def logp_sites(self, x, o):
        mu, s, w = self.unpack(o); lg = np.log(w) - 0.5 * ((x[..., None] - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
        mx = lg.max(-1, keepdims=True); return (mx + np.log(np.exp(lg - mx).sum(-1, keepdims=True)))[..., 0], lg - mx
    def nll(self, x): _, _, o = self.forward(x); return -self.logp_sites(x, o)[0].sum(1)
    def step(self, x, lr):
        p = self.p; hp, h, o = self.forward(x); K = self.K
        lp, lg = self.logp_sites(x, o); mu, s, w = self.unpack(o)
        r = np.exp(lg); r /= r.sum(-1, keepdims=True); B = x.shape[0]
        g = np.zeros_like(o); d = x[..., None] - mu
        g[..., :K] = -r * d / s ** 2
        g[..., K:2 * K] = -r * (d ** 2 / s ** 3 - 1 / s) * sig(o[..., K:2 * K])
        if K == 2: g[..., 4] = -(r[..., 0] - w[..., 0])
        g /= B
        gr = {'b2': g.sum(0), 'W2': np.einsum('nh,ndp->hdp', h, g) * self.m2}
        dh = np.einsum('ndp,hdp->nh', g, p['W2'] * self.m2) * (hp > 0)
        gr['b1'] = dh.sum(0); gr['W1'] = (0.5 * x).T @ dh * self.m1
        self.t += 1
        for k in p:
            m_, v_ = self.adam[k]; m_[:] = 0.9 * m_ + 0.1 * gr[k]; v_[:] = 0.999 * v_ + 0.001 * gr[k] ** 2
            p[k] -= lr * (m_ / (1 - 0.9 ** self.t)) / (np.sqrt(v_ / (1 - 0.999 ** self.t)) + 1e-8)
        return -lp.sum(1).mean()
    def sample(self, n, seed=1):                                                    # ancestral sampling, incremental hidden layer
        r = np.random.default_rng(seed); p = self.p; W1m, W2m = p['W1'] * self.m1, p['W2'] * self.m2
        hpre = np.tile(p['b1'], (n, 1)); x = np.zeros((n, D))
        for i in range(D):
            o = np.maximum(hpre, 0) @ W2m[:, i, :] + p['b2'][i]; mu, s, w = self.unpack(o[:, None, :]); mu, s, w = mu[:, 0], s[:, 0], w[:, 0]
            k = (r.random(n) > w[:, 0]).astype(int) if self.K == 2 else np.zeros(n, int)
            x[:, i] = mu[np.arange(n), k] + s[np.arange(n), k] * r.standard_normal(n); hpre += 0.5 * x[:, i][:, None] * W1m[i][None, :]
        return x

def fit(K, train, steps, bs=256, lr0=3e-3):
    net = ARMixture(K); r = np.random.default_rng(5); t0 = time.time()
    for it in range(steps):
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * it / steps)) + 1e-5
        l = net.step(train[r.integers(0, len(train), bs)], lr)
        if it % 500 == 0 or it == steps - 1: print(f'   K={K} step {it:5d}  train NLL {l:8.2f}  ({time.time() - t0:.0f}s)', flush=True)
    return net

if __name__ == '__main__':
    nch, nsw, thin = (32, 6000, 60) if QUICK else (64, 20000, 80)
    t0 = time.time(); X = metropolis(nch, nsw, thin, burn=1000); print(f'metropolis {time.time() - t0:.0f}s, samples {X.shape}', flush=True)
    ntr = int(0.75 * nch); train = X[:ntr].reshape(-1, D); test = X[ntr:].reshape(-1, D)         # test = unseen chains
    steps = 800 if QUICK else 3500
    nets = {1: fit(1, train, steps), 2: fit(2, train, steps)}
    mu, C = train.mean(0), np.cov(train.T) + 1e-6 * np.eye(D); Lc = np.linalg.cholesky(C)
    sgn, ld = np.linalg.slogdet(C); Ci = np.linalg.inv(C)
    nll_g = lambda x: 0.5 * np.einsum('ni,ij,nj->n', x - mu, Ci, x - mu) + 0.5 * ld + 0.5 * D * np.log(2 * np.pi)
    nS = 20000
    samples = {'Metropolis (target)': X.reshape(-1, D),                                   # all chains: best estimate of the target
               'Gaussian (multivariate)': mu + np.random.default_rng(2).standard_normal((nS, D)) @ Lc.T,
               'AR, 1 Gaussian / site': nets[1].sample(nS), 'AR, 2 Gaussians / site': nets[2].sample(nS)}
    nll = {'Gaussian (multivariate)': nll_g(test).mean(), 'AR, 1 Gaussian / site': nets[1].nll(test).mean(), 'AR, 2 Gaussians / site': nets[2].nll(test).mean()}
    for k, v in nll.items(): print(f'  held-out NLL per configuration  {k:26s} {v:9.2f}')
    cols = {'Metropolis (target)': 'k', 'Gaussian (multivariate)': 'tab:blue', 'AR, 1 Gaussian / site': 'tab:green', 'AR, 2 Gaussians / site': 'tab:red'}
    lab = {k: (k + (f'   NLL {nll[k]:.1f}' if k in nll else '')) for k in cols}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, s in samples.items():
        ax[0].hist(s.ravel(), bins=100, range=(-3.5, 3.5), density=True, histtype='step', color=cols[k], lw=1.8, label=lab[k])
        ax[1].hist(s.mean(1), bins=70, range=(-2, 2), density=True, histtype='step', color=cols[k], lw=1.8)
        c = [np.mean(s[:, :D - 40] * s[:, r_:D - 40 + r_]) for r_ in range(0, 40)]        # <phi_i phi_{i+r}>, averaged over sites i
        ax[2].plot(range(0, 40), np.array(c) / c[0], color=cols[k], lw=1.8, marker='o' if k == 'Metropolis (target)' else None, ms=3, ls='-' if k != 'Metropolis (target)' else ':')
    ax[0].set_xlabel(r'$\phi$ (single site)'); ax[0].set_ylabel('density'); ax[0].legend(frameon=False, fontsize=8, loc='upper center')
    ax[0].set_title('field value at one site'); ax[0].set_ylim(0, 0.62)
    ax[1].set_xlabel(r'magnetisation $\langle\phi\rangle$'); ax[1].set_ylabel('density'); ax[1].set_title('whole-configuration quantity')
    ax[2].set_xlabel('distance r [sites]'); ax[2].set_ylabel(r'$\langle\phi_i\phi_{i+r}\rangle / \langle\phi_i^2\rangle$'); ax[2].set_title('two-point correlation')
    plt.tight_layout(); plt.savefig('figures/fig_density_comparison.png', dpi=140); plt.close()
    print('saved figures/fig_density_comparison.png')
