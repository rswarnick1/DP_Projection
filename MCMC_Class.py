"""
dppm_mcmc.py

MCMC implementation for a Dirichlet Process Projection Model (finite truncation)
with two sampler variants (uncollapsed and collapsed) and two types of G updates:

- "mh": random-walk Metropolis-Hastings on the Stiefel manifold (via QR).
- "vmf_geomstats": approximate vMF-based Gibbs-style update using geomstats'
  von Mises–Fisher sampler on the hypersphere.

This is a pedagogical implementation, not optimized for production.
"""

import numpy as np

# Optional import for vMF updates using geomstats
_HAS_GEOMSTATS = False
try:
    from geomstats.geometry.hypersphere import Hypersphere
    import geomstats.backend as gs  # noqa: F401
    _HAS_GEOMSTATS = True
except ImportError:  # pragma: no cover
    Hypersphere = None


class DPPM_MCMC:
    def __init__(
        self,
        X,
        K,
        alpha=1.0,
        a_tau=2.0,
        b_tau=2.0,
        a_sigma=2.0,
        b_sigma=2.0,
        sampler_type="uncollapsed",
        G_update="mh",
        step_size_G=0.05,
        random_state=None,
    ):
        """
        Parameters
        ----------
        X : array, shape (n, p)
            Data matrix.
        K : int
            Truncation level for number of components / axes.
        alpha : float
            Dirichlet prior concentration for pi.
        a_tau, b_tau : float
            Inverse-Gamma prior hyperparameters for tau2_k.
        a_sigma, b_sigma : float
            Inverse-Gamma prior hyperparameters for sigma2.
        sampler_type : {"uncollapsed", "collapsed"}
            Which sampler variant to use.
        G_update : {"mh", "vmf_geomstats"}
            How to update the projection matrix G:
            - "mh": random-walk Metropolis-Hastings on Stiefel manifold.
            - "vmf_geomstats": vMF-based approximate Gibbs update using geomstats.
        step_size_G : float
            Standard deviation of random-walk proposals for G (for "mh").
        random_state : int or None
            Seed for reproducibility.
        """
        self.X = np.asarray(X)
        self.n, self.p = self.X.shape
        self.K = int(K)
        self.alpha = alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.sampler_type = sampler_type
        self.G_update = G_update
        self.step_size_G = step_size_G

        self.rng = np.random.default_rng(random_state)

        # Optional geomstats sphere for vMF updates
        self._sphere = None
        if self.G_update == "vmf_geomstats":
            if not _HAS_GEOMSTATS:
                raise ImportError(
                    "geomstats is required for G_update='vmf_geomstats'.\n"
                    "Install it via `pip install geomstats` or choose G_update='mh'."
                )
            # Hypersphere of dimension p-1 (embedded in R^p)
            self._sphere = Hypersphere(dim=self.p - 1, equip=False)

        # Initialize parameters
        self._init_params()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_params(self):
        n, p, K = self.n, self.p, self.K

        # Random orthonormal matrix G of shape (p, K)
        A = self.rng.normal(size=(p, K))
        self.G, _ = np.linalg.qr(A)  # orthonormal columns

        # Initialize cluster assignments
        self.c = self.rng.integers(low=0, high=K, size=self.n)

        # Initialize z
        self.z = self.rng.normal(size=self.n)

        # Initialize tau2_k and sigma2
        self.tau2 = np.ones(K)
        self.sigma2 = 1.0

        # Initialize pi
        self.pi = np.ones(K) / K

    # ------------------------------------------------------------------
    # Utility: inverse-gamma sampling
    # ------------------------------------------------------------------
    def _sample_inv_gamma(self, shape, scale):
        # numpy uses Gamma(k, theta) for shape k and scale theta (mean = k * theta)
        # If X ~ InvGamma(a, b) with pdf ~ x^{-a-1} exp(-b/x),
        # then 1/X ~ Gamma(a, 1/b).
        gam = self.rng.gamma(shape, 1.0 / scale)
        return 1.0 / gam

    # ------------------------------------------------------------------
    # Log-likelihood utilities
    # ------------------------------------------------------------------
    def _log_likelihood(self, G, tau2, sigma2, c, z):
        """Compute log-likelihood sum_i log p(x_i | parameters, c_i, z_i)."""
        X = self.X
        n, p = X.shape
        ll = 0.0
        inv_sigma2 = 1.0 / sigma2
        const = -0.5 * p * np.log(2.0 * np.pi * sigma2)
        for i in range(n):
            k = c[i]
            mu = z[i] * G[:, k]
            residual = X[i] - mu
            ll += const - 0.5 * inv_sigma2 * np.dot(residual, residual)
        return ll

    def _log_marginal_x_given_c(self, i, k, G, tau2_k, sigma2):
        """
        Log p(x_i | c_i = k, G, tau2_k, sigma2) for collapsed c-update.

        Using: x ~ N(0, sigma2 I + tau2 g g^T), where ||g|| = 1.
        """
        x = self.X[i]
        p = self.p
        g = G[:, k]

        # Precompute norms and projections
        x_norm2 = np.dot(x, x)
        gx = np.dot(g, x)

        # Sigma = sigma2 I + tau2 g g^T
        # det(Sigma) = sigma2^{p-1} (sigma2 + tau2)
        # Sigma^{-1} = (1/sigma2) I - (tau2/(sigma2*(sigma2+tau2))) g g^T
        logdet = (p - 1) * np.log(sigma2) + np.log(sigma2 + tau2_k)
        inv_quad = (x_norm2 / sigma2) - (tau2_k / (sigma2 * (sigma2 + tau2_k))) * (gx ** 2)

        logp = -0.5 * (p * np.log(2.0 * np.pi) + logdet + inv_quad)
        return logp

    # ------------------------------------------------------------------
    # Single-step updates
    # ------------------------------------------------------------------
    def _update_z(self):
        """
        Gibbs update for z_i.

        Posterior z_i | x_i, c_i=k, G, tau2_k, sigma2 ~ Normal(mean, var)
        where var = [||g_k||^2 / sigma2 + 1/tau2_k]^{-1}, mean = var * (g_k^T x_i / sigma2).
        Here ||g_k|| = 1, so var = [1/sigma2 + 1/tau2_k]^{-1}.
        """
        G, tau2, sigma2 = self.G, self.tau2, self.sigma2
        X = self.X
        n = self.n

        for i in range(n):
            k = self.c[i]
            gk = G[:, k]
            tau2_k = tau2[k]
            # posterior variance
            var = 1.0 / (1.0 / sigma2 + 1.0 / tau2_k)
            # posterior mean
            mean = var * (np.dot(gk, X[i]) / sigma2)
            self.z[i] = self.rng.normal(mean, np.sqrt(var))

    def _update_c_uncollapsed(self):
        """Gibbs update for c_i using z (uncollapsed sampler)."""
        X = self.X
        G, tau2, sigma2 = self.G, self.tau2, self.sigma2
        pi = self.pi
        n, p, K = self.n, self.p, self.K

        inv_sigma2 = 1.0 / sigma2
        const = -0.5 * p * np.log(2.0 * np.pi * sigma2)

        for i in range(n):
            log_probs = np.empty(K)
            for k in range(K):
                mu = self.z[i] * G[:, k]
                residual = X[i] - mu
                logp_x = const - 0.5 * inv_sigma2 * np.dot(residual, residual)
                log_probs[k] = np.log(pi[k] + 1e-16) + logp_x

            # Normalize to probabilities
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            self.c[i] = self.rng.choice(K, p=probs)

    def _update_c_collapsed(self):
        """Collapsed Gibbs update for c_i integrating out z_i."""
        G, tau2, sigma2 = self.G, self.tau2, self.sigma2
        pi = self.pi
        K = self.K

        for i in range(self.n):
            log_probs = np.empty(K)
            for k in range(K):
                logp_x = self._log_marginal_x_given_c(i, k, G, tau2[k], sigma2)
                log_probs[k] = np.log(pi[k] + 1e-16) + logp_x

            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            self.c[i] = self.rng.choice(K, p=probs)

    def _update_pi(self):
        """Gibbs update for mixture weights pi (Dirichlet)."""
        alpha_k = self.alpha / self.K
        counts = np.bincount(self.c, minlength=self.K)
        self.pi = self.rng.dirichlet(alpha_k + counts)

    def _update_tau2(self):
        """Gibbs update for tau2_k (Inverse-Gamma)."""
        a0, b0 = self.a_tau, self.b_tau
        for k in range(self.K):
            idx = np.where(self.c == k)[0]
            n_k = len(idx)
            if n_k == 0:
                # no data assigned, revert to prior
                self.tau2[k] = self._sample_inv_gamma(a0, b0)
            else:
                z_k = self.z[idx]
                shape = a0 + 0.5 * n_k
                scale = b0 + 0.5 * np.dot(z_k, z_k)
                self.tau2[k] = self._sample_inv_gamma(shape, scale)

    def _update_sigma2(self):
        """Gibbs update for sigma2 (Inverse-Gamma)."""
        a0, b0 = self.a_sigma, self.b_sigma
        X, G, z, c = self.X, self.G, self.z, self.c
        n, p = X.shape

        # Sum of squared residuals
        sse = 0.0
        for i in range(n):
            k = c[i]
            mu = z[i] * G[:, k]
            residual = X[i] - mu
            sse += np.dot(residual, residual)

        shape = a0 + 0.5 * n * p
        scale = b0 + 0.5 * sse
        self.sigma2 = self._sample_inv_gamma(shape, scale)

    # ------------------------------------------------------------------
    # Proposals/updates for G
    # ------------------------------------------------------------------
    def _propose_G_mh(self):
        """
        Propose a new G by adding Gaussian noise to columns and re-orthonormalizing.
        This is a simple random-walk on the Stiefel manifold.
        """
        G_prop = self.G + self.step_size_G * self.rng.normal(size=self.G.shape)
        # Re-orthonormalize via QR
        Q, _ = np.linalg.qr(G_prop)
        return Q

    def _update_G_mh(self):
        """Metropolis-Hastings update for G using random-walk proposal."""
        G_old = self.G
        tau2, sigma2, c, z = self.tau2, self.sigma2, self.c, self.z

        # Propose
        G_new = self._propose_G_mh()

        # Compute log-likelihood ratio (prior assumed uniform on Stiefel)
        ll_old = self._log_likelihood(G_old, tau2, sigma2, c, z)
        ll_new = self._log_likelihood(G_new, tau2, sigma2, c, z)
        log_alpha = ll_new - ll_old

        if np.log(self.rng.uniform()) < log_alpha:
            self.G = G_new  # accept

    def _update_G_geomstats_vmf(self):
        """
        Approximate vMF-based update for G using geomstats.

        For each column k:
            S_k = sum_{i: c_i=k} z_i x_i
        defines a vMF-like conditional:
            p(g_k | rest) ∝ exp( (1/sigma2) g_k^T S_k )

        We:
        1. Compute S_k, mu_k, kappa_k via _compute_vmf_sufficient_stats().
        2. For each k with kappa_k > 0, sample g_k ~ vMF(kappa_k, mu_k) on S^{p-1}.
            If kappa_k == 0 (no or weak data), sample g_k roughly from the prior:
            either uniform on the sphere (geomstats) or a random normal normalized.
        3. Re-orthonormalize columns via QR to project back to the Stiefel manifold.

        This is an approximate Gibbs step (no MH accept/reject) but leverages
        the vMF structure implied by the likelihood.
        """
        if self._sphere is None:
            raise RuntimeError("Geomstats sphere not initialized for vmf updates.")

        p, K = self.p, self.K

        # 1) Compute sufficient statistics for vMF conditionals
        S, mu, kappa = self._compute_vmf_sufficient_stats()

        G_new = np.zeros((p, K))

        for k in range(K):
            if kappa[k] <= 0.0:
                # No effective data for this component: sample ~ uniform on S^{p-1}
                # Try geomstats random_uniform; fall back to normal+normalize
                try:
                    sample = self._sphere.random_uniform(n_samples=1)
                    vec = np.array(sample).reshape(-1)
                except Exception:
                    vec = self.rng.normal(size=p)
            else:
                # Non-degenerate vMF(kappa_k, mu_k)
                mu_k = mu[:, k]
                # Clamp kappa to avoid numerical issues
                kappa_k = float(np.clip(kappa[k], 1e-6, 1e6))
                sample = self._sphere.random_von_mises_fisher(
                    kappa=kappa_k, mu=mu_k, n_samples=1
                )
                vec = np.array(sample).reshape(-1)

            # Normalize to be safe
            norm_vec = np.linalg.norm(vec)
            if norm_vec < 1e-12:
                vec = self.rng.normal(size=p)
                vec = vec / np.linalg.norm(vec)
            else:
                vec = vec / norm_vec

            G_new[:, k] = vec

        # 3) Re-orthonormalize columns: project back to Stiefel manifold
        Q, _ = np.linalg.qr(G_new)
        self.G = Q

    def _update_G(self):
        """Dispatch G update according to G_update mode."""
        if self.G_update == "mh":
            self._update_G_mh()
        elif self.G_update == "vmf_geomstats":
            self._update_G_geomstats_vmf()
        else:
            raise ValueError(f"Unknown G_update mode: {self.G_update}")

    # ------------------------------------------------------------------
    # Public API: run MCMC
    # ------------------------------------------------------------------
    def step(self):
        """Perform one MCMC iteration (either uncollapsed or collapsed)."""
        if self.sampler_type == "uncollapsed":
            # Order: z -> c -> pi -> tau2 -> sigma2 -> G
            self._update_z()
            self._update_c_uncollapsed()
        elif self.sampler_type == "collapsed":
            # Order: c (collapsed) -> z -> pi -> tau2 -> sigma2 -> G
            self._update_c_collapsed()
            self._update_z()
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")

        self._update_pi()
        self._update_tau2()
        self._update_sigma2()
        self._update_G()

    def run(self, n_iter=1000, burn_in=500, thin=1, store_everything=True):
        """
        Run MCMC and collect samples.

        Parameters
        ----------
        n_iter : int
            Total number of MCMC iterations.
        burn_in : int
            Number of initial iterations to discard.
        thin : int
            Keep one sample every `thin` iterations after burn-in.
        store_everything : bool
            If True, store z and c for each kept iteration (can be large).

        Returns
        -------
        samples : dict
            Dictionary with arrays of collected samples:
            - 'G': array of shape (n_samples, p, K)
            - 'pi': array of shape (n_samples, K)
            - 'tau2': array of shape (n_samples, K)
            - 'sigma2': array of shape (n_samples,)
            - 'c': array of shape (n_samples, n) if store_everything
            - 'z': array of shape (n_samples, n) if store_everything
        """
        kept_G = []
        kept_pi = []
        kept_tau2 = []
        kept_sigma2 = []
        kept_c = []
        kept_z = []

        for it in range(n_iter):
            self.step()

            if it >= burn_in and ((it - burn_in) % thin == 0):
                kept_G.append(self.G.copy())
                kept_pi.append(self.pi.copy())
                kept_tau2.append(self.tau2.copy())
                kept_sigma2.append(self.sigma2)
                if store_everything:
                    kept_c.append(self.c.copy())
                    kept_z.append(self.z.copy())

        samples = {
            "G": np.stack(kept_G, axis=0),
            "pi": np.stack(kept_pi, axis=0),
            "tau2": np.stack(kept_tau2, axis=0),
            "sigma2": np.array(kept_sigma2),
        }
        if store_everything:
            samples["c"] = np.stack(kept_c, axis=0)
            samples["z"] = np.stack(kept_z, axis=0)

        return samples
    
        # ------------------------------------------------------------------
    # vMF sufficient statistics for G-update
    # ------------------------------------------------------------------
    def _compute_vmf_sufficient_stats(self):
        """
        Compute sufficient statistics for the vMF-like conditional of g_k.

        For each component k:
            S_k = sum_{i: c_i = k} z_i x_i
        The log-likelihood contribution is approximately:
            log p(g_k | rest) ∝ (1 / sigma2) g_k^T S_k

        This is the natural parameter of a von Mises–Fisher distribution on S^{p-1}
        with mean direction:
            mu_k = S_k / ||S_k||
        and concentration parameter:
            kappa_k = ||S_k|| / sigma2

        Returns
        -------
        S : array, shape (p, K)
            Sufficient statistics for each component.
        mu : array, shape (p, K)
            Mean direction for each component (unit vectors). Undefined columns
            (no data) are set to zeros.
        kappa : array, shape (K,)
            Concentration parameter for each component (non-negative).
        """
        X, c, z, sigma2 = self.X, self.c, self.z, self.sigma2
        p, K = self.p, self.K

        S = np.zeros((p, K))
        mu = np.zeros((p, K))
        kappa = np.zeros(K)

        for i in range(self.n):
            k = c[i]
            S[:, k] += z[i] * X[i]

        for k in range(K):
            norm_S = np.linalg.norm(S[:, k])
            if norm_S < 1e-12:
                # No or very weak data for this component: leave mu=0, kappa=0
                continue
            mu[:, k] = S[:, k] / norm_S
            kappa[k] = norm_S / sigma2

        return S, mu, kappa

