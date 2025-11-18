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
        """
        Update mixture weights pi | c using Dirichlet conjugacy.

        Prior: pi ~ Dir(alpha/K, ..., alpha/K)
        Posterior: pi | c ~ Dir(alpha/K + n_1, ..., alpha/K + n_K)
        """
        # self.alpha is a scalar concentration parameter
        alpha = float(self.alpha)

        # base concentration per component
        alpha_k = alpha / float(self.K)

        # counts of each cluster (ensure 1-D float array of length K)
        counts = np.bincount(np.asarray(self.c, dtype=int), minlength=self.K).astype(float)

        # Dirichlet parameter vector (shape (K,))
        alpha_vec = alpha_k + counts

        # sample new mixture weights
        # works whether self.rng is np.random.Generator or RandomState
        self.pi = self.rng.dirichlet(alpha_vec)
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

    def _canonicalize_labels(self):
        """
        Enforce a deterministic ordering and sign convention on the components.

        Assumes:
        self.G      : shape (p, K)
        self.tau2   : shape (K,)
        self.pi     : shape (K,)
        self.c      : shape (n,) cluster labels in {0,...,K-1}
        self.z      : shape (n,) latent projections
        """
        K = self.G.shape[1]

        # 1) Order components by decreasing tau^2
        order = np.argsort(self.tau2)[::-1]   # largest tau^2 first

        # reorder parameters
        self.G    = self.G[:, order]
        self.tau2 = self.tau2[order]
        self.pi   = self.pi[order]

        # relabel cluster assignments accordingly
        # old label -> new label map
        inv_map = np.zeros(K, dtype=int)
        inv_map[order] = np.arange(K)
        self.c = inv_map[self.c]
    def _canonicalize_signs(self):
        """
        Make each direction g_k have a deterministic sign:
        ensure the entry with largest absolute value is positive.
        """
        p, K = self.G.shape
        for k in range(K):
            gk = self.G[:, k]
            j = np.argmax(np.abs(gk))  # index of largest-magnitude coordinate
            if gk[j] < 0:
                # flip sign of direction
                self.G[:, k] = -gk
                # and flip latent z_i for all points in this cluster
                mask = (self.c == k)
                self.z[mask] = -self.z[mask]
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
            if it%20==0:
                print("Iteration ",it)
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
        self._canonicalize_labels()
        self._canonicalize_signs()
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

class EigenGP_DPPM_MCMC(DPPM_MCMC):
    """
    DPPM variant where each direction g_k(t) is represented as a Gaussian process
    in an eigenfunction basis Phi with diagonal eigenvalues.

    Model:
        X_i(t) = Z_i g_{C_i}(t) + epsilon_i(t),
        g_k(t) = sum_{j=1}^J b_{jk} phi_j(t),  b_{·k} ~ N(0, diag(eigenvalues)).

    In matrix form, if Phi is (m x J) (grid x basis), and B is (J x K),
    then G = Phi @ B is (m x K) and plays the same role as in DPPM_MCMC.
    """

    def __init__(
        self,
        X,
        Phi,
        eigenvalues,
        K,
        alpha=1.0,
        a_tau=.0001,
        b_tau=.0001,
        a_sigma=.0001,
        b_sigma=.001,
        sampler_type="uncollapsed",
        random_state=None,
    ):
        """
        Parameters
        ----------
        X : array, shape (n, m)
            Observed functional data evaluated on a common grid of size m.
        Phi : array, shape (m, J)
            Basis matrix of eigenfunctions evaluated on the grid.
        eigenvalues : array, shape (J,)
            Eigenvalues of the GP covariance associated with the basis Phi.
        K : int
            Max number of mixture components (finite truncation).
        Other arguments are as in DPPM_MCMC.
        """
        self.Phi = np.asarray(Phi)
        self.eigenvalues = np.asarray(eigenvalues)
        if self.Phi.ndim != 2:
            raise ValueError("Phi must be a 2D array (m x J).")
        if self.eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must be a 1D array.")
        if self.Phi.shape[1] != self.eigenvalues.shape[0]:
            raise ValueError("Phi.shape[1] must match len(eigenvalues).")

        # Precompute Phi^T Phi (J x J) for the Gaussian regression
        self.S_Phi = self.Phi.T @ self.Phi  # (J x J)

        # Call parent constructor to set up z, c, tau2, sigma2, etc.
        # We ignore the G_update argument here because we override _update_G.
        super().__init__(
            X=X,
            K=K,
            alpha=alpha,
            a_tau=a_tau,
            b_tau=b_tau,
            a_sigma=a_sigma,
            b_sigma=b_sigma,
            sampler_type=sampler_type,
            G_update="vmf_geomstats",
            random_state=random_state,
        )

        # Replace the direct G initialization with an eigenfunction-based one.
        J = self.Phi.shape[1]
        # B has prior N(0, diag(eigenvalues)) on each column
        # We sample from that prior here as an initialization.
        std = np.sqrt(self.eigenvalues)
        self.B = self.rng.normal(size=(J, K)) * std[:, None]

        # Build G from B and orthonormalize columns
        self._update_G_from_B()

    # ------------------------------------------------------------------
    # Internal helpers for eigenfunction-GP representation
    # ------------------------------------------------------------------
    def _update_G_from_B(self):
        """
        Construct G = Phi @ B and orthonormalize its columns so that
        it can be used with the existing likelihood code.
        """
        G = self.Phi @ self.B  # shape: (m, K)
        # Orthonormalize columns via QR
        Q, _ = np.linalg.qr(G)
        self.G = Q[:, : self.K]

    # We override the generic G update to use a GP/B update instead.
    def _update_G(self):
        self._update_B_gp()
        self._update_G_from_B()

    def _update_B_gp(self):
        """
        Gibbs update for the eigenfunction coefficients B.

        For each component k, we solve a Gaussian linear regression problem:
            X_i ≈ z_i Phi b_k + noise,  b_k ~ N(0, diag(eigenvalues)).
        """
        X = self.X                      # (n x m)
        Phi = self.Phi                 # (m x J)
        S_Phi = self.S_Phi             # (J x J)
        eigenvalues = self.eigenvalues # (J,)
        Lambda_inv = 1.0 / eigenvalues

        sigma2 = self.sigma2
        n, m = X.shape
        J = Phi.shape[1]

        for k in range(self.K):
            idx = np.where(self.c == k)[0]
            if idx.size == 0:
                # No data assigned to this component: draw from prior
                eta = self.rng.normal(size=J)
                self.B[:, k] = np.sqrt(eigenvalues) * eta
                continue

            z_k = self.z[idx]  # latent projections for this component

            # Sufficient statistics for Gaussian regression:
            #   sum_i z_i^2   and   sum_i z_i Phi^T X_i
            sum_z2 = float(np.dot(z_k, z_k))

            # S = sum_i z_i * Phi^T X_i  (J-dimensional)
            S = np.zeros(J)
            for i in idx:
                S += self.z[i] * (Phi.T @ X[i])

            # Posterior precision: Lambda^{-1} + (sum z_i^2 / sigma2) * Phi^T Phi
            Precision = np.diag(Lambda_inv) + (sum_z2 / sigma2) * S_Phi  # (J x J)

            # Cholesky factor of the precision matrix
            L = np.linalg.cholesky(Precision)

            # mean = Precision^{-1} (S / sigma2) via two triangular solves
            rhs = S / sigma2
            y = np.linalg.solve(L, rhs)
            mean = np.linalg.solve(L.T, y)

            # Sample from N(mean, Precision^{-1}) using the same Cholesky
            eta = self.rng.normal(size=J)
            delta = np.linalg.solve(L.T, eta)
            self.B[:, k] = mean + delta


    def _canonicalize_labels(self):
        """
        Enforce a deterministic ordering and sign convention on the components.

        Assumes:
        self.G      : shape (p, K)
        self.tau2   : shape (K,)
        self.pi     : shape (K,)
        self.c      : shape (n,) cluster labels in {0,...,K-1}
        self.z      : shape (n,) latent projections
        """
        K = self.G.shape[1]

        # 1) Order components by decreasing tau^2
        order = np.argsort(self.tau2)[::-1]   # largest tau^2 first

        # reorder parameters
        self.G    = self.G[:, order]
        self.tau2 = self.tau2[order]
        self.pi   = self.pi[order]

        # relabel cluster assignments accordingly
        # old label -> new label map
        inv_map = np.zeros(K, dtype=int)
        inv_map[order] = np.arange(K)
        self.c = inv_map[self.c]

    def _canonicalize_labels(self):
        """
        Enforce a deterministic ordering and sign convention on the components.

        Assumes:
        self.G      : shape (p, K)
        self.tau2   : shape (K,)
        self.pi     : shape (K,)
        self.c      : shape (n,) cluster labels in {0,...,K-1}
        self.z      : shape (n,) latent projections
        """
        K = self.G.shape[1]

        # 1) Order components by decreasing tau^2
        order = np.argsort(self.tau2)[::-1]   # largest tau^2 first

        # reorder parameters
        self.G    = self.G[:, order]
        self.tau2 = self.tau2[order]
        self.pi   = self.pi[order]

        # relabel cluster assignments accordingly
        # old label -> new label map
        inv_map = np.zeros(K, dtype=int)
        inv_map[order] = np.arange(K)
        self.c = inv_map[self.c]
    def _canonicalize_signs(self):
        """
        Make each direction g_k have a deterministic sign:
        ensure the entry with largest absolute value is positive.
        """
        p, K = self.G.shape
        for k in range(K):
            gk = self.G[:, k]
            j = np.argmax(np.abs(gk))  # index of largest-magnitude coordinate
            if gk[j] < 0:
                # flip sign of direction
                self.G[:, k] = -gk
                # and flip latent z_i for all points in this cluster
                mask = (self.c == k)
                self.z[mask] = -self.z[mask]

class DPPMRegressionMCMC:
    """
    Convenience wrapper that uses DPPM_MCMC as a prior model on
    multi-task linear regression coefficients.

    We assume a setting with:
        Y: (n_samples, n_tasks)
        X: (n_samples, p_features)

    For each task t, we fit an initial OLS estimate of the coefficient
    vector beta_t, stack them into a matrix B_ols of shape (p, n_tasks),
    and run a Euclidean DPPM on the rows of B_ols^T (each task is one
    "observation" in p dimensions).

    This is a two-stage / empirical-Bayes style construction that treats
    the OLS estimates as noisy observations from the underlying DPPM
    prior on beta_t. It is useful pedagogically to illustrate how the
    Euclidean DPPM can act as a prior over regression coefficients.

    Parameters
    ----------
    X : array, shape (n, p)
        Design matrix shared across tasks.
    Y : array, shape (n, T)
        Matrix of responses for T tasks.
    K : int
        Truncation level / maximum number of projection axes.
    alpha, a_tau, b_tau, a_sigma, b_sigma, sampler_type, G_update,
    step_size_G, random_state :
        Passed through to the underlying DPPM_MCMC instance.

    Attributes
    ----------
    B_ols : array, shape (p, T)
        OLS estimates of the regression coefficients for each task.
    dppm : DPPM_MCMC
        Internal DPPM sampler run on B_ols.T (shape (T, p)).
    """

    def __init__(
        self,
        X,
        Y,
        K,
        alpha=1.0,
        a_tau=.0001,
        b_tau=.0001,
        a_sigma=.0001,
        b_sigma=.001,
        sampler_type="collapsed",
        G_update="vmf_geomstats",
        step_size_G=0.05,
        random_state=None,
    ):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        if self.X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, p_features).")
        if self.Y.ndim != 2:
            raise ValueError("Y must be 2D (n_samples, n_tasks).")
        n, p = self.X.shape
        n_y, T = self.Y.shape
        if n_y != n:
            raise ValueError("X and Y must have the same number of rows (samples).")

        self.n = n
        self.p = p
        self.T = T
        self.K = int(K)

        # Compute task-wise OLS estimates beta_t via least squares:
        #   beta_t = argmin ||Y[:, t] - X beta||_2^2
        B_ols = np.zeros((p, T))
        for t in range(T):
            # lstsq returns (coef, residuals, rank, singular_values)
            coef, *_ = np.linalg.lstsq(self.X, self.Y[:, t], rcond=None)
            B_ols[:, t] = coef
        self.B_ols = B_ols

        # Instantiate an internal DPPM on the OLS coefficient vectors.
        # Each "observation" is one task's coefficient vector of length p.
        X_for_dppm = B_ols.T  # shape (T, p)
        self.dppm = DPPM_MCMC(
            X=X_for_dppm,
            K=self.K,
            alpha=alpha,
            a_tau=a_tau,
            b_tau=b_tau,
            a_sigma=a_sigma,
            b_sigma=b_sigma,
            sampler_type=sampler_type,
            G_update=G_update,
            step_size_G=step_size_G,
            random_state=random_state,
        )

    def run(self, n_iter=1000, burn_in=500, thin=1, store_everything=True,G_update="vmf_geomstats"):
        """
        Run the internal DPPM_MCMC sampler and construct posterior draws
        for the regression coefficients beta_t implied by the projection model.

        Returns
        -------
        results : dict
            Dictionary with keys:
                - 'dppm_samples': raw samples dictionary returned by DPPM_MCMC.run
                - 'beta_samples': array of shape (n_kept, p, T)
                      posterior draws beta_t^{(s)} = z_t^{(s)} g_{c_t^{(s)}}.
                - 'beta_mean': array of shape (p, T)
                      posterior mean of beta_t over kept samples.
        """
        samples = self.dppm.run(
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            store_everything=store_everything
        )

        G_samps = samples["G"]          # (n_kept, p, K)
        if "z" not in samples or "c" not in samples:
            raise ValueError(
                "store_everything must be True in DPPMRegressionMCMC.run "
                "to reconstruct beta samples (need 'z' and 'c')."
            )
        z_samps = samples["z"]          # (n_kept, T)
        c_samps = samples["c"]          # (n_kept, T)

        n_kept = G_samps.shape[0]
        p = self.p
        T = self.T
        K = self.K

        beta_samps = np.zeros((n_kept, p, T))
        for s in range(n_kept):
            G = G_samps[s]   # (p, K)
            z = z_samps[s]   # (T,)
            c = c_samps[s]   # (T,)
            for t in range(T):
                k = int(c[t])
                beta_samps[s, :, t] = z[t] * G[:, k]

        beta_mean = beta_samps.mean(axis=0)

        return {
            "dppm_samples": samples,
            "beta_samples": beta_samps,
            "beta_mean": beta_mean,
        }

    def predict(self, X_new, beta_mean=None):
        """
        Compute predictions for each task using a matrix of coefficients.

        Parameters
        ----------
        X_new : array, shape (n_new, p)
            New design matrix.
        beta_mean : array, shape (p, T), optional
            Coefficient matrix to use for prediction. If None, uses the
            last computed posterior mean from run().

        Returns
        -------
        Y_pred : array, shape (n_new, T)
        """
        X_new = np.asarray(X_new)
        if X_new.ndim != 2:
            raise ValueError("X_new must be 2D (n_new, p).")
        if X_new.shape[1] != self.p:
            raise ValueError("X_new has wrong number of columns.")

        if beta_mean is None:
            raise ValueError(
                "beta_mean is None. Call run() first and pass the returned "
                "beta_mean, or store it on the instance."
            )

        return X_new @ beta_mean
