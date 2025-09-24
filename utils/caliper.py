import numpy as np

MAX_LEN = 2048
EPS_STD = 1.0e-12
LOGW_CLIP_MIN = -20.0  # exp(-20) ≈ 2e-9

def isMonotonic(x: np.ndarray) -> bool:
    if x.size <= 1:
        return True
    tol = (x.max() - x.min()) * 1e-3
    return bool(np.all(x[:-1] + tol >= x[1:]))

class CALIPER:
    def __init__(
        self,
        d: int,
        pred_len: int,
        seed: int = 1,
        n_jobs: int = -1,
        max_count: int = 5,
        Thetas: tuple[float, ...] | list[float] = (0, 0.1, 1, 2, 4, 8, 16),
        ridge: float = 0.0,
        min_ess_factor: float = 3.0,
    ):
        self.d = int(d)
        self.p = self.d + 1
        self.pred_len = int(pred_len)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.max_count = int(max_count)

        self.thetas = np.asarray(Thetas, dtype=float)
        self.n_thetas = int(self.thetas.size)

        self.ridge = float(ridge)
        self.min_ess = float(min_ess_factor) * (self.d + 1)

        self._I = np.eye(self.p, dtype=float)
        self._thetas_pos = self.thetas[self.thetas > 0.0]
        self._theta_max = float(self._thetas_pos.max()) if self._thetas_pos.size else 0.0

        self.reset()

    def reset(self) -> None:
        self.min_err = 1.0e5
        self.is_shift = False
        self.Errs = np.zeros(self.n_thetas, dtype=float)
        self.init = True
        self.prev_len = np.inf
        self.count = 0

    def detect(self, test_data, win_st: int, win_ed: int):
        data_re, data_stamp_re = test_data.get_restart_data(win_st, win_ed)

        if ((win_ed - win_st) > MAX_LEN) or self.is_shift:
            self.is_shift = True
            return True, data_re, data_stamp_re, None

        if data_re.shape[0] < 4:
            return False, data_re, data_stamp_re, self.Errs

        Z, sigma = self._zscore_data_re(data_re)
        X_hist, Y_next, x_query, y_query = Z[:-3], Z[1:-2], Z[-2], Z[-1]

        if not self._has_sufficient_ess(X_hist, x_query):
            return False, data_re, data_stamp_re, self.Errs

        errs_step = self._batched_wls_errors(self.thetas, X_hist, Y_next, x_query, y_query, sigma)
        if errs_step.size:
            self.Errs += errs_step

        if isMonotonic(self.Errs):
            self.count += 1
            if self.count == self.max_count:
                self.is_shift = True
                return True, data_re, data_stamp_re, self.Errs
        else:
            self.count = 0

        return False, data_re, data_stamp_re, self.Errs

    @staticmethod
    def _zscore_data_re(data_re: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = data_re.mean(axis=0)
        sig = data_re.std(axis=0)
        sig = np.where(sig < EPS_STD, 1.0, sig)
        Z = (data_re - mu) / sig
        return Z, sig

    def _has_sufficient_ess(self, X_std: np.ndarray, x_std: np.ndarray) -> bool:
        if self._theta_max <= 0.0:
            return True
        dists = np.linalg.norm(X_std - x_std, axis=1)
        denom = np.median(dists) if np.median(dists) > 1e-8 else max(dists.mean(), 1e-8)
        w = np.exp(np.clip(-self._theta_max * (dists / denom), LOGW_CLIP_MIN, 0.0))
        s1 = w.sum()
        s2 = float(np.dot(w, w))
        ess = (s1 * s1) / (s2 + 1e-12)
        return bool(ess >= self.min_ess)

    def _batched_wls_errors(
        self,
        thetas: np.ndarray,
        X_std: np.ndarray,
        Y_std: np.ndarray,
        x_std: np.ndarray,
        y_std: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        n_samples, n_dim = X_std.shape
        if n_samples == 0 or thetas.size == 0:
            return np.zeros(0, dtype=float)

        dists = np.linalg.norm(X_std - x_std, axis=1)
        scale = max(dists.mean(), 1e-8)
        r = dists / scale

        W = np.exp(np.clip(-thetas[:, None] * r[None, :], LOGW_CLIP_MIN, 0.0))  # (T, n)
        X_aug = np.concatenate([X_std, np.ones((n_samples, 1), dtype=float)], axis=1)  # (n, p)

        A = np.einsum('ni,tn,nj->tij', X_aug, W, X_aug)  # (T, p, p)
        if self.ridge:
            A += self.ridge * self._I[None, :, :]
        B = np.einsum('ni,tn,nk->tik', X_aug, W, Y_std)  # (T, p, d)

        Beta = np.linalg.solve(A, B)  # (T, p, d)

        x_aug = np.concatenate([x_std, [1.0]])
        y_hat_std = np.einsum('i,tid->td', x_aug, Beta)  # (T, d)

        diff_orig = (y_hat_std - y_std[None, :]) * sigma[None, :]
        return np.sqrt(np.mean(diff_orig * diff_orig, axis=1))
