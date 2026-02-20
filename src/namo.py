"""
NAMO and NAMO-D optimizers.
"""

import math
import torch


def _nanmin(x: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(torch.isnan(x), float("inf")).min()


def _nanmax(x: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(torch.isnan(x), -float("inf")).max()

def _nanmean(x: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(x)
    count = mask.sum()
    # sum non-NaNs, divide by number of non-NaNs
    mean = torch.where(mask, x, torch.zeros_like(x)).sum() / count.clamp(min=1)
    # if everything was NaN, return NaN (matches numpy.nanmean behavior)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def zeropower_via_svd(G):
    """
    Compute the zeroth power / orthogonalization of G using exact SVD.
    This produces the exact orthogonal matrix UV^T where USV^T = G is the SVD.
    This is more accurate than Newton-Schulz but potentially slower.
    """
    assert len(G.shape) == 2
    original_dtype = G.dtype

    # Convert to float32 for better numerical stability in SVD
    G_float = G.float()

    # Perform SVD
    U, S, Vt = torch.linalg.svd(G_float, full_matrices=False)

    # Return UV^T (the orthogonal part)
    result = U @ Vt

    # Convert back to original dtype
    return result.to(original_dtype)


class NAMO(torch.optim.Optimizer):
    """
    NAMO: NAMO rescales learning rates based on accumulated gradient norms.

    Arguments:
        muon_params: The parameters to be optimized by NAMO.
        lr: The learning rate (eta). (0.02 is a good default)
        wd: The weight decay coefficient. (0.1 is a good default)
        momentum: The momentum coefficient used for standard momentum. (0.95 is a good default)
        mu2: EMA coefficient for squared gradient norm.
        adamnorm_eps: Epsilon inside v_t = sqrt(v^2 + eps).
        nesterov: Whether to use Nesterov-style momentum. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        use_exact_svd: Whether to use exact SVD for orthogonalization instead of Newton-Schulz.
        scale_coeff: coefficient in adjust_lr_for_muon (default 0.2 in the paper).
        adamw_params: The parameters to be optimized by AdamW for non-2D parameters.
        adamw_betas, adamw_eps, adamw_wd: AdamW settings for backup params.
    """

    def __init__(
        self,
        lr=1e-2,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        mu2=0.99,
        adamnorm_eps=1e-8,
        nesterov=True,
        ns_steps=5,
        use_exact_svd=False,
        scale_coeff=0.2,  # tunable coeff for adjust_lr_for_muon
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=None,
    ):
        if adamw_wd is None:
            adamw_wd = wd

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            mu2=mu2,
            adamnorm_eps=adamnorm_eps,
            nesterov=nesterov,
            ns_steps=ns_steps,
            use_exact_svd=use_exact_svd,
            scale_coeff=scale_coeff,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use Adaptive Muon, and those for which we will not
        if muon_params is not None:
            for p in muon_params:
                # Use Adaptive Muon for every parameter in muon_params which is >= 2D
                assert p.ndim == 2 or p.ndim == 4, p.ndim
                self.state[p]["use_muon"] = True
        if adamw_params is not None:
            for p in adamw_params:
                # Do not use Adaptive Muon for parameters in adamw_params
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape, scale_coeff):
        """
        Scale lr by sqrt(max(A,B)) with a tunable coefficient.
        """
        A, B = param_shape[:2]
        adjusted_ratio = scale_coeff * math.sqrt(max(A, B))
        # adjusted_ratio = math.sqrt(max(1, A / B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ###################
            #      NAMO       #
            ###################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            mu2 = group["mu2"]
            eps_adn = group["adamnorm_eps"]
            scale_coeff = group["scale_coeff"]

            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # Initialize state
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["v_squared"] = torch.zeros((), device=g.device, dtype=torch.float32)
                    state["step"] = 0

                # Get state variables
                buf = state["momentum_buffer"]
                v_squared = state["v_squared"]

                # Gradient norm clipping (AdamNorm-style)
                gn = torch.linalg.vector_norm(g)  # 0-d tensor (float32)

                # Momentum update
                buf.lerp_(g, 1 - momentum)
                if group["nesterov"]:
                    g = g.lerp(buf, momentum)
                else:
                    g = buf
                gn_m = torch.linalg.vector_norm(g)  # 0-d tensor

                # EMA of squared grad norm
                v_squared.mul_(mu2).add_(gn * gn, alpha=(1.0 - mu2))
                state["v_squared"] = v_squared

                # Choose orthogonalization method
                if group["use_exact_svd"]:
                    u = zeropower_via_svd(g).view_as(p)
                else:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).view_as(p)

                # AdamNorm-style bias-corrected lr
                state["step"] += 1
                step_t = state["step"]
                bc1 = 1.0 - (momentum**step_t)
                bc2 = 1.0 - (mu2**step_t)

                v_t = torch.sqrt(v_squared + float(eps_adn))  # 0-d tensor
                adaptive_lr = (lr * math.sqrt(bc2) / (bc1 + 1e-12)) * (gn_m / v_t)
                adaptive_lr_clamped = adaptive_lr.clamp(max=1.0)

                # scale update with tunable coeff
                A, B = p.shape[:2]
                adjusted_lr = self.adjust_lr_for_muon(adaptive_lr, p.shape, scale_coeff)
                adjusted_lr = (adaptive_lr * (scale_coeff * math.sqrt(max(A, B)))).clamp(max=1.0)

                # apply weight decay
                p.data.mul_(1.0 - wd * adaptive_lr_clamped)

                # apply update
                u.mul_(adjusted_lr.to(dtype=u.dtype))
                p.data.add_(u, alpha=-1.0)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            adamw_lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - adamw_lr * weight_decay)
                p.data.add_(g, alpha=-adamw_lr / scale)

        return loss


class NAMO_D(torch.optim.Optimizer):
    """
    NAMO-D (column-wise): column-wise NAMO scaling for Muon-style orthogonal updates.

    For a 2D weight W (m x n), maintain:
      - momentum buffer M_t (same shape as grad)
      - V_t in R^n = EMA of column-wise grad-norm^2

    Column-wise scale:
      col_scale[j] = ||M_t[:,j]|| / sqrt(V_t[j] + eps)

    This corresponds to right-multiplying by diag(col_scale): update ~ O * diag(col_lr_up)
    and applying column-wise weight decay.
    """

    def __init__(
        self,
        lr=1e-2,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        mu2=0.99,
        adamnorm_eps=1e-8,
        nesterov=True,
        ns_steps=5,
        use_exact_svd=False,
        scale_coeff=0.2,
        col_state_clamp_c=0.75,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=None,
    ):
        if adamw_wd is None:
            adamw_wd = wd

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            mu2=mu2,
            adamnorm_eps=adamnorm_eps,
            nesterov=nesterov,
            ns_steps=ns_steps,
            use_exact_svd=use_exact_svd,
            scale_coeff=scale_coeff,
            col_state_clamp_c=col_state_clamp_c,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = muon_params + adamw_params
        super().__init__(params, defaults)

        for p in muon_params:
            assert p.ndim in (2, 4), p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def _col_norms(x_2d: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(x_2d, dim=0)  # [n]

    def adjust_lr_for_muon(self, lr: float, param_shape, scale_coeff: float) -> float:
        A, B = param_shape[:2]
        return lr * (scale_coeff * math.sqrt(max(A, B)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            muon_params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            mu2 = group["mu2"]
            eps_adn = group["adamnorm_eps"]
            scale_coeff = group["scale_coeff"]
            c_clamp = group.get("col_state_clamp_c", 0.0)

            # ---- Muon params (column-wise AdamNorm scaling) ----
            for p in muon_params:
                g = p.grad
                if g is None:
                    continue

                # flatten to 2D: [m, n]
                g2 = g.reshape(g.size(0), -1) if g.ndim > 2 else g
                m, n = g2.shape

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g2)
                    state["V"] = torch.zeros(n, device=g2.device, dtype=torch.float32)  # NOTE: [n]
                    state["step"] = 0

                buf = state["momentum_buffer"]  # [m, n]
                V = state["V"]  # float32 [n]

                # update V with column-wise grad norms
                nc_G = self._col_norms(g2).to(dtype=V.dtype)  # float32 [n]
                V.mul_(mu2).addcmul_(nc_G, nc_G, value=(1.0 - mu2))

                # momentum update: buf = mu * buf + (1-mu) * g2
                buf.lerp_(g2, 1.0 - momentum)

                # choose M for update (Nesterov or not)
                m_for_update = g2.lerp(buf, momentum) if group["nesterov"] else buf

                # orthogonal direction O = Orth(M)
                if group["use_exact_svd"]:
                    O2 = zeropower_via_svd(m_for_update)
                else:
                    O2 = zeropower_via_newtonschulz5(m_for_update, steps=group["ns_steps"])

                # bias-correction scalars (Adam-style)
                state["step"] += 1
                t = state["step"]
                bc1 = 1.0 - (momentum**t)
                bc2 = 1.0 - (mu2**t)
                base_lr = lr * math.sqrt(bc2) / (bc1 + 1e-12)

                # per-col scale: ||M_col|| / sqrt(V + eps)
                nc_M = self._col_norms(m_for_update).to(dtype=V.dtype)  # [n]
                col_scale = nc_M * torch.rsqrt(V + float(eps_adn))  # [n]

                # clamp toward the average
                if 0.0 < float(c_clamp) <= 1.0:
                    c = float(c_clamp)
                    mu = _nanmean(col_scale)  # scalar tensor
                    # lower bound = mean * c
                    floor = torch.where(
                        torch.isfinite(mu) & (mu > 0),
                        mu * c,
                        col_scale.new_zeros(())
                    )   
                    # upper bound = mean / c
                    ceil = torch.where(
                        torch.isfinite(mu) & (mu > 0),
                        mu / c,
                        col_scale.new_full((), float("inf"))
                    )
                    torch.clamp(col_scale, min=floor, max=ceil, out=col_scale)
                    
                # weight decay uses UN-SHAPED per-col lr
                col_lr_wd = (col_scale * base_lr).clamp(max=1.0)
                decay = col_lr_wd.mul(-wd).add_(1.0).clamp_min_(0.0)

                # apply per-column decay to parameters: multiply each column j by decay[j]
                p2 = p.data.reshape(p.data.size(0), -1) if p.data.ndim > 2 else p.data
                p2.mul_(decay.to(dtype=p2.dtype).unsqueeze(0))

                # # weight decay using constant base_lr (scalar)
                # decay = max(1.0 - wd * base_lr, 0.0)   # float
                # # p2 = p.data.reshape(p.data.size(0), -1) if p.data.ndim > 2 else p.data
                # p2 = p.data.view(p.data.size(0), -1) if p.data.ndim > 2 else p.data
                # p2.mul_(decay)  # scalar multiply, broadcasts automatically

                # update uses SHAPED per-col lr
                shape_lr = self.adjust_lr_for_muon(base_lr, p.shape, scale_coeff)
                col_lr_up = (col_scale * shape_lr).clamp(max=1.0)
                p2.addcmul_(O2.to(dtype=p2.dtype), col_lr_up.to(dtype=p2.dtype).unsqueeze(0), value=-1.0)

            # ---- AdamW backup params ----
            backup_params = [p for p in group["params"] if not self.state[p].get("use_muon", False)]
            adamw_lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in backup_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                m1 = state["moment1"]
                m2 = state["moment2"]

                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)

                gh = m1 / (eps + m2.sqrt())
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                scale = bc1 / (bc2**0.5)

                p.data.mul_(1 - adamw_lr * weight_decay)
                p.data.add_(gh, alpha=-adamw_lr / scale)

        return loss
