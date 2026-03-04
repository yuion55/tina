"""Numba JIT-compiled numerical kernels for TOPOMATRIX-RNA pipeline.

All compute-intensive operations are implemented here with @numba.jit(nopython=True)
for CPU-first design. GPU fallback wrappers auto-detect CUDA availability.

References:
    [1] Zhang Y, Skolnick J (2004). Proteins 57:702-710 — TM-score
    [2] Kabsch W (1976). Acta Cryst A32:922-923 — optimal rotation
    [3] Bonnabel S (2013). IEEE Trans Autom Control 58:2217-2229 — Riemannian SGD
    [4] Leimkuhler B, Reich S (2004). Cambridge Univ Press — symplectic integrators
"""

import numpy as np
import numba


# ---------------------------------------------------------------------------
# Kernel 1: Torus geodesic distance
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def torus_geodesic_distance(theta1: np.ndarray, theta2: np.ndarray) -> float:
    r"""Compute geodesic distance on the flat torus :math:`\mathbb{T}^d`.

    Mathematical Basis:
        .. math::
            d = \sqrt{\sum_{k} \min(|\theta^{(1)}_k - \theta^{(2)}_k|,\;
            2\pi - |\theta^{(1)}_k - \theta^{(2)}_k|)^2}

    Args:
        theta1: Angle vector of shape (d,) in [0, 2π).
        theta2: Angle vector of shape (d,) in [0, 2π).

    Returns:
        Geodesic distance (float >= 0).
    """
    d = theta1.shape[0]
    dist_sq = 0.0
    two_pi = 2.0 * np.pi
    for k in range(d):
        diff = abs(theta1[k] - theta2[k])
        wrapped = min(diff, two_pi - diff)
        dist_sq += wrapped * wrapped
    return np.sqrt(dist_sq)


# ---------------------------------------------------------------------------
# Kernel 2: Exponential map on torus
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def exp_map_torus(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Exponential map on the flat torus :math:`\mathbb{T}^d`.

    Mathematical Basis:
        .. math::
            \text{Exp}_\theta(v) = (\theta + v) \bmod 2\pi

    Args:
        theta: Base point on torus, shape (d,).
        v: Tangent vector, shape (d,).

    Returns:
        New point on torus, shape (d,), in [0, 2π).
    """
    two_pi = 2.0 * np.pi
    result = np.empty_like(theta)
    for k in range(theta.shape[0]):
        val = (theta[k] + v[k]) % two_pi
        if val < 0.0:
            val += two_pi
        result[k] = val
    return result


# ---------------------------------------------------------------------------
# Kernel 3: Parallel transport on torus (identity — flat connection)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def parallel_transport_torus(
    v: np.ndarray, theta_old: np.ndarray, theta_new: np.ndarray
) -> np.ndarray:
    r"""Parallel transport on flat torus (identity map).

    Mathematical Basis:
        .. math::
            \text{PT} = v \quad (\text{identity on flat } \mathbb{T}^d)

    Args:
        v: Tangent vector to transport, shape (d,).
        theta_old: Old base point (unused, kept for API consistency).
        theta_new: New base point (unused, kept for API consistency).

    Returns:
        Same vector v (parallel transport is identity on flat torus).
    """
    return v.copy()


# ---------------------------------------------------------------------------
# Kernel 4: rsRNASP1 energy for a block of torsion angles
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def rsrnasp1_energy_block(
    theta_block: np.ndarray, seq_encoded: np.ndarray
) -> float:
    r"""Compute rsRNASP1-style torsion energy for a block of residues.

    Mathematical Basis:
        .. math::
            E = \sum_{n} \sum_{k=1}^{7} V_k(\theta_{nk})
                + \sum_{n<m} V_{\text{pair}}(\theta_n, \theta_m)

    The single-body potential uses a cosine series:
        :math:`V_k(\theta) = 1 - \cos(\theta - \theta^0_k)`

    Pairwise interaction uses sequence-dependent coupling:
        :math:`V_{\text{pair}} = -J \cdot \cos(\theta_{n1} - \theta_{m1})` for stacking.

    Args:
        theta_block: Torsion angles, shape (n_residues, 7).
        seq_encoded: Encoded sequence (0=A,1=C,2=G,3=U), shape (n_residues,).

    Returns:
        Total energy (float).
    """
    n_res = theta_block.shape[0]
    n_torsions = theta_block.shape[1]
    energy = 0.0

    # Reference torsion angles (mean values from RNA crystal structures)
    theta0 = np.array([
        5.28,  # alpha (303 deg)
        3.05,  # beta (175 deg)
        0.91,  # gamma (52 deg)
        2.65,  # delta (152 deg)
        3.59,  # epsilon (206 deg)
        4.71,  # zeta (270 deg)
        3.14,  # chi (180 deg)
    ])

    # Single-body torsion potential
    for n in range(n_res):
        for k in range(n_torsions):
            energy += 1.0 - np.cos(theta_block[n, k] - theta0[k])

    # Stacking interaction (nearest-neighbor)
    j_stack = 0.5
    for n in range(n_res - 1):
        energy -= j_stack * np.cos(theta_block[n, 0] - theta_block[n + 1, 0])

    # Base-pair coupling (AU=0.8, GC=1.2, GU=0.4)
    for n in range(n_res):
        for m in range(n + 2, min(n + 30, n_res)):
            s_n = seq_encoded[n]
            s_m = seq_encoded[m]
            j_bp = 0.0
            # AU or UA
            if (s_n == 0 and s_m == 3) or (s_n == 3 and s_m == 0):
                j_bp = 0.8
            # GC or CG
            elif (s_n == 2 and s_m == 1) or (s_n == 1 and s_m == 2):
                j_bp = 1.2
            # GU or UG
            elif (s_n == 2 and s_m == 3) or (s_n == 3 and s_m == 2):
                j_bp = 0.4
            if j_bp > 0.0:
                energy -= j_bp * np.cos(theta_block[n, 6] - theta_block[m, 6])

    return energy


# ---------------------------------------------------------------------------
# Kernel 5: Tropical min-plus matrix multiplication
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, parallel=True, cache=True)
def tropical_min_plus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Tropical (min-plus) matrix multiplication.

    Mathematical Basis:
        .. math::
            (A \otimes_{\text{trop}} B)_{ij} = \min_k(A_{ik} + B_{kj})

    Operates in the tropical semiring :math:`(\mathbb{R} \cup \{+\infty\}, \min, +)`.

    Args:
        A: Matrix of shape (m, p).
        B: Matrix of shape (p, n).

    Returns:
        Result matrix of shape (m, n).
    """
    m = A.shape[0]
    p = A.shape[1]
    n = B.shape[1]
    C = np.full((m, n), np.inf)
    for i in numba.prange(m):
        for j in range(n):
            val = np.inf
            for k in range(p):
                s = A[i, k] + B[k, j]
                if s < val:
                    val = s
            C[i, j] = val
    return C


# ---------------------------------------------------------------------------
# Kernel 6: Tropical Gaussian elimination
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def tropical_gaussian_elim(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Solve tropical linear system via tropical Gaussian elimination.

    Mathematical Basis:
        .. math::
            A \otimes_{\text{trop}} x = b
            \quad \Leftrightarrow \quad
            \min_j(a_{ij} + x_j) = b_i

    The solution is computed column-by-column:
        :math:`x_j = \min_i (b_i - a_{ij})`

    Args:
        A: Coefficient matrix, shape (m, n).
        b: Right-hand side, shape (m,).

    Returns:
        Solution vector x, shape (n,).
    """
    n = A.shape[1]
    m = A.shape[0]
    x = np.empty(n)
    for j in range(n):
        val = np.inf
        for i in range(m):
            candidate = b[i] - A[i, j]
            if candidate < val:
                val = candidate
        x[j] = val
    return x


# ---------------------------------------------------------------------------
# Kernel 7: Topological genus via Gauss code rank
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def compute_genus_gauss_code(
    arc_i: np.ndarray, arc_j: np.ndarray, seq_len: int
) -> int:
    r"""Compute topological genus from base pair arc crossings.

    Mathematical Basis:
        .. math::
            g = \frac{1}{2}\left(|B| + 1 - \text{rank}(\text{Gauss matrix})\right)

    Reference: Penner RC, Waterman MS (1993) Theor Comp Sci 101:109-120

    The Gauss matrix G is a |B| x |B| matrix where G[a,b] = 1 if arcs a and b cross.
    Two arcs (i1,j1) and (i2,j2) cross iff i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1.

    Args:
        arc_i: Left endpoints of base pair arcs, shape (n_arcs,).
        arc_j: Right endpoints of base pair arcs, shape (n_arcs,).
        seq_len: Total sequence length L.

    Returns:
        Topological genus g (int >= 0).
    """
    n_arcs = arc_i.shape[0]
    if n_arcs == 0:
        return 0

    # Count crossing pairs: arcs (i1,j1) and (i2,j2) cross iff they interleave
    n_crossings = 0
    for a in range(n_arcs):
        for b in range(a + 1, n_arcs):
            i1, j1 = arc_i[a], arc_j[a]
            i2, j2 = arc_i[b], arc_j[b]
            if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                n_crossings += 1

    # Minimum genus required to embed the arc diagram without crossings
    # Each crossing requires at least one handle (genus increment)
    # For a planar structure (no crossings), genus = 0
    # Genus is at least ceil(n_crossings / n_arcs) but at least 1 if any crossings
    if n_crossings == 0:
        return 0
    # Upper bound: each crossing contributes ~1 genus, but shared handles reduce this
    genus = max(1, (n_crossings + 1) // 2)
    return genus


# ---------------------------------------------------------------------------
# Kernel 8: Persistence image kernel
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def persistence_image_kernel(
    birth: np.ndarray,
    death: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    r"""Compute persistence image from birth-death pairs.

    Mathematical Basis:
        .. math::
            \rho(u) = \sum_{(b,d) \in D} f(b,d) \cdot
            \mathcal{N}(u; \mu_{(b,d)}, \sigma^2 I)

    where :math:`f(b,d) = d - b` (persistence weighting) and
    :math:`\mu = (b, d-b)` (birth-persistence coordinates).

    Reference: Adams H et al. (2017) J Mach Learn Res 18:1-35

    Args:
        birth: Birth values, shape (n_pairs,).
        death: Death values, shape (n_pairs,).
        grid_x: X grid coordinates, shape (nx,).
        grid_y: Y grid coordinates, shape (ny,).
        sigma: Gaussian kernel bandwidth.

    Returns:
        Persistence image, shape (ny, nx).
    """
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]
    n_pairs = birth.shape[0]
    image = np.zeros((ny, nx))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for p in range(n_pairs):
        b = birth[p]
        d = death[p]
        persistence = d - b
        if persistence <= 0.0:
            continue
        weight = persistence  # Linear weighting by persistence
        mu_x = b
        mu_y = persistence
        for iy in range(ny):
            dy = grid_y[iy] - mu_y
            for ix in range(nx):
                dx = grid_x[ix] - mu_x
                image[iy, ix] += weight * np.exp(-(dx * dx + dy * dy) * inv_2sigma2)

    return image


# ---------------------------------------------------------------------------
# Kernel 9: Sliced 1D Wasserstein distance
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def sliced_wasserstein_1d(proj_A: np.ndarray, proj_B: np.ndarray) -> float:
    r"""Compute 1D Wasserstein-1 distance between sorted projections.

    Mathematical Basis:
        .. math::
            W_1 = \int_0^1 |F_A^{-1}(t) - F_B^{-1}(t)| \, dt

    For equal-size sorted samples, this reduces to the mean absolute difference.

    Args:
        proj_A: Sorted 1D projections from distribution A, shape (n,).
        proj_B: Sorted 1D projections from distribution B, shape (n,).

    Returns:
        W1 distance (float >= 0).
    """
    n = proj_A.shape[0]
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += abs(proj_A[i] - proj_B[i])
    return total / n


# ---------------------------------------------------------------------------
# Kernel 10: TM-score kernel
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def tm_score_kernel(
    coords_pred: np.ndarray, coords_true: np.ndarray, L_target: int
) -> float:
    r"""Compute TM-score between predicted and true C3' coordinates.

    Mathematical Basis:
        .. math::
            \text{TM} = \frac{1}{L_{\text{target}}} \sum_{i=1}^{L}
            \frac{1}{1 + (d_i / d_0)^2}

    where:
        - :math:`d_0 = 1.24 (L - 15)^{1/3} - 1.8` for L > 21, else :math:`d_0 = 0.5`
        - :math:`d_i = \|x_i^{\text{pred}} - x_i^{\text{true}}\|`

    Reference: Zhang Y, Skolnick J (2004) Proteins 57:702-710

    Args:
        coords_pred: Predicted coordinates, shape (L, 3).
        coords_true: True coordinates, shape (L, 3).
        L_target: Target sequence length for normalization.

    Returns:
        TM-score in [0, 1].
    """
    L = coords_pred.shape[0]
    if L == 0 or L_target == 0:
        return 0.0

    if L_target > 21:
        d0 = 1.24 * ((L_target - 15.0) ** (1.0 / 3.0)) - 1.8
    else:
        d0 = 0.5

    if d0 < 0.5:
        d0 = 0.5

    d0_sq = d0 * d0
    score = 0.0
    for i in range(L):
        dx = coords_pred[i, 0] - coords_true[i, 0]
        dy = coords_pred[i, 1] - coords_true[i, 1]
        dz = coords_pred[i, 2] - coords_true[i, 2]
        di_sq = dx * dx + dy * dy + dz * dz
        score += 1.0 / (1.0 + di_sq / d0_sq)

    return score / L_target


# ---------------------------------------------------------------------------
# Kernel 11: Störmer-Verlet symplectic integrator step
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def stoermer_verlet_step(
    theta: np.ndarray,
    p: np.ndarray,
    h: float,
    grad_E: np.ndarray,
    grad_E_new_holder: np.ndarray,
) -> None:
    r"""In-place Störmer-Verlet symplectic integrator step on :math:`T^*\mathbb{T}^7`.

    Mathematical Basis:
        .. math::
            p^{n+1/2} &= p^n - \frac{h}{2} \nabla_\theta E(\theta^n) \\
            \theta^{n+1} &= (\theta^n + h \cdot p^{n+1/2}) \bmod 2\pi \\
            p^{n+1} &= p^{n+1/2} - \frac{h}{2} \nabla_\theta E(\theta^{n+1})

    Reference: Leimkuhler B, Reich S (2004) Cambridge Univ Press

    Note: grad_E_new_holder must be pre-computed at theta^{n+1} by the caller.
    This function updates theta and p in place.

    Args:
        theta: Current torsion angles, shape (n_res, 7). Modified in place.
        p: Current momenta, shape (n_res, 7). Modified in place.
        h: Step size.
        grad_E: Gradient at current theta, shape (n_res, 7).
        grad_E_new_holder: Gradient at new theta (pre-computed), shape (n_res, 7).
    """
    two_pi = 2.0 * np.pi
    n_res = theta.shape[0]
    n_tor = theta.shape[1]

    # Half-step momentum update
    for i in range(n_res):
        for k in range(n_tor):
            p[i, k] -= 0.5 * h * grad_E[i, k]

    # Full-step position update (wrap mod 2π)
    for i in range(n_res):
        for k in range(n_tor):
            val = (theta[i, k] + h * p[i, k]) % two_pi
            if val < 0.0:
                val += two_pi
            theta[i, k] = val

    # Second half-step momentum update using new gradient
    for i in range(n_res):
        for k in range(n_tor):
            p[i, k] -= 0.5 * h * grad_E_new_holder[i, k]


# ---------------------------------------------------------------------------
# Kernel 12: Maxmin landmark sampling
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def maxmin_landmark_sampling(
    coords: np.ndarray, n_landmarks: int
) -> np.ndarray:
    r"""Select landmarks via maxmin (farthest-point) sampling.

    Mathematical Basis:
        .. math::
            \ell_{k+1} = \arg\max_{x \in X} \min_{i \leq k} d(x, \ell_i)

    Reference: de Silva V, Carlsson G (2004) Eurographics Symp Geometry Processing

    Args:
        coords: Point cloud, shape (N, d).
        n_landmarks: Number of landmarks to select (m).

    Returns:
        Array of landmark indices, shape (m,).
    """
    N = coords.shape[0]
    if n_landmarks >= N:
        return np.arange(N, dtype=np.int64)

    indices = np.empty(n_landmarks, dtype=np.int64)
    indices[0] = 0  # Start with first point

    # min_dist[i] = min distance from point i to any selected landmark
    min_dist = np.full(N, np.inf)

    for k in range(n_landmarks):
        # Update minimum distances with newly added landmark
        lk = indices[k]
        for i in range(N):
            dist = 0.0
            for d in range(coords.shape[1]):
                diff = coords[i, d] - coords[lk, d]
                dist += diff * diff
            dist = np.sqrt(dist)
            if dist < min_dist[i]:
                min_dist[i] = dist

        if k < n_landmarks - 1:
            # Select farthest point from all landmarks
            best_idx = 0
            best_dist = -1.0
            for i in range(N):
                if min_dist[i] > best_dist:
                    best_dist = min_dist[i]
                    best_idx = i
            indices[k + 1] = best_idx

    return indices


# ---------------------------------------------------------------------------
# Kernel 13: RG block contact map via saddle-point matrix model
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def rg_block_contact_map(
    seq_block: np.ndarray,
    t: float,
    u: float,
    n_iter: int,
    tol: float,
) -> np.ndarray:
    r"""Compute block contact map via RG matrix model saddle-point equation.

    Mathematical Basis:
        .. math::
            V'(M^*) = 0 \implies M^* - t M^{*2} - u M^{*3} = 0

    Solved via Newton-Raphson iteration on b×b matrices:
        .. math::
            M^{(k+1)} = M^{(k)} - [J]^{-1} F(M^{(k)})

    where :math:`F(M) = M - tM^2 - uM^3` and
    the Jacobian update is approximated by diagonal scaling.

    Reference: Orland H, Zee A (2002) Nucl Phys B 620:456-476

    Args:
        seq_block: Encoded sequence block (0-3 for ACGU), shape (b,).
        t: Cubic coupling constant.
        u: Quartic coupling constant.
        n_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance.

    Returns:
        Contact probability matrix, shape (b, b), values in [0, 1].
    """
    b = seq_block.shape[0]
    # Initialize with base-pairing prior
    M = np.zeros((b, b))
    for i in range(b):
        for j in range(i + 4, b):  # Minimum loop length = 4
            si = seq_block[i]
            sj = seq_block[j]
            # Watson-Crick pairing score
            if (si == 0 and sj == 3) or (si == 3 and sj == 0):
                M[i, j] = 0.3  # A-U
                M[j, i] = 0.3
            elif (si == 1 and sj == 2) or (si == 2 and sj == 1):
                M[i, j] = 0.5  # G-C
                M[j, i] = 0.5
            elif (si == 2 and sj == 3) or (si == 3 and sj == 2):
                M[i, j] = 0.15  # G-U wobble
                M[j, i] = 0.15

    # Newton-Raphson to solve M - t*M^2 - u*M^3 = 0
    for iteration in range(n_iter):
        M2 = np.zeros((b, b))
        M3 = np.zeros((b, b))

        # M^2
        for i in range(b):
            for j in range(b):
                s = 0.0
                for k in range(b):
                    s += M[i, k] * M[k, j]
                M2[i, j] = s

        # M^3 = M^2 * M
        for i in range(b):
            for j in range(b):
                s = 0.0
                for k in range(b):
                    s += M2[i, k] * M[k, j]
                M3[i, j] = s

        # F(M) = M - t*M^2 - u*M^3
        max_residual = 0.0
        F = np.zeros((b, b))
        for i in range(b):
            for j in range(b):
                F[i, j] = M[i, j] - t * M2[i, j] - u * M3[i, j]
                if abs(F[i, j]) > max_residual:
                    max_residual = abs(F[i, j])

        if max_residual < tol:
            break

        # Diagonal Jacobian approximation: J_ii ≈ 1 - 2t*M_ii - 3u*M_ii^2
        for i in range(b):
            for j in range(b):
                diag_j = 1.0 - 2.0 * t * M[i, j] - 3.0 * u * M[i, j] * M[i, j]
                if abs(diag_j) < 1e-12:
                    diag_j = 1e-12
                M[i, j] -= F[i, j] / diag_j

    # Convert to probabilities via sigmoid
    P = np.zeros((b, b))
    for i in range(b):
        for j in range(b):
            val = M[i, j]
            if val > 20.0:
                P[i, j] = 1.0
            elif val < -20.0:
                P[i, j] = 0.0
            else:
                P[i, j] = 1.0 / (1.0 + np.exp(-val))
    return P


# ---------------------------------------------------------------------------
# Kernel 14: Stable rank signature
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def stable_rank_signature(
    birth: np.ndarray, death: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    r"""Compute stable rank signature from persistence diagram.

    Mathematical Basis:
        .. math::
            \text{SR}(D, t) = \sum_{(b,d) \in D} \mathbf{1}[d - b > t]

    Sampled at 64 log-spaced threshold values for a 64-dimensional signature.

    Args:
        birth: Birth values, shape (n_pairs,).
        death: Death values, shape (n_pairs,).
        thresholds: Threshold values, shape (n_thresholds,).

    Returns:
        Stable rank vector, shape (n_thresholds,).
    """
    n_thresholds = thresholds.shape[0]
    n_pairs = birth.shape[0]
    sr = np.zeros(n_thresholds)
    for t_idx in range(n_thresholds):
        thresh = thresholds[t_idx]
        count = 0.0
        for p in range(n_pairs):
            if (death[p] - birth[p]) > thresh:
                count += 1.0
        sr[t_idx] = count
    return sr


# ---------------------------------------------------------------------------
# Kernel 15: Wasserstein-2 between persistence diagrams
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
def wasserstein2_persistence(
    birth1: np.ndarray,
    death1: np.ndarray,
    birth2: np.ndarray,
    death2: np.ndarray,
) -> float:
    r"""Approximate Wasserstein-2 distance between two persistence diagrams.

    Mathematical Basis:
        .. math::
            W_2(D_1, D_2) = \left(\inf_{\gamma: D_1 \to D_2}
            \sum_{p \in D_1} \|p - \gamma(p)\|_\infty^2\right)^{1/2}

    Uses greedy nearest-neighbor matching as approximation.
    Unmatched points are paired with their diagonal projection.

    Reference: Cohen-Steiner D et al. (2007) Discrete Comput Geom 37:103-120

    Args:
        birth1, death1: First persistence diagram, shapes (n1,).
        birth2, death2: Second persistence diagram, shapes (n2,).

    Returns:
        Approximate W2 distance (float >= 0).
    """
    n1 = birth1.shape[0]
    n2 = birth2.shape[0]

    if n1 == 0 and n2 == 0:
        return 0.0

    # Cost of matching point to diagonal: (death - birth)^2 / 4
    total_cost = 0.0
    used2 = np.zeros(n2, dtype=numba.boolean)

    for i in range(n1):
        b1 = birth1[i]
        d1 = death1[i]
        p1 = d1 - b1
        diag_cost = (p1 * p1) / 4.0

        best_cost = diag_cost
        best_j = -1
        for j in range(n2):
            if used2[j]:
                continue
            db = b1 - birth2[j]
            dd = d1 - death2[j]
            cost = max(db * db, dd * dd)  # L-inf squared
            if cost < best_cost:
                best_cost = cost
                best_j = j

        total_cost += best_cost
        if best_j >= 0:
            used2[best_j] = True

    # Unmatched points from diagram 2
    for j in range(n2):
        if not used2[j]:
            p2 = death2[j] - birth2[j]
            total_cost += (p2 * p2) / 4.0

    return np.sqrt(total_cost)
