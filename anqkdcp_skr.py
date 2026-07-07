"""
ANQKDCP-SKR: SKR-gated exact CTMC model for blocking probability in
QKD-enabled MCF-based SS-EONs with counter-propagation.

Extends anqkdcp_stable.ANQKDCP with a per-transition secret-key-rate
admission gate:

  * every arrival transition places one new quantum channel; before the
    candidate successor state is accepted, the SKR of that quantum
    channel is evaluated INSIDE the candidate state (so its QBER already
    sees every co-active companion/data channel, including the request's
    own), and the transition survives only if SKR >= r_min;
  * noise model is crosstalk-only with a constant floor (consistent with
    dedicated-core isolation): the only state-dependent noise is
    residual inter-core XT from occupied classical cells in core-modes
    PHYSICALLY adjacent to the quantum core, split into co- and
    counter-propagating contributions;
  * `noise_adj_edges` is a NEW input: the physical XT adjacency of the
    MCF geometry (cm-index pairs). It is deliberately separate from
    adj_edges_p1/p2: those drive slot-level XT-avoidance INSIDE each
    partition, while noise_adj_edges carries the couplings the
    bi-partition cut (quantum core <-> classical core), which slot
    marking cannot handle and the SKR gate polices;
  * the gate is POLICY-AGNOSTIC: it prunes whatever candidate set the
    parent allocation policy produces, so both first fit ('ff') and
    random fit ('rf') are supported.  Under FF the semantics are
    "first fit, then gate" (a gate-failed first fit blocks the plan);
    under RF the parent's rate split is renormalized over the admitted
    candidates automatically because the gate shrinks the successor set
    BEFORE the parent divides the arrival rate;
  * admission-time semantics: existing calls are never re-evaluated or
    torn down, so track_calls exact departures carry over unchanged;
  * blocking decomposition: per class, BP = BP_spec + BP_skr, where
    BP_spec sums pi over states with no spectrum placement at all and
    BP_skr over states where spectrum exists but every candidate fails
    the gate.  The decomposition re-check runs under the same policy
    the chain was generated with.

Set skr_enabled=False (or r_min=0 with gamma=0) to recover the stable
model exactly, under either policy.
"""

import math
import warnings
from anqkdcp_stable import (ANQKDCP, ANQKDCPError,
                            FREE, Q, C_CH, D_CH, CHANNEL_VALUES)

POLICIES = ('ff', 'rf')


class SKRParams:
    """Per-link physical parameters of the crosstalk-only SKR model.
    All values are user inputs; only m_co / m_ct vary with the state."""

    def __init__(self,
                 link_km=50.0,      # link length L [km]
                 alpha_db=0.20,     # fiber attenuation alpha [dB/km]
                 node_loss_db=1.0,  # fixed insertion loss per link [dB]
                 det_eff=0.10,      # detector efficiency eta_d
                 mu_photon=0.50,    # mean photon number per pulse
                 y_const=1.0e-4,    # constant noise floor (dark counts +
                                    # all non-XT impairments at minimum)
                 e_opt=0.015,       # optical misalignment error
                 ec_f=1.16,         # error-correction inefficiency f
                 gamma_co=1.6e-4,   # residual XT yield per co-propagating
                                    # adjacent classical slot
                 gamma_ct=5.0e-5,   # ... per counter-propagating slot
                 r_min=2.0e-4):     # admission threshold R_min
        if link_km <= 0 or alpha_db < 0 or node_loss_db < 0:
            raise ANQKDCPError("invalid link loss parameters")
        if not (0 < det_eff <= 1) or mu_photon <= 0:
            raise ANQKDCPError("invalid detector/source parameters")
        if min(y_const, gamma_co, gamma_ct, r_min) < 0 or ec_f < 1:
            raise ANQKDCPError("invalid noise/threshold parameters")
        self.link_km, self.alpha_db, self.node_loss_db = \
            link_km, alpha_db, node_loss_db
        self.det_eff, self.mu_photon = det_eff, mu_photon
        self.y_const, self.e_opt, self.ec_f = y_const, e_opt, ec_f
        self.gamma_co, self.gamma_ct, self.r_min = gamma_co, gamma_ct, r_min
        loss_db = alpha_db * link_km + node_loss_db
        self.transmittance = 10.0 ** (-loss_db / 10.0)
        self.p_click = 1.0 - math.exp(-mu_photon * self.transmittance * det_eff)

    @staticmethod
    def _h(x):
        if x <= 0.0 or x >= 1.0:
            return 0.0
        return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)

    def skr(self, m_co, m_ct):
        """(QBER, SKR) for a quantum channel seeing m_co co-propagating
        and m_ct counter-propagating adjacent classical slots."""
        y0 = self.y_const + self.gamma_co * m_co + self.gamma_ct * m_ct
        gain = self.p_click + y0
        if gain <= 0.0:
            return 0.5, 0.0
        e = (self.e_opt * self.p_click + 0.5 * y0) / gain
        he = self._h(e)
        r = gain * (1.0 - self.ec_f * he - he)
        return e, max(0.0, r)

    def admits(self, m_co, m_ct):
        return self.skr(m_co, m_ct)[1] >= self.r_min


class ANQKDCP_SKR(ANQKDCP):
    # ------------------------------------------------------------------
    def __init__(self, *args, noise_adj_edges=(), skr_params=None,
                 skr_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.skr_params = skr_params if skr_params is not None else SKRParams()
        self.skr_enabled = bool(skr_enabled)

        # physical XT adjacency (any cm pairs, typically inter-partition)
        nbrs = {cm: set() for cm in range(self.n_cm)}
        for a, b in noise_adj_edges:
            if not (0 <= a < self.n_cm and 0 <= b < self.n_cm) or a == b:
                raise ANQKDCPError(f"noise_adj_edges entry {(a, b)} invalid")
            nbrs[a].add(b)
            nbrs[b].add(a)
        self._noise_nbrs = {cm: tuple(sorted(s)) for cm, s in nbrs.items()}
        if self.skr_enabled and not noise_adj_edges:
            warnings.warn("SKR gate enabled with empty noise_adj_edges: "
                          "QBER is load-independent (constant floor only)")
        # row -> (cm, fiber) reverse map for locating the new quantum cell
        self._row_to_cmf = {r: cmf for cmf, r in self._cm_to_row.items()}
        # allocation policy the chain is generated with; recorded on the
        # first gated allocation so blocking_decomposition re-checks under
        # the SAME policy without the caller having to repeat it
        self._gen_policy = None
        # diagnostics
        self.skr_admitted = 0
        self.skr_pruned = 0

    # ------------------------------------------------------------------
    def _noise_counts(self, grid, q_cm, q_fiber):
        """(m_co, m_ct): occupied classical cells in core-modes physically
        adjacent to q_cm, split by propagation direction vs the quantum."""
        m_co = m_ct = 0
        for cm in self._noise_nbrs[q_cm]:
            for fiber in (0, 1):
                row = grid[self._cm_to_row[(cm, fiber)]]
                occ = 0
                for v in row:
                    if v == C_CH or v == D_CH:
                        occ += 1
                if fiber == q_fiber:
                    m_co += occ
                else:
                    m_ct += occ
        return m_co, m_ct

    def _new_quantum_cell(self, old_grid, new_grid):
        for r in range(self.rows):
            orow, nrow = old_grid[r], new_grid[r]
            if orow is nrow:
                continue
            for s in range(self.slot):
                if nrow[s] == Q and orow[s] != Q:
                    return r, s
        return None

    def skr_of_candidate(self, old_grid, new_grid):
        """(QBER, SKR, m_co, m_ct) of the quantum channel newly placed
        between old_grid and new_grid. None if no new quantum found."""
        cell = self._new_quantum_cell(old_grid, new_grid)
        if cell is None:
            return None
        q_cm, q_fiber = self._row_to_cmf[cell[0]]
        m_co, m_ct = self._noise_counts(new_grid, q_cm, q_fiber)
        e, r = self.skr_params.skr(m_co, m_ct)
        return e, r, m_co, m_ct

    # ------------------------------------------------------------------
    def _allocate_request(self, state, cls, policy='ff'):
        """Spectrum-feasible candidates from the parent, filtered by the
        SKR gate.  Works for every parent policy:

          'ff' : the parent returns at most one candidate per plan, so
                 gating yields "first fit, then gate" -- a gate-failed
                 first fit blocks that plan; the model does NOT fall
                 through to the next spectral fit inside the plan;
          'rf' : the parent returns every feasible placement; pruning
                 the set here means the parent splits the arrival rate
                 over the ADMITTED candidates only, i.e. the RF rate
                 split is renormalized by the gate automatically.
        """
        if policy not in POLICIES:
            raise ANQKDCPError(f"unknown allocation policy {policy!r}; "
                               f"expected one of {POLICIES}")
        self._gen_policy = policy
        cands = super()._allocate_request(state, cls, policy)
        if not self.skr_enabled or not cands:
            return cands
        kept = []
        for ns in cands:
            info = self.skr_of_candidate(state, ns)
            if info is None:            # defensive; should not happen
                continue
            if info[1] >= self.skr_params.r_min:
                kept.append(ns)
                self.skr_admitted += 1
            else:
                self.skr_pruned += 1
        return kept

    # ------------------------------------------------------------------
    def blocking_decomposition(self, policy=None):
        """Per-class and average blocking split into spectrum blocking
        (no spectrum placement exists) and SKR blocking (spectrum exists,
        every candidate fails the gate). Returns
        (avg_total, avg_spec, avg_skr, per_class dict).

        The spectrum-only re-check runs under `policy`; when omitted it
        reuses the policy the chain was generated with.  Note that the
        re-check is a boolean EXISTENCE test, and first fit finds a
        placement iff one exists, so 'ff' and 'rf' classify every
        blocked state identically; matching the generation policy is
        kept for exactness and self-documentation."""
        if not self.states:
            raise ANQKDCPError("call generate_states() first")
        if policy is None:
            policy = self._gen_policy or 'ff'
        elif policy not in POLICIES:
            raise ANQKDCPError(f"unknown allocation policy {policy!r}; "
                               f"expected one of {POLICIES}")
        avg, per_class_total, pi = self.blocking_probability()

        per_class = {}
        num_s = num_k = den = 0.0
        for cls in self.classes:
            bp_spec = bp_skr = 0.0
            for i in range(len(self.states)):
                if cls not in self.blocking[i]:
                    continue
                grid = self.states[i][0] if self.track_calls else self.states[i]
                # spectrum-only re-check (parent allocator, no gate)
                spectrum_ok = bool(
                    super()._allocate_request(grid, cls, policy))
                if spectrum_ok:
                    bp_skr += pi[i]
                else:
                    bp_spec += pi[i]
            # the two mechanisms tile the blocking states exactly; any
            # drift here indicates an inconsistency with the parent model
            if not math.isclose(bp_spec + bp_skr, per_class_total[cls],
                                rel_tol=1e-9, abs_tol=1e-12):
                warnings.warn(
                    f"class {cls}: decomposition mismatch "
                    f"(spec {bp_spec:.3e} + skr {bp_skr:.3e} != "
                    f"total {per_class_total[cls]:.3e})")
            per_class[cls] = {"total": per_class_total[cls],
                              "spectrum": float(bp_spec),
                              "skr": float(bp_skr)}
            num_s += self.lambda_val[cls] * bp_spec
            num_k += self.lambda_val[cls] * bp_skr
            den += self.lambda_val[cls]
        avg_spec = num_s / den if den > 0 else 0.0
        avg_skr = num_k / den if den > 0 else 0.0
        return avg, avg_spec, avg_skr, per_class

    def summary(self):
        s = super().summary()
        s.update({"skr_enabled": self.skr_enabled,
                  "policy": self._gen_policy,
                  "r_min": self.skr_params.r_min,
                  "link_km": self.skr_params.link_km,
                  "skr_admitted": self.skr_admitted,
                  "skr_pruned": self.skr_pruned})
        return s
