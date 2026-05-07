# counter_rf.py — DAAM-CP analytical model for SS-EONs (RF-only, optimized)
#
# Random-Fit-only counter-propagation enumerator extended with Algorithm 1
# Phase 2 from Ahmed et al. (Algo_ton_ancpdefag.pdf):
#   - Defragmentation state generation (lines 15-31)
#   - Parallel defrag for non-interfering core-mode pairs (line 21-23)
#   - Single-lightpath defrag for interfering pairs (line 25-26)
#   - Combined CTMC with retuning rate µ_d using equilibrium eqs. (3)-(4)
#
# BP / utilisation formula:
#   r = sum((1-s) * lambda * k / mu) / (core * mode * slot * 2)
#
# Rate matrix is built directly from count_out / count_inc (bypasses the
# dense EqToMx path, reducing memory from O(n²) to O(n * avg_edges)).

import sys
import time
import gc
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr, spsolve
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

MAX_STATES = 1_000_000
SOLVER_LSQR_IT_FACTOR = 5
INITIAL_CAPACITY = 1024


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# =============================================================================
# Compact state store: single contiguous int8 array with grow-as-needed buffer.
# =============================================================================
class StateStore:
    """Fixed-shape (col, slot) state store backed by a contiguous int8 array.
    Provides O(1) append/lookup with bytes-based hash index."""

    __slots__ = ("col", "slot", "n", "_buf", "_index")

    def __init__(self, col: int, slot: int):
        self.col = col
        self.slot = slot
        self.n = 0
        self._buf = np.zeros((INITIAL_CAPACITY, col, slot), dtype=np.int8)
        self._index: Dict[bytes, int] = {}

    def _ensure_capacity(self, need: int):
        cap = self._buf.shape[0]
        if need <= cap:
            return
        new_cap = cap
        while new_cap < need:
            new_cap *= 2
        new_buf = np.zeros((new_cap, self.col, self.slot), dtype=np.int8)
        new_buf[:self.n] = self._buf[:self.n]
        self._buf = new_buf

    def add(self, mat: np.ndarray) -> int:
        """Add a state; return its index. Deduplicates."""
        a = np.ascontiguousarray(np.asarray(mat, dtype=np.int8))
        if a.shape != (self.col, self.slot):
            raise ValueError(f"state shape {a.shape} != ({self.col},{self.slot})")
        key = a.tobytes()
        idx = self._index.get(key)
        if idx is not None:
            return idx
        if self.n >= MAX_STATES:
            return 0
        self._ensure_capacity(self.n + 1)
        self._buf[self.n] = a
        self._index[key] = self.n
        self.n += 1
        return self.n - 1

    def __len__(self):
        return self.n

    def get(self, i: int) -> np.ndarray:
        return self._buf[i]

    def get_copy(self, i: int) -> np.ndarray:
        return self._buf[i].copy()

    def key_of(self, i: int) -> bytes:
        return self._buf[i].tobytes()

    def find(self, mat: np.ndarray) -> Optional[int]:
        a = np.ascontiguousarray(np.asarray(mat, dtype=np.int8))
        return self._index.get(a.tobytes())

    def to_list_of_lists(self, i: int) -> List[List[int]]:
        return self._buf[i].astype(int).tolist()

    def trim(self):
        """Release unused buffer capacity after enumeration is complete."""
        if self._buf.shape[0] > self.n:
            self._buf = self._buf[:self.n].copy()


# =============================================================================
# Main solver: RF-only CTMC enumerator + DAAM-CP Phase 2 defragmentation.
# =============================================================================
class BlockingProbability:
    """Random-Fit blocking probability solver with optional Phase-2 defrag.

    State semantics (row=core-mode-direction, col=slot):
      0  = available slot
      1  = allocated slot
     -1  = XT-avoided slot (IC-XT/IM-XT guard) or own-tail guard

    A row r is 'interfering' iff int_list[r] != 0; non-interfering rows can
    be co-retuned (parallel defrag, Alg.1 lines 21-23). Interfering rows
    undergo single-lightpath defrag (Alg.1 lines 25-26)."""

    def __init__(self,
                 core: int, mode: int, slot: int, classes: List[int],
                 edgelist, lambda_val, mu_val,
                 output: str, store: StateStore, adj_m: List[int],
                 mu_d: Optional[float] = None,
                 enable_defrag: bool = False):
        self.core = core
        self.mode = mode
        self.slot = slot
        self.classes = list(classes)
        self.lambda_val = list(lambda_val)
        self.mu_val = list(mu_val)
        self.col = self.core * self.mode * 2  # counter-prop: forward + backward
        self.edgeList = edgelist
        self.adj_mat = [[0] * self.col for _ in range(self.col)]
        self.int_list = list(adj_m)   # must be set before _create_edges
        self._create_edges()
        self.store = store
        self.matrix: Optional[np.ndarray] = None  # scratch (slot, col) for RandomFitfillAdj

        # Precompute last-column-of-core set for O(1) isLastCoreCol lookup.
        self._last_core_col_set = {i * self.mode - 1 for i in range(1, self.core + 1)}

        self._count_out: Optional[List[Dict[int, int]]] = None
        self._last_P: Optional[np.ndarray] = None

        self.mu_d: Optional[float] = mu_d if (mu_d is not None and mu_d > 0) else None
        self.enable_defrag: bool = bool(enable_defrag) and (self.mu_d is not None)
        # defrag_states[i] = (u_idx, class_k, target_row_r, packed_normal_idx)
        self.defrag_states: List[Tuple[int, int, int, int]] = []
        # (u_idx, class_k) -> count of defrag options
        self.n_defrag_per_uk: Dict[Tuple[int, int], int] = {}
        self._last_P_defrag: float = 0.0
        self._last_mean_retune_time: float = 0.0

    # ---------- adjacency helpers ----------
    def isLastCoreCol(self, col: int) -> bool:
        return col in self._last_core_col_set

    def isAdj(self, c1: int, c2: int) -> bool:
        return self.adj_mat[c1][c2] == 1

    def _create_edges(self):
        for edge in self.edgeList:
            r, c = edge[0], edge[1]
            if r < self.col and c < self.col:
                self.adj_mat[r][c] = 1
        # XT adjacency from consecutive int_list (1,2) pairs.  Rows j and j+1
        # are XT-adjacent when one carries value 1 and the other value 2 in the
        # pattern [0,1,2,0,1,2,...].  Without this, rows outside the edgelist
        # range are unconstrained and the state space explodes.
        for j in range(len(self.int_list) - 1):
            v1, v2 = self.int_list[j], self.int_list[j + 1]
            if v1 != 0 and v2 != 0 and v1 != v2 and j + 1 < self.col:
                self.adj_mat[j][j + 1] = 1
                self.adj_mat[j + 1][j] = 1

    # ---------- RF cross-row XT primitive ----------
    # 'self.matrix' is the transposed state shaped (slot, col).
    # 'row' here is a SLOT index, 'col' is the LIGHTPATH/row index.

    def _mark_xt_and_guard(self, row: int, size: int, flip: bool, c: int):
        """Mark XT slots AND 1-slot guard bands in adjacent column c.

        XT zone  : size slots starting at 'row' (direction depends on flip).
        Guard band: 1 slot before the first XT slot and 1 slot after the last,
                    following the XT adjacency defined by adj_m (int_list)."""
        for i in range(size):
            rr = row - i if flip else row + i
            if 0 <= rr < self.slot and self.matrix[rr][c] != 1:
                self.matrix[rr][c] = -1
        # 1-slot guard band on each side of the XT zone.
        rr_before = (row + 1) if flip else (row - 1)
        if 0 <= rr_before < self.slot and self.matrix[rr_before][c] == 0:
            self.matrix[rr_before][c] = -1
        rr_after = (row - size) if flip else (row + size)
        if 0 <= rr_after < self.slot and self.matrix[rr_after][c] == 0:
            self.matrix[rr_after][c] = -1

    def RandomFitfillAdj(self, row: int, col: int, size: int, flip: bool):
        """Mark XT + guard-band slots in every row that is XT-adjacent to 'col'.

        The original code only examined higher-indexed neighbours, causing
        asymmetric marking: placing in a lastCoreCol row never marked the
        lower-indexed partner, so states with both adjacent rows occupied
        were reachable and the state space exploded.  Iterating over all
        adj_mat entries fixes the asymmetry without changing the semantics."""
        if col >= self.col:
            return
        for c in range(self.col):
            if c != col and self.isAdj(col, c):
                self._mark_xt_and_guard(row, size, flip, c)

    # ---------- RF placement primitive ----------
    def _rf_apply_placement(self, state: np.ndarray, j: int, slot_start: int, x: int) -> np.ndarray:
        """Apply a single RF placement of class x at row j, slot start.
        Returns a new contiguous (col, slot) int8 array."""
        tmp = state.copy()
        tmp[j, slot_start:slot_start + x] = 1
        # Guard bands are cross-row only: _mark_xt_and_guard adds ±1 guard slots
        # in XT-adjacent rows via RandomFitfillAdj below.
        A = np.ascontiguousarray(tmp.T, dtype=np.int8)
        saved = self.matrix
        try:
            self.matrix = A
            self.RandomFitfillAdj(slot_start, j, x, False)
            self.matrix = self.matrix[::-1, ::-1]
            self.RandomFitfillAdj(self.slot - slot_start - 1, self.col - j - 1, x, True)
            out = self.matrix[::-1, ::-1].T
        finally:
            self.matrix = saved
        return np.ascontiguousarray(out, dtype=np.int8)

    # ---------- enumeration ----------
    def randomFit(self, start: int = 0):
        """Enumerate (or extend) the RF state space from index 'start'.

        Returns (count_out_list, count_inc_list) for states [start, len(store)).
        Pass start=n_pre to process only newly added packed states."""
        if start == 0 and len(self.store) == 0:
            self.store.add(np.zeros((self.col, self.slot), dtype=np.int8))

        slotArr: List[Dict[int, int]] = []
        transitionArr: List[Dict[int, List[int]]] = []
        sidx = start
        while sidx < len(self.store):
            sc = self.store.get(sidx)  # view (col, slot)
            slotValue: Dict[int, int] = {}
            finalT: Dict[int, List[int]] = {}
            for j in range(self.col):
                row = sc[j]
                for x in self.classes:
                    # Try every valid starting position (advance by 1, matching reference).
                    for state_check in range(self.slot - x + 1):
                        if not (row[state_check:state_check + x] == 0).all():
                            continue
                        out = self._rf_apply_placement(sc, j, state_check, x)
                        ind = self.store.add(out)
                        slotValue[x] = slotValue.get(x, 0) + 1
                        finalT.setdefault(x, []).append(ind)
            slotArr.append(slotValue)
            transitionArr.append(finalT)
            sidx += 1
        return slotArr, transitionArr

    # ---------- transition bookkeeping ----------
    def transitionIncState(self, count_inc):
        """Build incoming-transition map: state_idx -> {class: [source_indices]}."""
        incoming: Dict[int, Dict[int, List[int]]] = {}
        for j in range(len(count_inc)):
            for k, v in count_inc[j].items():
                for i in v:
                    if i not in incoming:
                        incoming[i] = {k: [j]}
                    else:
                        incoming[i].setdefault(k, []).append(j)
        return incoming

    # ---------- direct sparse CTMC matrix builder ----------
    def _build_ctmc_matrix(self, count_out, count_inc, transitionInc):
        """Build the CTMC generator matrix directly as sparse COO triples.

        Each row d encodes the balance equation for state d:
          diagonal:       outflow via arrivals + outflow via departures
          Q[d, j] = -µk  for j in count_inc[d][k]   (departure j→d inflow)
          Q[d, s] = -λk/Ns  for s in transitionInc[d][k]  (arrival s→d inflow)

        Avoids the O(n²) dense allocation used by EqToMx."""
        n = len(self.store)
        lam = {self.classes[i]: float(self.lambda_val[i]) for i in range(len(self.classes))}
        mu  = {self.classes[i]: float(self.mu_val[i])     for i in range(len(self.classes))}

        r_list: List[int] = []
        c_list: List[int] = []
        v_list: List[float] = []
        diag = np.zeros(n, dtype=np.float64)

        for d in range(n):
            # Outflow via arrivals (λ_k per class with any valid placement from d)
            for k in count_out[d]:
                diag[d] += lam.get(k, 0.0)

            # For each class-k connection in d (one per source s in transitionInc[d][k]):
            #   - departure outflow from d: len(sources) * µ_k
            #   - departure inflow to source s: A[s, d] = -µ_k
            #   - arrival inflow to d from source s: A[d, s] = -λ_k / N_s_k
            if d in transitionInc:
                for k, sources in transitionInc[d].items():
                    mu_k = mu.get(k, 0.0)
                    lam_k = lam.get(k, 0.0)
                    diag[d] += len(sources) * mu_k
                    for s in sources:
                        r_list.append(s); c_list.append(d); v_list.append(-mu_k)
                        n_pos = len(count_inc[s].get(k, []))
                        if n_pos > 0:
                            r_list.append(d); c_list.append(s)
                            v_list.append(-lam_k / n_pos)

        for i in range(n):
            if diag[i] != 0.0:
                r_list.append(i); c_list.append(i); v_list.append(diag[i])

        return r_list, c_list, v_list, n

    def _make_sparse_matrix(self, r_list, c_list, v_list, n):
        if _HAS_SCIPY:
            return csr_matrix((v_list, (r_list, c_list)), shape=(n, n), dtype=np.float64)
        A = np.zeros((n, n), dtype=np.float64)
        for r, c, v in zip(r_list, c_list, v_list):
            A[r, c] += v
        return A

    # ---------- linear solver (direct preferred over iterative) ----------
    def _solve_stationary(self, A) -> np.ndarray:
        """Solve Ax = b (with last row replaced by normalisation π·1 = 1)."""
        if _HAS_SCIPY:
            if not hasattr(A, 'tocsr'):
                A = csr_matrix(A)
            n = A.shape[0]
            A = A.tolil()
            A[-1, :] = 1.0
            A = A.tocsr()
            b = np.zeros(n, dtype=np.float64)
            b[-1] = 1.0
            try:
                P = spsolve(A, b)
                if np.all(np.isfinite(P)):
                    P = np.maximum(P, 0.0)
                    s = P.sum()
                    return (P / s) if (s > 0 and np.isfinite(s)) else np.ones(n) / n
            except Exception:
                pass
            # Fall back to iterative LSQR for ill-conditioned systems.
            it_lim = max(50, SOLVER_LSQR_IT_FACTOR * n)
            P = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=it_lim)[0]
        else:
            A = np.asarray(A, dtype=np.float64).copy() if not hasattr(A, 'toarray') else A.toarray()
            n = A.shape[0]
            A[-1, :] = 1.0
            b = np.zeros(n, dtype=np.float64)
            b[-1] = 1.0
            try:
                P = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                P = np.linalg.lstsq(A, b, rcond=None)[0]
        P = np.maximum(P, 0.0)
        s = P.sum()
        return (P / s) if (s > 0 and np.isfinite(s)) else np.ones_like(P) / len(P)

    # =========================================================================
    # ===========  DAAM-CP Algorithm 1 Phase 2: defragmentation  ==============
    # =========================================================================

    @staticmethod
    def _row_alloc_count(row: np.ndarray) -> int:
        return int((row == 1).sum())

    @staticmethod
    def _row_can_directly_fit_k(row: np.ndarray, k: int) -> bool:
        """True iff there exist k contiguous zeros somewhere in the row."""
        if k <= 0:
            return True
        n = len(row)
        if n < k:
            return False
        is_zero = (row == 0).astype(np.int8)
        csum = np.concatenate(([0], np.cumsum(is_zero)))
        windows = csum[k:] - csum[:n - k + 1]
        return bool((windows == k).any())

    def _state_can_directly_fit_k(self, state: np.ndarray, k: int) -> bool:
        for r in range(state.shape[0]):
            if self._row_can_directly_fit_k(state[r], k):
                return True
        return False

    def _row_is_interfering(self, r_idx: int) -> bool:
        return 0 <= r_idx < len(self.int_list) and self.int_list[r_idx] != 0

    def _cross_row_blocked_cols(self, state: np.ndarray, r_idx: int) -> set:
        cols = set()
        rm1 = r_idx - 1
        if 0 <= rm1 < len(self.int_list) and self.int_list[rm1] % 2 == 1:
            cols.update(int(c) for c in np.where(state[rm1] == 1)[0])
        rp1 = r_idx + 1
        if (0 <= rp1 < len(self.int_list)
                and self.int_list[rp1] != 0 and self.int_list[rp1] % 2 == 0):
            cols.update(int(c) for c in np.where(state[rp1] == 1)[0])
        return cols

    def _row_defrag_feasible(self, state: np.ndarray, r_idx: int, k: int) -> bool:
        n_alloc = self._row_alloc_count(state[r_idx])
        if n_alloc == 0:
            return False
        need = n_alloc + 1 + k
        if self._row_is_interfering(r_idx):
            blocked = self._cross_row_blocked_cols(state, r_idx)
            usable = self.slot - len(blocked)
        else:
            usable = self.slot
        return need <= usable

    def _is_state_fragmented_for_k(self, state: np.ndarray, k: int) -> bool:
        if int((state == 0).sum()) < k:
            return False
        if self._state_can_directly_fit_k(state, k):
            return False
        for r in range(state.shape[0]):
            if self._row_defrag_feasible(state, r, k):
                return True
        return False

    def _build_packed_state(self, state: np.ndarray, r_idx: int, k: int) -> Optional[np.ndarray]:
        n_alloc = self._row_alloc_count(state[r_idx])
        if n_alloc == 0:
            return None
        is_interfering = self._row_is_interfering(r_idx)
        blocked = self._cross_row_blocked_cols(state, r_idx) if is_interfering else set()
        usable = self.slot - len(blocked)
        if n_alloc + 1 + k > usable:
            return None

        avail = [c for c in range(self.slot) if c not in blocked]
        if len(avail) < n_alloc + 1 + k:
            return None

        blocks_to_replay: List[Tuple[int, int, int]] = []
        for rr in range(state.shape[0]):
            if rr == r_idx:
                continue
            for (bs, bl) in self._identify_blocks(state[rr]):
                blocks_to_replay.append((rr, bs, bl))

        orig_blocks_r: List[int] = [bl for (_, bl) in self._identify_blocks(state[r_idx])]

        cursor = 0
        for blen in orig_blocks_r:
            if cursor + blen > len(avail):
                return None
            if any(avail[cursor + p] - avail[cursor] != p for p in range(blen)):
                return None
            blocks_to_replay.append((r_idx, avail[cursor], blen))
            cursor += blen
            if cursor < len(avail):
                cursor += 1

        placed_k = False
        while cursor + k <= len(avail):
            if all(avail[cursor + p] - avail[cursor] == p for p in range(k)):
                blocks_to_replay.append((r_idx, avail[cursor], k))
                placed_k = True
                break
            cursor += 1
        if not placed_k:
            return None

        canvas = np.zeros((self.col, self.slot), dtype=np.int8)
        for (rr, st, ln) in blocks_to_replay:
            canvas = self._rf_apply_placement(canvas, rr, st, ln)
        return canvas

    def generate_defrag_states_phase2(self) -> int:
        if not self.enable_defrag:
            return 0
        self.defrag_states = []
        self.n_defrag_per_uk = {}
        n_pre = len(self.store)
        for u_idx in range(n_pre):
            state_u = self.store.get(u_idx)
            for k in self.classes:
                if not self._is_state_fragmented_for_k(state_u, k):
                    continue
                for r_idx in range(self.col):
                    if self._row_can_directly_fit_k(state_u[r_idx], k):
                        continue
                    if not self._row_defrag_feasible(state_u, r_idx, k):
                        continue
                    packed = self._build_packed_state(state_u, r_idx, k)
                    if packed is None:
                        continue
                    packed_idx = self.store.add(packed)
                    self.defrag_states.append((u_idx, int(k), r_idx, packed_idx))
                    key_uk = (u_idx, int(k))
                    self.n_defrag_per_uk[key_uk] = self.n_defrag_per_uk.get(key_uk, 0) + 1
        return len(self.defrag_states)

    @staticmethod
    def _identify_blocks(row: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of (start, length) for each run of 1s in row."""
        is_one = row == 1
        if not is_one.any():
            return []
        padded = np.empty(len(row) + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = is_one.view(np.int8)
        padded[-1] = 0
        diff = np.diff(padded)
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        return list(zip(starts.tolist(), (ends - starts).tolist()))

    def _add_explicit_departures_sparse(self, rows: List[int], cols: List[int],
                                        data: List[float], diag: np.ndarray,
                                        u_prime: int):
        state = self.store.get(u_prime)
        for r_idx in range(self.col):
            for (b_start, b_len) in self._identify_blocks(state[r_idx]):
                if b_len not in self.classes:
                    continue
                blocks_keep: List[Tuple[int, int, int]] = []
                for rr in range(self.col):
                    for (bs, bl) in self._identify_blocks(state[rr]):
                        if rr == r_idx and bs == b_start and bl == b_len:
                            continue
                        blocks_keep.append((rr, bs, bl))
                Y_state = np.zeros((self.col, self.slot), dtype=np.int8)
                for (rr, bs, bl) in blocks_keep:
                    Y_state = self._rf_apply_placement(Y_state, rr, bs, bl)
                Y_idx = self.store.find(Y_state)
                if Y_idx is None or Y_idx == u_prime:
                    continue
                k_idx = self.classes.index(b_len)
                mu_k = float(self.mu_val[k_idx])
                diag[u_prime] += mu_k
                rows.append(Y_idx)
                cols.append(u_prime)
                data.append(-mu_k)

    def _solve_with_defrag(self, A_normal) -> Tuple[float, float]:
        """Sparse augmentation of normal-state generator matrix per eq. (4)."""
        if hasattr(A_normal, 'tocoo'):
            A_coo = A_normal.tocoo()
            n_normal = A_normal.shape[0]
        else:
            A_normal = np.asarray(A_normal, dtype=np.float64)
            n_normal = A_normal.shape[0]
            nz = np.argwhere(A_normal != 0.0)
            A_coo = type('coo', (), {})()
            A_coo.row = nz[:, 0]
            A_coo.col = nz[:, 1]
            A_coo.data = A_normal[nz[:, 0], nz[:, 1]]

        n_defrag = len(self.defrag_states)
        N = n_normal + n_defrag

        rows: List[int] = list(np.asarray(A_coo.row).tolist())
        cols: List[int] = list(np.asarray(A_coo.col).tolist())
        data: List[float] = list(np.asarray(A_coo.data, dtype=np.float64).tolist())
        diag = np.zeros(N, dtype=np.float64)
        keep_rows, keep_cols, keep_data = [], [], []
        for r, c, v in zip(rows, cols, data):
            if r == c:
                diag[r] += v
            else:
                keep_rows.append(r)
                keep_cols.append(c)
                keep_data.append(v)
        rows, cols, data = keep_rows, keep_cols, keep_data

        L_by_class = {self.classes[i]: float(self.lambda_val[i])
                      for i in range(len(self.classes))}

        processed_uk = set()
        for v_local, (u_idx, k, _r, packed_idx) in enumerate(self.defrag_states):
            v_idx = n_normal + v_local
            if (u_idx, k) not in processed_uk:
                diag[u_idx] += L_by_class[k]
                processed_uk.add((u_idx, k))
            N_uk = self.n_defrag_per_uk[(u_idx, k)]
            rate_uv = L_by_class[k] / float(N_uk)
            rows.append(v_idx)
            cols.append(u_idx)
            data.append(-rate_uv)
            diag[v_idx] += float(self.mu_d)
            rows.append(packed_idx)
            cols.append(v_idx)
            data.append(-float(self.mu_d))

        packed_indices = sorted(set(pk for (_u, _k, _r, pk) in self.defrag_states))
        for u_prime in packed_indices:
            if abs(diag[u_prime]) > 1e-12:
                continue
            self._add_explicit_departures_sparse(rows, cols, data, diag, u_prime)

        for i in range(N):
            if diag[i] != 0.0:
                rows.append(i)
                cols.append(i)
                data.append(float(diag[i]))

        if _HAS_SCIPY:
            A = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)
        else:
            A = np.zeros((N, N), dtype=np.float64)
            for r, c, v in zip(rows, cols, data):
                A[r, c] += v

        P = self._solve_stationary(A)
        self._last_P = P.copy()
        self._last_P_defrag = float(P[n_normal:].sum()) if n_defrag > 0 else 0.0
        self._last_mean_retune_time = (1.0 / float(self.mu_d)) if self.mu_d else 0.0

        defrag_uk = set((u, k) for (u, k, _, _) in self.defrag_states)
        s = []
        for cls in self.classes:
            mass = 0.0
            for i in range(n_normal):
                rf_blocking = (
                    self._count_out[i].get(cls, 0) == 0
                    if (self._count_out is not None and i < len(self._count_out))
                    else True
                )
                if rf_blocking and (i, cls) not in defrag_uk:
                    mass += float(P[i])
            s.append(mass)

        N_bp = sum(s[j] * self.lambda_val[j] for j in range(len(s)))
        D = float(sum(self.lambda_val)) if self.lambda_val else 1.0
        N2 = sum(((1.0 - s[j]) * self.lambda_val[j] * self.classes[j]) / self.mu_val[j]
                 for j in range(len(s)))
        b_p = N_bp / D if D != 0 else 0.0
        r = N2 / (self.core * self.mode * self.slot * 2)
        return b_p, r

    # ---------- BP solver ----------
    def calcBP(self, A) -> Tuple[float, float]:
        n = len(self.store)
        P = self._solve_stationary(A)
        self._last_P = P.copy()
        if self._count_out is None:
            raise RuntimeError("_count_out not set; call run() first")
        s = []
        for cls in self.classes:
            mask = np.fromiter(
                (self._count_out[i].get(cls, 0) == 0 if i < len(self._count_out) else True
                 for i in range(n)),
                count=n, dtype=bool
            )
            s.append(float(P[mask].sum()))
        N = sum(s[j] * self.lambda_val[j] for j in range(len(s)))
        D = float(sum(self.lambda_val)) if self.lambda_val else 1.0
        N2 = sum(((1.0 - s[j]) * self.lambda_val[j] * self.classes[j]) / self.mu_val[j]
                 for j in range(len(s)))
        b_p = N / D if D != 0 else 0.0
        r = N2 / (self.core * self.mode * self.slot * 2)
        return b_p, r

    def run(self, count_out, count_inc, transitionInc) -> Tuple[float, float]:
        """Solve for the current lambda/mu. Builds the sparse CTMC matrix
        directly from count_out/count_inc, bypassing EqToMx."""
        n = len(self.store)
        if len(count_out) < n:
            count_out = list(count_out) + [{} for _ in range(n - len(count_out))]
        if len(count_inc) < n:
            count_inc = list(count_inc) + [{} for _ in range(n - len(count_inc))]
        self._count_out = count_out

        r_list, c_list, v_list, _ = self._build_ctmc_matrix(count_out, count_inc, transitionInc)
        A = self._make_sparse_matrix(r_list, c_list, v_list, n)

        if self.enable_defrag and self.defrag_states:
            return self._solve_with_defrag(A)
        return self.calcBP(A)


# =============================================================================
# RF-only driver
# =============================================================================
def calculateProb(enable_defrag: bool = True, mu_d: float = 50.0,
                  classes=None, edgeList=None, core=None, mode=None, slot=None,
                  load_val: int = 3, adj_m=None,
                  log_path: str = 'CPoutput_rf.txt'):
    """Solve BP/U for RF policy with optional DAAM-CP Phase-2 defrag."""
    if classes is None:   classes  = [2]
    if edgeList is None:  edgeList = [(0, 1), (0, 2), (1, 0), (1, 3),
                                      (2, 0), (2, 3), (3, 1), (3, 2)]
    if core is None:      core = 3
    if mode is None:      mode = 2
    if slot is None:      slot = 5
    if adj_m is None:     adj_m = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    orig_stdout = sys.stdout
    f = open(log_path, 'w')
    sys.stdout = Tee(orig_stdout, f)
    try:
        _run_calculateProb(enable_defrag, mu_d, classes, edgeList,
                           core, mode, slot, load_val, adj_m)
    finally:
        sys.stdout = orig_stdout
        f.close()


def _run_calculateProb(enable_defrag, mu_d, classes, edgeList,
                       core, mode, slot, load_val, adj_m):
    print(f"[config] core={core} mode={mode} slot={slot} classes={classes} "
          f"enable_defrag={enable_defrag} mu_d={mu_d if enable_defrag else 'N/A'}")

    suffix = 'defrag' if enable_defrag else 'nodefrag'
    file_name = f'CPStates_rf_{slot}_{classes}_{load_val}_{suffix}.csv'
    with open(file_name, 'w') as fF:
        fF.write('policy,ind,load,states_normal,states_defrag,core,mode,slots,classes,'
                 'blocking_probability,resource_utilization,'
                 'mean_retuning_time,P_defrag_occupied,time_sec\n')

    base_lambda = [(x + 1) * 0.1 for x in range(len(classes))]
    base_mu     = [1 for _ in classes]

    # --- One-time enumeration (state set is rate-independent) ---
    t0 = time.time()
    col = core * mode * 2
    store = StateStore(col=col, slot=slot)
    rf = BlockingProbability(core, mode, slot, classes, edgeList,
                             base_lambda, base_mu, 'rf',
                             store, adj_m, mu_d=mu_d, enable_defrag=enable_defrag)
    count_out, count_inc = rf.randomFit()
    n_pre = len(store)
    n_def = 0
    if enable_defrag:
        n_def = rf.generate_defrag_states_phase2()
        if len(store) > n_pre:
            # Extend transitions for newly added packed states only.
            extra_out, extra_inc = rf.randomFit(start=n_pre)
            count_out = count_out + extra_out
            count_inc = count_inc + extra_inc

    store.trim()  # release unused buffer capacity
    transitionInc = rf.transitionIncState(count_inc)
    t_setup = time.time() - t0
    print(f"[RF] normal states={len(store)} (was {n_pre}) | defrag states={n_def} "
          f"| setup={t_setup:.2f}s")

    # --- Sweep load points ---
    for i in range(load_val):
        scale = i + 1
        lam = [(x + 1) * 0.1 * scale for x in range(len(classes))]
        mu  = [1 for _ in classes]
        ro  = [lam[k] / mu[k] for k in range(len(classes))]
        load = sum(ro) / len(ro)

        t1 = time.time()
        rf_run = BlockingProbability(core, mode, slot, classes, edgeList,
                                     lam, mu, 'rf', store, adj_m,
                                     mu_d=mu_d, enable_defrag=enable_defrag)
        rf_run.defrag_states   = rf.defrag_states
        rf_run.n_defrag_per_uk = rf.n_defrag_per_uk
        b_p, r_u = rf_run.run(count_out, count_inc, transitionInc)
        t_run = time.time() - t1

        T_ret = (1.0 / float(mu_d)) if (enable_defrag and mu_d) else 0.0
        Pd    = rf_run._last_P_defrag if enable_defrag else 0.0
        with open(file_name, 'a') as fF:
            fF.write(','.join(map(str, ['rf', i, load, len(store), n_def,
                                        core, mode, slot, classes,
                                        float(b_p), float(r_u),
                                        T_ret, Pd, t_run])) + '\n')
        if enable_defrag:
            print(f"  load={load:.2f}  RF: BP={b_p:.6e} U={r_u:.4f} "
                  f"P_def={Pd:.4e}  [T_retune=1/µd={T_ret:.4f}]  ({t_run:.2f}s)")
        else:
            print(f"  load={load:.2f}  RF: BP={b_p:.6e} U={r_u:.4f}  ({t_run:.2f}s)")

        del rf_run
        gc.collect()


if __name__ == "__main__":
    calculateProb(enable_defrag=True, mu_d=50.0)
