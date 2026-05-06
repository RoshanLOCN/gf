# counter.py — DAAM-CP analytical model for SS-EONs (RF-only, optimized)
#
# Random-Fit-only counter-propagation enumerator extended with Algorithm 1
# Phase 2 from Ahmed et al. (Algo_ton_ancpdefag.pdf):
#   - Defragmentation state generation (lines 15-31)
#   - Parallel defrag for non-interfering core-mode pairs (line 21-23)
#   - Single-lightpath defrag for interfering pairs (line 25-26)
#   - Combined CTMC with retuning rate µ_d using equilibrium eqs. (3)-(4)
#
# Optimisations vs. dual-policy version:
#   - States stored as a single contiguous int8 numpy array
#     (n_states, col, slot) -> ~28x lower memory than list-of-list-of-int.
#   - State keys derived from .tobytes() directly on numpy slices (no copy).
#   - Sparse CSR augmentation in _solve_with_defrag (was dense float64).
#   - Vectorised row-level helpers (alloc count, contiguous-fit checks).
#   - FF code paths and unused imports stripped (saves ~50% LOC).
#
# BP / utilisation formula adopted from optimized_analcpdef.py:
#   r = sum((1-s) * lambda * k / mu) / (core * mode * slot * 2)
#
# Requires EquationToMatrix.py (EqToMx).

import sys
import time
import gc
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.sparse import csr_matrix, lil_matrix, identity as sp_identity
    from scipy.sparse.linalg import lsqr, spsolve
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from EquationToMatrix import EqToMx

MAX_STATES = 1_000_000
SOLVER_LSQR_IT_FACTOR = 5
INITIAL_CAPACITY = 1024  # initial allocation for the state array


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# =============================================================================
# Compact state store: single contiguous int8 array with grow-as-needed buffer.
# =============================================================================
class StateStore:
    """Fixed-shape (col, slot) state store backed by a contiguous int8 array.
    Provides O(1) append/lookup with bytes-based hash index. Memory cost is
    n_states * col * slot bytes, typically 10-100x lower than the equivalent
    list-of-list-of-int representation."""

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
        new_buf[: self.n] = self._buf[: self.n]
        self._buf = new_buf

    def add(self, mat: np.ndarray) -> int:
        """Add a state (numpy or list); return its index. Deduplicates."""
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
        # Returns a writable VIEW into the store. Mutate with care.
        return self._buf[i]

    def get_copy(self, i: int) -> np.ndarray:
        return self._buf[i].copy()

    def key_of(self, i: int) -> bytes:
        return self._buf[i].tobytes()

    def find(self, mat: np.ndarray) -> Optional[int]:
        a = np.ascontiguousarray(np.asarray(mat, dtype=np.int8))
        return self._index.get(a.tobytes())

    def to_list_of_lists(self, i: int) -> List[List[int]]:
        # Compatibility shim for code paths still expecting Python lists
        # (currently only the EqToMx output bridge path).
        return self._buf[i].astype(int).tolist()


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
        self.row = self.slot
        self.edgeList = edgelist
        self.adj_mat = [[0] * self.col for _ in range(self.col)]
        self.int_list = list(adj_m)
        self.store = store
        # 'self.matrix' is scratch space used by RandomFitfillAdj (transposed
        # int8 view, shape (slot, col)). Allocated lazily.
        self.matrix: Optional[np.ndarray] = None

        self._count_out: Optional[List[Dict[int, int]]] = None
        self._last_P: Optional[np.ndarray] = None
        self.__matrix_name = (f'matrix_{output}_classes_{len(self.classes)}_core_'
                              f'{self.core}_mode_{self.mode}_slot_{self.slot}.txt')

        # Defragmentation extension state.
        self.mu_d: Optional[float] = mu_d if (mu_d is not None and mu_d > 0) else None
        self.enable_defrag: bool = bool(enable_defrag) and (self.mu_d is not None)
        # defrag_states[i] = (u_idx, class_k, target_row_r, packed_normal_idx)
        self.defrag_states: List[Tuple[int, int, int, int]] = []
        # (u_idx, class_k) -> count of defrag options (splits λ across v's)
        self.n_defrag_per_uk: Dict[Tuple[int, int], int] = {}
        # Reporting:
        self._last_P_defrag: float = 0.0
        self._last_mean_retune_time: float = 0.0

    # ---------- adjacency helpers ----------
    def isLastCoreCol(self, col: int) -> bool:
        cores = [i * self.mode for i in range(1, self.core + 1)]
        for i in cores:
            if col < i - 1: return False
            if col == i - 1: return True
        return False

    def isAdj(self, c1: int, c2: int) -> bool:
        return self.adj_mat[c1][c2] == 1

    # ---------- RF cross-row XT primitive ----------
    # Note on indexing: 'self.matrix' is the transposed state shaped (slot, col).
    # 'row' here is a SLOT index, 'col' is the LIGHTPATH/row index.
    def RandomFitfillAdj(self, row: int, col: int, size: int, flip: bool):
        if col == self.col:
            return
        coreCol = self.isLastCoreCol(col)
        try:
            if coreCol:
                for j in range(1, self.core):
                    c = col + j * self.mode
                    if c >= self.col:
                        break
                    if self.isAdj(col, c):
                        for i in range(size):
                            rr = row - i if flip else row + i
                            if self.matrix[rr][c] != 1:
                                self.matrix[rr][c] = -1
            else:
                if self.isAdj(col, col + 1):
                    for i in range(size):
                        rr = row - i if flip else row + i
                        if self.matrix[rr][col + 1] != 1:
                            self.matrix[rr][col + 1] = -1
                for j in range(1, self.core):
                    c = col + j * self.mode
                    if c >= self.col:
                        break
                    if self.isAdj(col, c):
                        for i in range(size):
                            rr = row - i if flip else row + i
                            if self.matrix[rr][c] != 1:
                                self.matrix[rr][c] = -1
        except Exception:
            pass

    # ---------- RF placement primitive (used by both enum + defrag pack) ----------
    def _rf_apply_placement(self, state: np.ndarray, j: int, slot_start: int, x: int) -> np.ndarray:
        """Apply a single RF placement of class x at row j, slot start, on
        'state' (shape (col, slot)). Mirrors the exact transpose+fill+flip+
        repeat dance used in randomFit, so the resulting state byte-matches
        whatever randomFit would generate. Returns a new (col, slot) array."""
        tmp = state.copy()
        tmp[j, slot_start:slot_start + x] = 1
        if slot_start + x < self.slot:
            tmp[j, slot_start + x] = -1
        # Transpose: (slot, col)
        A = tmp.T.copy()
        saved = self.matrix
        try:
            self.matrix = A
            self.RandomFitfillAdj(slot_start, j, x, False)
            self.matrix = self.matrix[::-1, ::-1]
            self.RandomFitfillAdj(self.row - slot_start - 1, self.col - j - 1, x, True)
            out = self.matrix[::-1, ::-1].T
        finally:
            self.matrix = saved
        return out.astype(np.int8, copy=False)

    # ---------- enumeration ----------
    def randomFit(self):
        """Enumerate the RF state space and collect per-state count_out
        (transitions out by class) and count_inc (transitions in by class).

        Returns (count_out_list, count_inc_list) keyed by state index."""
        # Seed with empty state if store is fresh.
        if len(self.store) == 0:
            self.store.add(np.zeros((self.col, self.slot), dtype=np.int8))

        slotArr: List[Dict[int, int]] = []
        transitionArr: List[Dict[int, List[int]]] = []
        sidx = 0
        # Process states in BFS order; new states discovered append to store
        # and the loop processes them in turn.
        while sidx < len(self.store):
            sc = self.store.get(sidx)  # view (col, slot)
            slotValue: Dict[int, int] = {}
            finalT: Dict[int, List[int]] = {}
            for j in range(self.col):
                row = sc[j]  # 1D view of length slot
                for x in self.classes:
                    state_check = 0
                    while state_check + x <= self.slot:
                        # k zeros at [state_check:state_check+x]?
                        zeros_run = bool((row[state_check:state_check + x] == 0).all())
                        if not zeros_run:
                            state_check += 1
                            continue
                        # boundary: either slot[start+x] == 0 or start+x == slot
                        end_at_slot = (state_check + x == self.slot)
                        if not end_at_slot and row[state_check + x] != 0:
                            state_check += 1
                            continue
                        # Apply placement
                        out = self._rf_apply_placement(sc, j, state_check, x)
                        ind = self.store.add(out)
                        slotValue[x] = slotValue.get(x, 0) + 1
                        finalT.setdefault(x, []).append(ind)
                        state_check += x
                        # NB: sc was a view; subsequent placements should still
                        # see the ORIGINAL row, so we re-fetch it here. Since
                        # _rf_apply_placement copies, sc is unmodified.
                    # end while state_check
                # end for x
            # end for j
            slotArr.append(slotValue)
            transitionArr.append(finalT)
            sidx += 1
        return slotArr, transitionArr

    # ---------- transition bookkeeping (unchanged from baseline) ----------
    def transitionIncState(self, count_inc):
        finalD: Dict[int, int] = {}
        incoming: Dict[int, Dict[int, List[int]]] = {}
        for j in range(len(count_inc)):
            for k, v in count_inc[j].items():
                for i in v:
                    finalD[i] = finalD.get(i, 0) + 1
                    if i not in incoming:
                        incoming[i] = {k: [j]}
                    else:
                        incoming[i].setdefault(k, []).append(j)
        return finalD, incoming

    def createEquation(self, classes_d, mu_d_sym, count, transition,
                       globalTransition, ind, count_inc_dict, finalOutput):
        equation = '(' + '('
        tempD = {}
        l = [classes_d.get(i) for i in count.keys()]
        equation += '+'.join(l)
        tl = []
        if globalTransition.get(ind - 1):
            for k, v in globalTransition.get(ind - 1).items():
                tl.append(str(len(v)) + '*' + mu_d_sym.get(k))
        tempD['outR'] = l + tl
        if tl:
            if l: equation += "+(" + '+'.join(tl) + ")"
            else: equation += '+'.join(tl)
        equation += ')'
        equation += ' * P' + str(ind) + ')'
        if l: equation += '-'
        size = len(transition.keys())
        for k, v in transition.items():
            equation += mu_d_sym.get(k) + '(' + '+'.join(map(str, v)) + ')'
            if 'incR' in tempD: tempD['incR'].append({mu_d_sym.get(k): v})
            else: tempD['incR'] = [{mu_d_sym.get(k): v}]
            size -= 1
            if size > 0: equation += '-'
        if globalTransition.get(ind - 1):
            for k, v in globalTransition.get(ind - 1).items():
                for j in v:
                    equation += ("- ((" + classes_d.get(k) + '/' +
                                 str(len(count_inc_dict[j][k])) + ") * P" + str(j + 1) + ")")
                    if 'EincR' in tempD:
                        tempD['EincR'].append({classes_d.get(k): [str(len(count_inc_dict[j][k])), j + 1]})
                    else:
                        tempD['EincR'] = [{classes_d.get(k): [str(len(count_inc_dict[j][k])), j + 1]}]
        equation += '=0'
        finalOutput.append(tempD)
        return equation

    # ---------- linear solver (sparse path preferred) ----------
    def _solve_stationary(self, A) -> np.ndarray:
        """Solve πA = 0 with normalisation π·1 = 1. Accepts dense ndarray
        or sparse matrix. Uses scipy LSQR if available."""
        if _HAS_SCIPY:
            if not hasattr(A, 'tocsr'):
                A = csr_matrix(A)
            n = A.shape[0]
            A = A.tolil()
            A[-1, :] = 0.0
            A[-1, 0:n] = 1.0
            A = A.tocsr()
            b = np.zeros(n, dtype=np.float64); b[-1] = 1.0
            it_lim = max(50, SOLVER_LSQR_IT_FACTOR * n)
            P = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=it_lim)[0]
        else:
            A = np.asarray(A, dtype=np.float64).copy() if not hasattr(A, 'toarray') else A.toarray()
            n = A.shape[0]
            A[-1, :] = 1.0
            b = np.zeros(n, dtype=np.float64); b[-1] = 1.0
            try:
                P = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                P = np.linalg.lstsq(A, b, rcond=None)[0]
        P = np.maximum(P, 0.0)
        s = P.sum()
        return (P / s) if s > 0 and np.isfinite(s) else np.ones_like(P) / len(P)

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
        # vectorised contiguous-run check
        is_zero = (row == 0).astype(np.int8)
        # rolling window of length k via cumsum
        if n < k:
            return False
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
        """Cols in r_idx blocked by neighbouring rows we won't retune
        (single-lightpath defrag scenario)."""
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
        """Alg.1 lines 21-26: row r supports class k after defrag if it has
        at least one own-lightpath to retune AND (n_alloc + 1 + k) <= usable."""
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
        """Alg.1 lines 18-19: enough total free, not contiguous, AND some
        row defrag-feasible."""
        if int((state == 0).sum()) < k:
            return False
        if self._state_can_directly_fit_k(state, k):
            return False
        for r in range(state.shape[0]):
            if self._row_defrag_feasible(state, r, k):
                return True
        return False

    def _build_packed_state(self, state: np.ndarray, r_idx: int, k: int) -> Optional[np.ndarray]:
        """Construct the post-retuning normal state. Cross-row blocked cols
        stay; own allocations + new class-k pack into remaining cells.
        Then replay placements via RF primitive so XT guards byte-match."""
        n_alloc = self._row_alloc_count(state[r_idx])
        if n_alloc == 0:
            return None
        is_interfering = self._row_is_interfering(r_idx)
        blocked = self._cross_row_blocked_cols(state, r_idx) if is_interfering else set()
        usable = self.slot - len(blocked)
        if n_alloc + 1 + k > usable:
            return None

        # Build target row layout: blocked first (as -1), then n_alloc 1's
        # (we'll overwrite them via RF placement below), then guard, then k.
        avail = [c for c in range(self.slot) if c not in blocked]
        if len(avail) < n_alloc + 1 + k:
            return None
        # Slot positions after pack:
        #   own classes occupy avail[0:n_alloc] (single slots — assuming
        #   class-1 lightpaths in row r; we treat each as a unit-block);
        #   guard at avail[n_alloc];
        #   new class-k at avail[n_alloc+1 : n_alloc+1+k] (must be contiguous
        #   AND align with the packed positions).
        # NB: if some original lightpaths in row r were class>1, they'd
        # still pack to consecutive units; the resulting state remains a
        # valid SS-EON state.
        # We construct the packed state by replaying RF placements on a
        # canvas where the OTHER rows are kept (with their own lightpaths
        # and current XT pattern), and row r is rebuilt from scratch.
        canvas = state.copy()
        # Wipe row r's lightpaths and XT-only entries (keep nothing on r).
        canvas[r_idx, :] = 0
        # Other rows keep lightpaths but we'll have to re-derive their XT
        # contributions onto row r_idx after we place new ones — easier to
        # rebuild the entire state's XT by replaying ALL placements on a
        # blank canvas.
        blocks_to_replay: List[Tuple[int, int, int]] = []
        # Collect blocks from the original state for non-r rows.
        for rr in range(state.shape[0]):
            if rr == r_idx:
                continue
            row = state[rr]
            i = 0
            while i < self.slot:
                if row[i] == 1:
                    j = i
                    while j < self.slot and row[j] == 1:
                        j += 1
                    blocks_to_replay.append((rr, i, j - i))
                    i = j
                else:
                    i += 1

        # Determine how the original row r decomposes into class blocks
        # so we replay them with their actual sizes (preserving total alloc
        # count and matching them to known classes). Best-effort: identify
        # blocks of 1's in the original row and pack them in order.
        orig_blocks_r: List[int] = []
        row_orig = state[r_idx]
        i = 0
        while i < self.slot:
            if row_orig[i] == 1:
                j = i
                while j < self.slot and row_orig[j] == 1:
                    j += 1
                orig_blocks_r.append(j - i)
                i = j
            else:
                i += 1

        # Pack: place orig blocks in order at avail[0..], then new class-k
        # at the next contiguous run that fits.
        cursor = 0
        for blen in orig_blocks_r:
            if cursor + blen > len(avail):
                return None
            # Need contiguity in slot indices for this block.
            if any(avail[cursor + p] - avail[cursor] != p for p in range(blen)):
                return None
            blocks_to_replay.append((r_idx, avail[cursor], blen))
            cursor += blen
            # Skip a guard slot if there's still room.
            if cursor < len(avail):
                cursor += 1
        # Now place class-k (must be contiguous in avail).
        # Move cursor to first position with k-contig avail.
        placed_k = False
        while cursor + k <= len(avail):
            if all(avail[cursor + p] - avail[cursor] == p for p in range(k)):
                blocks_to_replay.append((r_idx, avail[cursor], k))
                placed_k = True
                break
            cursor += 1
        if not placed_k:
            return None

        # Rebuild full state by replaying every block on a blank canvas.
        canvas = np.zeros((self.col, self.slot), dtype=np.int8)
        for (rr, st, ln) in blocks_to_replay:
            canvas = self._rf_apply_placement(canvas, rr, st, ln)
        return canvas

    def generate_defrag_states_phase2(self) -> int:
        """Algorithm 1 lines 15-31: scan all normal states; for each
        fragmented (u, k) pair, generate a defrag state v(u,k,r) per
        defrag-feasible row r and its packed normal target u'."""
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
        blocks = []
        i = 0
        n = len(row)
        while i < n:
            if row[i] == 1:
                j = i
                while j < n and row[j] == 1:
                    j += 1
                blocks.append((i, j - i))
                i = j
            else:
                i += 1
        return blocks

    def _add_explicit_departures_sparse(self, rows: List[int], cols: List[int],
                                        data: List[float], diag: np.ndarray,
                                        u_prime: int):
        """Append explicit departure transitions for a packed state that
        EqToMx left with zero outflow. Mutates rows/cols/data/diag in place."""
        state = self.store.get(u_prime)
        for r_idx in range(self.col):
            for (b_start, b_len) in self._identify_blocks(state[r_idx]):
                if b_len not in self.classes:
                    continue
                # Y_dep = state with this block zeroed; rebuild XT via replay.
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
                rows.append(Y_idx); cols.append(u_prime); data.append(-mu_k)

    def _solve_with_defrag(self, A_normal) -> Tuple[float, float]:
        """Sparse augmentation of normal-state generator matrix per eq. (4).

        For each defrag pair (u, k) with N_uk options:
          A[u,u]      += λ_k                         (outflow into defrag)
          A[v,u]      -= λ_k / N_uk                  (inflow into v from u)
          A[v,v]      += µ_d                         (outflow from v)
          A[u',v]     -= µ_d                         (inflow into u' from v)

        Plus explicit µ_k departures for packed states unreachable in the
        baseline RF closure (keeps mass conservation)."""
        # A_normal may be dense ndarray; convert to COO triple.
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

        # Accumulate diag separately to avoid duplicate (i,i) entries.
        rows: List[int] = list(np.asarray(A_coo.row).tolist())
        cols: List[int] = list(np.asarray(A_coo.col).tolist())
        data: List[float] = list(np.asarray(A_coo.data, dtype=np.float64).tolist())
        diag = np.zeros(N, dtype=np.float64)
        # Pull existing diag values out so we can add to them safely.
        keep_rows, keep_cols, keep_data = [], [], []
        for r, c, v in zip(rows, cols, data):
            if r == c:
                diag[r] += v
            else:
                keep_rows.append(r); keep_cols.append(c); keep_data.append(v)
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
            rows.append(v_idx); cols.append(u_idx); data.append(-rate_uv)
            diag[v_idx] += float(self.mu_d)
            rows.append(packed_idx); cols.append(v_idx); data.append(-float(self.mu_d))

        # Explicit departures for unreachable packed states.
        # An unreachable packed state has diag[u']==0 (no outflow from EqToMx).
        packed_indices = sorted(set(pk for (_u, _k, _r, pk) in self.defrag_states))
        for u_prime in packed_indices:
            if abs(diag[u_prime]) > 1e-12:
                continue
            self._add_explicit_departures_sparse(rows, cols, data, diag, u_prime)

        # Append diag entries.
        for i in range(N):
            if diag[i] != 0.0:
                rows.append(i); cols.append(i); data.append(float(diag[i]))

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

        # BP per class: stationary mass of states that fail RF for class k
        # AND have no defrag relief.
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

    # ---------- baseline (no-defrag) BP path ----------
    def calcBP(self, output, _ignored) -> Tuple[float, float]:
        n = len(self.store)
        if _HAS_SCIPY and not isinstance(output, np.ndarray):
            A = csr_matrix(np.asarray(output, dtype=np.float64))
        else:
            A = np.asarray(output, dtype=np.float64)
            if A.shape != (n, n):
                raise ValueError(f"EqToMx returned {A.shape}, expected ({n},{n})")
            if _HAS_SCIPY:
                A = csr_matrix(A)
        P = self._solve_stationary(A)
        self._last_P = P.copy()
        if self._count_out is None:
            raise RuntimeError("count_out not available; run(...) must be called first")
        s = []
        for cls in self.classes:
            mask = np.fromiter((
                (self._count_out[i].get(cls, 0) == 0) if i < len(self._count_out) else True
                for i in range(n)
            ), count=n, dtype=bool)
            s.append(float(P[mask].sum()))
        N = sum(s[j] * self.lambda_val[j] for j in range(len(s)))
        D = float(sum(self.lambda_val)) if self.lambda_val else 1.0
        N2 = sum(((1.0 - s[j]) * self.lambda_val[j] * self.classes[j]) / self.mu_val[j]
                 for j in range(len(s)))
        b_p = N / D if D != 0 else 0.0
        r = N2 / (self.core * self.mode * self.slot * 2)
        return b_p, r

    def run(self, count_out, count_inc, transitionIncState):
        n = len(self.store)
        if len(count_out) < n:
            count_out = list(count_out) + [{} for _ in range(n - len(count_out))]
        if len(count_inc) < n:
            count_inc = list(count_inc) + [{} for _ in range(n - len(count_inc))]
        self._count_out = count_out
        classesD = {self.classes[i]: f"L{i+1}"  for i in range(len(self.classes))}
        muSym    = {self.classes[i]: f"mu{i+1}" for i in range(len(self.classes))}
        finalOutput = []
        for i in range(len(count_out)):
            _ = self.createEquation(classesD, muSym, count_out[i], count_inc[i],
                                    transitionIncState, i + 1, count_inc, finalOutput)
        l_ops  = [f"{v}={self.lambda_val[j]}" for j, v in enumerate(classesD.values())]
        mu_ops = [f"{v}={self.mu_val[j]}"     for j, v in enumerate(muSym.values())]
        E = EqToMx(len(self.store), finalOutput, self.__matrix_name)
        outputMatrix, bp_structure = E.run(l_ops, mu_ops, muSym)
        if self.enable_defrag and self.defrag_states:
            return self._solve_with_defrag(np.asarray(outputMatrix, dtype=np.float64))
        return self.calcBP(outputMatrix, bp_structure)


# =============================================================================
# RF-only driver
# =============================================================================
def calculateProb(enable_defrag: bool = True, mu_d: float = 50.0,
                  classes=None, edgeList=None, core=None, mode=None, slot=None,
                  load_val: int = 3, adj_m=None,
                  log_path: str = 'CPoutput_rf.txt'):
    """Solve BP/U for RF policy with optional DAAM-CP Phase-2 defrag.

    Args:
      enable_defrag : True  -> apply Algorithm 1 lines 15-31 + eqs. (3),(4)
                      False -> baseline RF blocking model
      mu_d          : retuning completion rate (mean retuning time = 1/mu_d).
                      Ignored when enable_defrag is False.
      load_val      : number of load points to sweep (lambda scaling 1..load_val).
    """
    f = open(log_path, 'w')
    sys.stdout = Tee(sys.stdout, f)

    if classes is None:   classes  = [2]
    if edgeList is None:  edgeList = [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)]
    if core is None:      core = 3
    if mode is None:      mode = 2
    if slot is None:      slot = 5
    if adj_m is None:     adj_m = [1,2,0,1,2,0,1,2,0,1,2]

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

    # --- One-time enumeration with base rates (state set is rate-independent) ---
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
            count_out, count_inc = rf.randomFit()
    finalD, transitionInc = rf.transitionIncState(count_inc)
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
    # Default: defrag enabled. Set enable_defrag=False for the baseline model.
    calculateProb(enable_defrag=True, mu_d=50.0)
