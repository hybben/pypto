"""FlashAttention performance kernel with TKV=256 N-chunking.

Scaled-up version of test_fa_performance.py that supports TKV=256 (and
potentially 512) by tiling the QK matmul along the N axis and the PV matmul
along the K axis.  Each on-chip tile stays [?, 128] per chunk; only the loop
count changes.

Key differences from test_fa_performance.py:
  - TKV is now 256 (configurable)
  - N_CHUNKS = TKV // 128 inner loops over 128-col chunks
  - Two-pass softmax: pass 1 finds global row-max across all N-chunks,
    pass 2 computes exp(x - max), row-sum, cast, and stores P
  - PV matmul accumulates across K-chunks via matmul_acc

Usage:
    python3 tests/ut/frontend/flash_attention/test_fa_perf_tkv.py
"""

import math
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 1
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128;  TKV = 512;  TD = 128
TS_HALF = TS // 2
CHUNK_KV = 128                      # hardware chunk size (L0B double buffering)
N_CHUNKS = TKV // CHUNK_KV          # number of 128-col chunks
SCALE = 1.0 / math.sqrt(TD)

# Byte sizes per tile (all tiles are [?, CHUNK_KV] or [?, TD] per chunk)
Q_HALF_F16 = TS_HALF * TD * 2         # [64, 128] FP16 = 16KB
KT_F16     = TD * CHUNK_KV * 2        # [128, 128] FP16 = 32KB
V_F16      = CHUNK_KV * TD * 2        # [128, 128] FP16 = 32KB
P_HALF_F16 = TS_HALF * CHUNK_KV * 2   # [64, 128] FP16 = 16KB
QK_HALF_F32 = TS_HALF * CHUNK_KV * 4  # [64, 128] FP32 = 32KB
PV_HALF_F32 = TS_HALF * TD * 4        # [64, 128] FP32 = 32KB

# ---- MAT (512KB) ----
MA0 = 0;                  MA0_PONG = MA0 + Q_HALF_F16
MA1 = Q_HALF_F16 * 2;    MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2;  MA2_PONG = MA2 + P_HALF_F16
MA3 = MA2 + P_HALF_F16 * 2; MA3_PONG = MA3 + V_F16
assert MA3_PONG + V_F16 <= 512 * 1024, f"MAT overflow: {MA3_PONG + V_F16} > {512*1024}"

LA0 = 0;  LA1 = Q_HALF_F16
RA0 = 0;  RA1 = KT_F16
CA0 = 0;  CA1 = QK_HALF_F32

# ---- VEC addresses (192KB) ----
# VEC tiles process [64, 128] per chunk (CHUNK_KV, not TKV)
VB4_CK = TS_HALF * CHUNK_KV * 4   # [64, 128] FP32 = 32KB
VB2_CK = TS_HALF * CHUNK_KV * 2   # [64, 128] FP16 = 16KB
VB4    = TS_HALF * TD * 4          # [64, 128] FP32 = 32KB
VB2    = TS_HALF * TD * 2          # [64, 128] FP16 = 16KB
VB_RED = TS_HALF * 1 * 4           # 256B -- [64,1] FP32

VA0  = 0                            # qk_vec   [64, 128] FP32
VA1  = VA0 + VB4_CK                 # tmp_vec  [64, 128] FP32
VA2  = VA1 + VB4_CK                 # p_f16    [64, 128] FP16
VA3  = VA2 + VB2_CK                 # reduce_dst [64, 1] FP32
# global_max x 2 (by q_count % 2)
VA_GMAX0 = VA3 + VB_RED;  VA_GMAX1 = VA_GMAX0 + VB_RED
# global_sum x 2 (by q_count % 2)
VA_GSUM0 = VA_GMAX1 + VB_RED;  VA_GSUM1 = VA_GSUM0 + VB_RED
# exp_corr x FIFO_SIZE (by task_id % FIFO_SIZE)
VA_EXP_BASE = VA_GSUM1 + VB_RED
EXP_CORR_ADDRS = [VA_EXP_BASE + i * VB_RED for i in range(FIFO_SIZE)]
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * VB_RED
# chunk_max: [64, 1] FP32 -- for accumulating row-max across N-chunks
VA_CMAX = VA_AFTER_EXP
# chunk_sum: [64, 1] FP32 -- for accumulating row-sum across N-chunks
VA_CSUM = VA_CMAX + VB_RED
VA7  = VA_CSUM + VB_RED              # running_o [64, 128] FP32
VA8  = VA7 + VB4                     # pv_vec    [64, 128] FP32
VA9  = VA8 + VB4                     # o_f16     [64, 128] FP16
assert VA9 + VB2 <= 192 * 1024, f"VEC overflow: {VA9 + VB2} > {192*1024}"

event_ids_01 = (0, 1)
event_ids_23 = (2, 3)

# Cross-core event IDs (0-15 available)
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS  = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16, f"Too many cross-core event IDs: need {3*FIFO_SIZE}, max 16"
QK_MAX_EID = FIFO_SIZE
P_MAX_EID  = 2 * FIFO_SIZE
PV_MAX_EID = 3 * FIFO_SIZE

# PV buffer: 2 Q-slots x FIFO_SIZE task-slots per core
PV_CORE_STRIDE = 2 * FIFO_SIZE * TS

Sq2      = pl.DynVar('Sq')
Sq_fifo_nc = pl.DynVar('SqFifoNc')  # = Sq * FIFO_SIZE * N_CHUNKS (flat QK/P buffer rows)
Skv2     = pl.DynVar('Skv')
D2       = pl.DynVar('D')


# ================================================================
def alloc_cube_buffer():
    """Allocate Cube tiles. All MAT tiles are per-chunk [?, CHUNK_KV=128]."""
    q_mat_type = plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    q_mat_0 = plm.make_tile(q_mat_type, addr=MA0, size=Q_HALF_F16)
    q_mat_1 = plm.make_tile(q_mat_type, addr=MA0_PONG, size=Q_HALF_F16)

    k_mat_type = plm.TileType(shape=[TD, CHUNK_KV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)

    p_mat_type = plm.TileType(shape=[TS_HALF, CHUNK_KV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    p_mat_0 = plm.make_tile(p_mat_type, addr=MA2, size=P_HALF_F16)
    p_mat_1 = plm.make_tile(p_mat_type, addr=MA2_PONG, size=P_HALF_F16)

    v_mat_type = plm.TileType(shape=[CHUNK_KV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    v_mat_0 = plm.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = plm.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)

    left_0 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA0, size=Q_HALF_F16)
    left_1 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA1, size=Q_HALF_F16)
    right_0 = plm.make_tile(plm.TileType(shape=[CHUNK_KV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=KT_F16)
    right_1 = plm.make_tile(plm.TileType(shape=[CHUNK_KV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=KT_F16)
    acc_0 = plm.make_tile(plm.TileType(shape=[TS_HALF, CHUNK_KV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_HALF_F32)
    acc_1 = plm.make_tile(plm.TileType(shape=[TS_HALF, CHUNK_KV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_HALF_F32)

    return ((q_mat_0, q_mat_1), (k_mat_0, k_mat_1), (p_mat_0, p_mat_1),
            (v_mat_0, v_mat_1), (left_0, left_1), (right_0, right_1), (acc_0, acc_1))


# Write alloc_exp_corr_fifo to a temp .py so auto-inline can read its source.
import tempfile as _tf, importlib.util as _ilu, os as _os
def _gen_alloc_exp_corr():
    lines = ["import pypto.language as pl", "import pypto.language.manual as plm", ""]
    lines.append("def alloc_exp_corr_fifo():")
    names, rm_names = [], []
    for i, addr in enumerate(EXP_CORR_ADDRS):
        lines.append(f"    ec{i} = plm.make_tile(plm.TileType(shape=[{TS_HALF}, 1], dtype=pl.FP32, "
                     f"target_memory=pl.MemorySpace.Vec, blayout=2), addr={addr}, size={VB_RED})")
        lines.append(f"    ec{i}_rm = plm.make_tile(plm.TileType(shape=[1, {TS_HALF}], dtype=pl.FP32, "
                     f"target_memory=pl.MemorySpace.Vec), addr={addr}, size={VB_RED})")
        names.append(f"ec{i}"); rm_names.append(f"ec{i}_rm")
    lines.append(f"    return ({', '.join(names)}), ({', '.join(rm_names)})")
    src = "\n".join(lines) + "\n"
    tmp = _os.path.join(_tf.gettempdir(), "_alloc_exp_corr_fifo_tkv.py")
    with open(tmp, "w") as f:
        f.write(src)
    spec = _ilu.spec_from_file_location("_alloc_exp_corr_fifo_tkv", tmp)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.alloc_exp_corr_fifo

alloc_exp_corr_fifo = _gen_alloc_exp_corr()


# ================================================================
#  Cube: compute_qk with N-chunking
# ================================================================
def compute_qk(ctx):
    """QK = Q * K^T with N-chunking: loop over N_CHUNKS 128-col chunks."""
    q_mat_idx = ctx.q_count % 2
    qk_fifo_slot = ctx.task_id % FIFO_SIZE
    skv_off = ctx.task_id * TKV

    for n_chunk in pl.range(0, N_CHUNKS):
        skv_chunk_off = skv_off + n_chunk * CHUNK_KV

        # Load K chunk [skv_chunk_off, 0] with transposed layout
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[ctx.buf_idx])
        if n_chunk == 0:
            if ctx.task_id == 0:
                plm.load(q_mat_buf[q_mat_idx], q, [ctx.sq_off + ctx.row_off, 0])
        plm.load(k_mat_buf[ctx.buf_idx], k, [skv_chunk_off, 0], layout="dn")
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        # TMOV Q -> left (reuse across chunks), K_chunk -> right
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])
        plm.move(left_buf[ctx.l0ab_idx], q_mat_buf[q_mat_idx])
        plm.move(right_buf[ctx.l0ab_idx], k_mat_buf[ctx.buf_idx])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[ctx.buf_idx])

        # Matmul (fresh for each N-chunk -- NOT accumulating)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
        plm.matmul(acc_buf[ctx.l0c_idx], left_buf[ctx.l0ab_idx], right_buf[ctx.l0ab_idx])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])

        # l0c_store to flat qk_buf: chunk n_chunk in row block n_chunk * sq_fifo_dim
        plm.l0c_store(acc_buf[ctx.l0c_idx],
                      [n_chunk * sq_fifo_dim + qk_fifo_slot * sq_dim + ctx.sq_off + ctx.row_off, 0],
                      [TS_HALF, CHUNK_KV], qk_buf)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])

        ctx.l0ab_idx = 1 - ctx.l0ab_idx
        ctx.l0c_idx = 1 - ctx.l0c_idx

    # Signal QK ready after all N-chunks are stored
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_fifo_slot], max_event_id=QK_MAX_EID)
    return


# ================================================================
#  Cube: compute_pv with K-chunking (accumulate across chunks)
# ================================================================
def compute_pv(ctx):
    """PV = P * V with K-chunking: accumulate across N_CHUNKS."""
    q_mat_idx = ctx.q_count % 2
    pv_task_slot = ctx.task_id % FIFO_SIZE
    pv_fifo_slot = ctx.task_id % FIFO_SIZE

    for k_chunk in pl.range(0, N_CHUNKS):
        p_chunk_off = ctx.task_id * TKV + k_chunk * CHUNK_KV
        v_chunk_off = ctx.task_id * TKV + k_chunk * CHUNK_KV

        # Load V chunk and P chunk
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[ctx.buf_idx])
        plm.load(v_mat_buf[ctx.buf_idx], v, [v_chunk_off, 0])
        if k_chunk == 0:
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY_IDS[pv_fifo_slot], max_event_id=P_MAX_EID)
        plm.load(p_mat_buf[ctx.buf_idx], p_buf, [k_chunk * sq_fifo_dim + pv_fifo_slot * sq_dim + ctx.sq_off + ctx.row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        # TMOV P -> left, V -> right
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])
        plm.move(left_buf[ctx.l0ab_idx], p_mat_buf[ctx.buf_idx])
        plm.move(right_buf[ctx.l0ab_idx], v_mat_buf[ctx.buf_idx])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[ctx.buf_idx])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        # Matmul: fresh for first chunk, accumulate for subsequent
        # Only wait for FIX→M on first chunk (no l0c_store between accumulations)
        if k_chunk == 0:
            pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
            plm.matmul(acc_buf[ctx.l0c_idx], left_buf[ctx.l0ab_idx], right_buf[ctx.l0ab_idx])
        else:
            plm.matmul_acc(acc_buf[ctx.l0c_idx], acc_buf[ctx.l0c_idx], left_buf[ctx.l0ab_idx], right_buf[ctx.l0ab_idx])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])

        ctx.l0ab_idx = 1 - ctx.l0ab_idx
        # Only flip l0c_idx on the LAST chunk (ACC must stay on same bank for accumulation)
        if k_chunk == N_CHUNKS - 1:
            pass  # flip below after l0c_store

    # l0c_store accumulated result to pv_buf
    plm.l0c_store(acc_buf[ctx.l0c_idx],
                  [ctx.core_id * PV_CORE_STRIDE + q_mat_idx * FIFO_SIZE * TS + pv_task_slot * TS + ctx.row_off, 0],
                  [TS_HALF, TD], pv_buf)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
    ctx.l0c_idx = 1 - ctx.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_task_slot], max_event_id=PV_MAX_EID)
    return


# ================================================================
#  Vector: two-pass softmax for N-chunks
# ================================================================
@pl.inline
def softmax_body(task_id):
    """Two-pass softmax across N_CHUNKS for TKV>128.

    Pass 1: find global row-max across all N-chunk QK tiles.
    Pass 2: exp(x - max), row-sum, cast to FP16, store P.
    Uses row_off, sq_off, sq_dim from closure.
    """
    p_fifo_slot = task_id % FIFO_SIZE
    skv_off = task_id * TKV

    # ---- Pass 1: find chunk_max across all N-chunks ----
    for n in pl.range(0, N_CHUNKS):
        plm.load(qk_vec, qk_buf, [n * sq_fifo_dim + p_fifo_slot * sq_dim + sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.row_max(reduce_dst, qk_vec, tmp_vec)
        pl.system.bar_v()
        if n == 0:
            plm.muls(chunk_max_rm, reduce_dst_rm, 1.0)
        else:
            plm.maximum(chunk_max_rm, chunk_max_rm, reduce_dst_rm)
            pl.system.bar_v()

    # ---- Apply chunk_max to global state (same logic as before) ----
    if task_id == 0:
        plm.muls(global_max_rm_cur, chunk_max_rm, 1.0)
    if task_id > 0:
        plm.maximum(reduce_dst_rm, chunk_max_rm, global_max_rm_cur)
        pl.system.bar_v()
        plm.sub(exp_corr_rm_fifo[p_fifo_slot], global_max_rm_cur, reduce_dst_rm)
        pl.system.bar_v()
        plm.muls(global_max_rm_cur, reduce_dst_rm, 1.0)
        pl.system.bar_v()
        plm.muls(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot], SCALE)
        plm.exp(exp_corr_rm_fifo[p_fifo_slot], exp_corr_rm_fifo[p_fifo_slot])

    # ---- Pass 2: exp(x - max) * SCALE, row-sum, cast, store per chunk ----
    for n in pl.range(0, N_CHUNKS):
        plm.load(qk_vec, qk_buf, [n * sq_fifo_dim + p_fifo_slot * sq_dim + sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        # Use chunk_max (ColMajor alias) for row_expand_sub
        plm.row_expand_sub(tmp_vec, qk_vec, chunk_max)
        plm.muls(tmp_vec, tmp_vec, SCALE)
        plm.exp(qk_vec, tmp_vec)
        pl.system.bar_v()
        plm.row_sum(reduce_dst, qk_vec, tmp_vec)
        pl.system.bar_v()
        if n == 0:
            plm.muls(chunk_sum_rm, reduce_dst_rm, 1.0)
        else:
            plm.add(chunk_sum_rm, chunk_sum_rm, reduce_dst_rm)
        plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(p_buf, p_f16, [n * sq_fifo_dim + p_fifo_slot * sq_dim + sq_off + row_off, 0])

    # ---- Integrate chunk_sum into global_sum ----
    if task_id == 0:
        plm.muls(global_sum_rm_cur, chunk_sum_rm, 1.0)
    if task_id > 0:
        pl.system.bar_v()
        plm.mul(global_sum_rm_cur, global_sum_rm_cur, exp_corr_rm_fifo[p_fifo_slot])
        pl.system.bar_v()
        plm.add(global_sum_rm_cur, global_sum_rm_cur, chunk_sum_rm)
    return


@pl.inline
def compute_p(task_id):
    """Softmax on QK tile -> P. Includes cross-core sync."""
    p_fifo_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_fifo_slot], max_event_id=QK_MAX_EID)
    softmax_body(task_id)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_fifo_slot], max_event_id=P_MAX_EID)
    return


def compute_gu(task_id, q_count):
    """GU: running output update. Uses row_off from closure. Includes cross-core sync."""
    q_mat_idx = q_count % 2
    pv_slot = task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot], max_event_id=PV_MAX_EID)
    if task_id == 0:
        plm.load(running_o, pv_buf, [core_id * PV_CORE_STRIDE + q_mat_idx * FIFO_SIZE * TS + pv_slot * TS + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    if task_id > 0:
        gu_fifo_slot = task_id % FIFO_SIZE
        plm.load(pv_vec, pv_buf, [core_id * PV_CORE_STRIDE + q_mat_idx * FIFO_SIZE * TS + pv_slot * TS + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.row_expand_mul(running_o, running_o, exp_corr_fifo[gu_fifo_slot])
        plm.add(running_o, running_o, pv_vec)
    return


# ================================================================
#  Kernel
# ================================================================
@fe.kernel
def fa_perf_tkv_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[Sq_fifo_nc, D2], pl.FP32],    # flat: N_CHUNKS row blocks × CHUNK_KV cols
    p_buf:  pl.Tensor[[Sq_fifo_nc, D2], pl.FP16],    # flat: N_CHUNKS row blocks × CHUNK_KV cols
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, D2], pl.FP32],
) -> pl.Tensor[[Sq2, D2], pl.FP16]:

    sq_dim = Sq2
    skv_dim = Skv2
    sq_fifo_dim = sq_dim * FIFO_SIZE  # row block size per N-chunk in flat qk/p buffers
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.block.index_cast(pl.block.get_block_num())
    core_id = pl.block.index_cast(pl.block.get_block_idx())

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_mat_buf, k_mat_buf, p_mat_buf, v_mat_buf, left_buf, right_buf, acc_buf = alloc_cube_buffer()

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

        ctx = pl.struct(sq_off=0, row_off=0, task_id=0, q_count=0, buf_idx=0, l0ab_idx=0, l0c_idx=0, core_id=core_id)

        for qi in pl.range(core_id, sq_tiles, num_cores):
            ctx.sq_off = qi * TS

            for row_idx in pl.range(0, 2):
                ctx.row_off = row_idx * TS_HALF

                # ---- Prologue: pre-compute QK[0 .. QK_PRELOAD-1] ----
                for pre in pl.range(0, QK_PRELOAD):
                    ctx.task_id = pre
                    ctx.buf_idx = (ctx.q_count * skv_tiles + pre) % 2
                    compute_qk(ctx)

                # ---- Main loop: QK[ki+preload] ahead + PV[ki] current ----
                for ki in pl.range(0, skv_tiles):
                    next_ki = ki + QK_PRELOAD
                    if next_ki < skv_tiles:
                        ctx.task_id = next_ki
                        ctx.buf_idx = (ctx.q_count * skv_tiles + next_ki) % 2
                        compute_qk(ctx)
                    ctx.task_id = ki
                    ctx.buf_idx = (ctx.q_count * skv_tiles + ki) % 2
                    compute_pv(ctx)

                ctx.q_count = ctx.q_count + 1

        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        qk_vec     = plm.make_tile(plm.TileType(shape=[TS_HALF, CHUNK_KV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_CK)
        tmp_vec    = plm.make_tile(plm.TileType(shape=[TS_HALF, CHUNK_KV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_CK)
        p_f16      = plm.make_tile(plm.TileType(shape=[TS_HALF, CHUNK_KV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_CK)
        reduce_dst = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA3, size=VB_RED)
        reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)

        red_type    = plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2)
        red_rm_type = plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)

        # Double-buffered global_max / global_sum (by q_count % 2)
        gmax_rm_0 = plm.make_tile(red_rm_type, addr=VA_GMAX0, size=VB_RED)
        gmax_rm_1 = plm.make_tile(red_rm_type, addr=VA_GMAX1, size=VB_RED)
        global_max_rm_buf = (gmax_rm_0, gmax_rm_1)

        gsum_0    = plm.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1    = plm.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        gsum_rm_0 = plm.make_tile(red_rm_type, addr=VA_GSUM0, size=VB_RED)
        gsum_rm_1 = plm.make_tile(red_rm_type, addr=VA_GSUM1, size=VB_RED)
        global_sum_buf    = (gsum_0, gsum_1)
        global_sum_rm_buf = (gsum_rm_0, gsum_rm_1)

        # FIFO exp_corr (by task_id % FIFO_SIZE)
        exp_corr_fifo, exp_corr_rm_fifo = alloc_exp_corr_fifo()

        # chunk_max: persists across N-chunk iterations in pass 1
        # ColMajor alias for row_expand_sub, RowMajor alias for maximum/muls
        chunk_max    = plm.make_tile(red_type, addr=VA_CMAX, size=VB_RED)
        chunk_max_rm = plm.make_tile(red_rm_type, addr=VA_CMAX, size=VB_RED)
        # chunk_sum: accumulates row-sum across N-chunks in pass 2
        chunk_sum_rm = plm.make_tile(red_rm_type, addr=VA_CSUM, size=VB_RED)

        running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
        pv_vec    = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
        o_f16     = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)

        q_count = 0
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS

            for row_idx in pl.range(0, 2):
                row_off = row_idx * TS_HALF
                q_idx = q_count % 2
                # Aliases for current Q tile's global state
                global_max_rm_cur = global_max_rm_buf[q_idx]
                global_sum_cur    = global_sum_buf[q_idx]
                global_sum_rm_cur = global_sum_rm_buf[q_idx]

                # ---- Prologue: compute_p for pre-loaded QK tiles ----
                for pre in pl.range(0, QK_PRELOAD):
                    compute_p(pre)

                # ---- Main loop: P[ki+preload] ahead + GU[ki] current ----
                for ki in pl.range(0, skv_tiles):
                    next_ki = ki + QK_PRELOAD
                    if next_ki < skv_tiles:
                        compute_p(next_ki)
                    compute_gu(ki, q_count)

                q_count = q_count + 1

                # Final: normalize and store output for this row half
                plm.row_expand_div(running_o, running_o, global_sum_cur)
                plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store(o, o_f16, [sq_off + row_off, 0])
    return o


# ================================================================
#  Reference + Tests
# ================================================================
def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T) * scale_val
    attn = torch.softmax(qk, dim=-1)
    return torch.matmul(attn, v.float()).half()


def test_fa_perf_tkv():
    compiled = fe.compile(fa_perf_tkv_kernel, arch="a3", codegen_mode="cce")
    print("compiled:", compiled.lib_path)
    device = "npu:5"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        (8192, 8192, TD, 24),
    ]:
        print(f"\nFA-Perf-TKV ({sq},{skv},{d}) TKV={TKV} N_CHUNKS={N_CHUNKS} cores={num_cores}  QK_PRELOAD={QK_PRELOAD}")
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE * N_CHUNKS, CHUNK_KV), device=device, dtype=torch.float32)
        p_t  = torch.zeros((sq * FIFO_SIZE * N_CHUNKS, CHUNK_KV), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        fe.launch(None, num_cores, compiled, q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
        torch.npu.synchronize()
        o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        diff = (o_t - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o_t, o_ref, rtol=5e-2, atol=5e-2)
        print("  PASS")


if __name__ == "__main__":
    print(f"FA perf TKV={TKV}: double-buffer + QK pre-compute + N-chunking (QK_PRELOAD={QK_PRELOAD}, FIFO={FIFO_SIZE}, N_CHUNKS={N_CHUNKS})")
    print("=" * 60)
    test_fa_perf_tkv()
    print("\nAll FlashAttention TKV tests passed!")
