"""FlashAttention kernel using PyPTO IR manual (non-SSA) mode.

K loaded with DN layout (TLOAD transposes on-chip), TILE_SQ=128, TILE_SKV=128, TILE_D=64.

Usage:
    python3 tests/ut/frontend/test_fa.py
"""

import math
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm

# Tile dimensions
TS = 128          # Sq tile size
TKV = 128         # Skv tile size (KV dimension tiling)
TD = 64           # D / head dimension
TS_HALF = TS // 2 # 64 — each vector sub-block processes half of TS rows
SCALE = 1.0 / math.sqrt(TD)

# Cube tile byte sizes
Q_F16   = TS * TD * 2       # 16384  — [128, 64] FP16
KT_F16  = TD * TKV * 2      # 16384  — [64, 128] FP16 (K^T tile)
QK_F16  = TS * TKV * 2      # 32768  — [128, 128] FP16 (QK result per KV tile)
V_F16   = TKV * TD * 2      # 16384  — [128, 64] FP16
P_F16   = TS * TKV * 2      # 32768  — [128, 128] FP16 (P tile per KV iter)
QK_F32  = TS * TKV * 4      # 65536  — [128, 128] FP32 (ACC)
PV_F32  = TS * TD * 4       # 32768  — [128, 64] FP32 (ACC)

# MAT addresses (512KB budget)
MA0 = 0                      # q_mat  [128, 64]  → 16 KB
MA1 = Q_F16                  # k_mat  [64, 128]  → 16 KB
MA2 = MA1 + KT_F16           # p_mat  [128, 128] → 32 KB
MA3 = MA2 + P_F16            # v_mat  [128, 64]  → 16 KB
# Total: 80 KB

# LEFT addresses (64KB budget)
LA0 = 0                      # q_left  [128, 64]  → 16 KB
LA1 = Q_F16                  # p_left  [128, 128] → 32 KB
# Total: 48 KB

# RIGHT addresses (64KB budget)
RA0 = 0                      # k_right [64, 128]  → 16 KB
RA1 = KT_F16                 # v_right [128, 64]  → 16 KB
# Total: 32 KB

# ACC addresses (128KB budget on a3)
CA0 = 0                      # qk_acc  [128, 128] FP32 → 64 KB
CA1 = QK_F32                 # pv_acc  [128, 64]  FP32 → 32 KB
# Total: 96 KB

# VEC addresses — sub-block processes [TS_HALF, TKV] for QK, [TS_HALF, TD] for O
# Reduce tiles use ColMajor [64,1] (256B) for reduce/expand ops,
# with RowMajor [1,64] (256B) aliases at same addr for element-wise ops (via TRESHAPE).
VB4_KV = TS_HALF * TKV * 4   # 32768  — [64, 128] FP32
VB2_KV = TS_HALF * TKV * 2   # 16384  — [64, 128] FP16
VB4    = TS_HALF * TD * 4    # 16384  — [64, 64] FP32
VB2    = TS_HALF * TD * 2    # 8192   — [64, 64] FP16
VB_RED = TS_HALF * 1 * 4     # 256    — [64, 1] FP32 ColMajor (reduce tile)
VA0 = 0                       # qk_vec    [64, 128] FP32
VA1 = VA0 + VB4_KV            # tmp_vec   [64, 128] FP32
VA2 = VA1 + VB4_KV            # p_f16     [64, 128] FP16
VA3 = VA2 + VB2_KV            # reduce_dst  [64, 1] FP32 ColMajor
VA4 = VA3 + VB_RED            # global_max  [64, 1] FP32 ColMajor
VA5 = VA4 + VB_RED            # global_sum  [64, 1] FP32 ColMajor
VA6 = VA5 + VB_RED            # exp_corr    [64, 1] FP32 ColMajor
VA7 = VA6 + VB_RED            # running_o   [64, 64] FP32
VA8 = VA7 + VB4               # pv_vec      [64, 64] FP32
VA9 = VA8 + VB4               # o_f16       [64, 64] FP16
# Total: VA9 + VB2 = 122880 bytes = 120 KB < 192 KB ✓

# Cross-core sync event IDs
QK_READY = 0
P_READY = 1
PV_READY = 2

Sq2 = pl.DynVar('Sq')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')

"""
1. 右矩阵转置的写法
  对于输入 tensor K [N, K]，matmul 需要 K^T 即 [K, N] 的 RIGHT tile:
	┬───────────────────────────────────────────────────┐
	│              写法 2: DN + RIGHT直载                │
	┼───────────────────────────────────────────────────┤
	│ shape=[K,N], stride=[-1,1] → 但被 DN 解释为 [1,K]  │
	┼───────────────────────────────────────────────────┤
	│ DN (列主序)                                        │
	┼───────────────────────────────────────────────────┤
	│ [K_tile, N_tile] (如 [64,128])                    │
	┼───────────────────────────────────────────────────┤
	│ RowMajor (1)                                      │
	┼───────────────────────────────────────────────────┤
	│ ColMajor (2)                                      │
	┼───────────────────────────────────────────────────┤
	│ 列主序读（硬件DMA按列取数据）实现转置                │
	┴───────────────────────────────────────────────────┘
  - GlobalTensor 使用 Layout::DN 列主序，stride=[1, K]
  - TLOAD 按列主序读取数据，相当于硬件层面做了转置
  - 直接写入 RIGHT tile（blayout=row_major, slayout=col_major)
"""
@fe.kernel
def fa_k_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[Sq2, Skv2], pl.FP32],
    p_buf: pl.Tensor[[Sq2, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[Sq2, D2], pl.FP32],
) -> pl.Tensor[[Sq2, D2], pl.FP16]:
    with pl.section_cube():
        skv_dim = Skv2
        skv_tiles = (skv_dim + (TKV - 1)) // TKV

        # MAT tiles
        q_mat = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA0, size=Q_F16)
        k_mat = plm.make_tile(plm.TileType(shape=[TD, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2), addr=MA1, size=KT_F16)
        p_mat = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2, size=P_F16)
        v_mat = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3, size=V_F16)
        # LEFT / RIGHT
        q_left = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA0, size=Q_F16)
        k_right = plm.make_tile(plm.TileType(shape=[TD, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=KT_F16)
        p_left = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA1, size=P_F16)
        v_right = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=V_F16)
        # ACC
        qk_acc = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_F32)
        pv_acc = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_F32)

        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sq_off = core_id * TS

        # Load Q once
        plm.load(q_mat, q, [sq_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.move(q_left, q_mat)

        plm.load(k_mat, k, [0, 0], layout="dn")
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        plm.move(k_right, k_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(qk_acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.l0c_store(qk_acc, [sq_off, 0], [TS, TKV], qk_buf)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
        pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)

        # First KV tile: PV
        plm.load(p_mat, p_buf, [sq_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        plm.move(p_left, p_mat)
        plm.move(v_right, v_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(pv_acc, p_left, v_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.l0c_store(pv_acc, [sq_off, 0], [TS, TD], pv_buf)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)

        # Remaining KV tiles
        for j in pl.range(1, skv_tiles):
            skv_off = j * TKV
            plm.load(k_mat, k, [skv_off, 0], layout="dn")
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            plm.move(k_right, k_mat)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            plm.matmul(qk_acc, q_left, k_right)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.l0c_store(qk_acc, [sq_off, skv_off], [TS, TKV], qk_buf)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)

            plm.load(p_mat, p_buf, [sq_off, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            plm.load(v_mat, v, [skv_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            plm.move(p_left, p_mat)
            plm.move(v_right, v_mat)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            plm.matmul(pv_acc, p_left, v_right)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.l0c_store(pv_acc, [sq_off, 0], [TS, TD], pv_buf)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        skv_dim = Skv2
        skv_tiles = (skv_dim + (TKV - 1)) // TKV

        # VEC tiles: sub-block processes [TS_HALF, TKV] for QK, [TS_HALF, TD] for O
        qk_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_KV)
        tmp_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)
        p_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)

        # Reduce tiles: ColMajor [64,1] for TROWMAX/TROWSUM/TROWEXPAND* ops
        reduce_dst = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA3, size=VB_RED)
        global_max = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA4, size=VB_RED)
        global_sum = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA5, size=VB_RED)
        exp_corr = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA6, size=VB_RED)

        # RowMajor [1,64] aliases at same addresses — for element-wise ops (via TRESHAPE)
        reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)
        global_max_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA4, size=VB_RED)
        global_sum_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA5, size=VB_RED)
        exp_corr_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA6, size=VB_RED)

        running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
        pv_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
        o_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)

        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sq_off = core_id * TS
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * TS_HALF

        # First KV: FlashSoftmax INIT
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
        plm.load(qk_vec, qk_buf, [sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.row_max(reduce_dst, qk_vec, tmp_vec)
        plm.muls(global_max, reduce_dst, 1.0)
        plm.row_expand_sub(tmp_vec, qk_vec, global_max)
        plm.muls(tmp_vec, tmp_vec, SCALE)
        plm.exp(qk_vec, tmp_vec)
        plm.row_sum(reduce_dst, qk_vec, tmp_vec)
        plm.muls(global_sum, reduce_dst, 1.0)
        plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(p_buf, p_f16, [sq_off + row_off, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
        plm.load(running_o, pv_buf, [sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        # Remaining KV: FlashSoftmax UPDATE + GlobalUpdate
        for j in pl.range(1, skv_tiles):
            skv_off = j * TKV
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
            plm.load(qk_vec, qk_buf, [sq_off + row_off, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

            # New tile max → update global_max, compute correction
            # TROWMAX outputs to CM; use RM aliases for element-wise ops (same memory)
            plm.row_max(reduce_dst, qk_vec, tmp_vec)
            plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
            # Correction = exp(SCALE * (old_max - new_max))
            plm.sub(exp_corr_rm, global_max_rm, reduce_dst_rm)
            plm.muls(exp_corr_rm, exp_corr_rm, SCALE)
            plm.exp(exp_corr_rm, exp_corr_rm)
            # Apply correction to running state
            plm.mul(global_sum_rm, global_sum_rm, exp_corr_rm)
            plm.row_expand_mul(running_o, running_o, exp_corr)
            # Update global_max = new_max (CM TMULS copy)
            plm.muls(global_max, reduce_dst, 1.0)
            # Softmax on new QK tile
            plm.row_expand_sub(tmp_vec, qk_vec, global_max)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec, tmp_vec)
            plm.row_sum(reduce_dst, qk_vec, tmp_vec)
            plm.add(global_sum_rm, global_sum_rm, reduce_dst_rm)

            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(p_buf, p_f16, [sq_off + row_off, skv_off])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
            plm.load(pv_vec, pv_buf, [sq_off + row_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.add(running_o, running_o, pv_vec)

        plm.row_expand_div(running_o, running_o, global_sum)
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


def test_fa_k():
    compiled = fe.compile(fa_k_kernel, arch="a3")
    print("compiled:", compiled.lib_path)
    device = "npu:1"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d in [(128, 128, 64)]:
        print(f"\nFA-K ({sq},{skv},{d})")
        q = torch.rand((sq, d), device=device, dtype=torch.float16)
        k = torch.rand((skv, d), device=device, dtype=torch.float16)
        v = torch.rand((skv, d), device=device, dtype=torch.float16)
        o = torch.empty((sq, d), device=device, dtype=torch.float16)
        qk_buf = torch.zeros((sq, skv), device=device, dtype=torch.float32)
        p_buf = torch.zeros((sq, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((sq, d), device=device, dtype=torch.float32)
        fe.launch(None, 1, compiled, q, k, v, o, qk_buf, p_buf, pv_buf)
        torch.npu.synchronize()
        o_ref = flash_attention_ref(q, k, v, d)
        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        print("  PASS")


if __name__ == "__main__":
    print("FA DN layout for K, TILE 128x128x64")
    print("=" * 60)
    test_fa_k()
    print("\nAll FlashAttention tests passed!")
