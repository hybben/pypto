# FlashAttention Test Status

## Test File
`tests/ut/frontend/test_fa.py`

## Current Status: PASSING

`fa_k_kernel` passes multi-core precision checks with TD=128.

| Kernel | Shapes | Cores | Max Diff | Tolerance |
|--------|--------|-------|----------|-----------|
| fa_k_kernel | (128, 256, 128) | 1 | 0.0005 | atol=1e-3 |
| fa_k_kernel | (512, 512, 128) | 2 | 0.0005 | atol=1e-3 |
| fa_k_kernel | (8192, 8192, 128) | 24 | ~0.001 | atol=1e-3 |

## Root Causes Fixed (This Session)

### 1. Missing pipe_barrier(PIPE_V) — THE MAIN BUG

On Ascend a2/a3, the vector pipeline does NOT enforce data dependencies between TVEC instructions. Without `pl.system.bar_v()` between dependent ops, later ops read **stale data** from the vector unit → silently wrong softmax correction → output matched "no correction" baseline exactly.

**Symptom**: Single-tile test PASSED (no correction needed), multi-tile test FAILED with large error (0.79). Debug showed output exactly matched `(PV_0 + PV_1) / (sum0 + sum1)` without the online softmax correction factor.

**Fix**: Add `pl.system.bar_v()` after every TVEC op whose output is consumed by the next op (RAW dependency). Pattern from reference `pto_macro_fa_softmax.hpp`:

```python
# INIT:
plm.row_max(reduce_dst, qk_vec, tmp_vec)
pl.system.bar_v()                          # ← TROWMAX → TROWEXPANDSUB
plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
plm.muls(global_max, reduce_dst, 1.0)     # independent copy, no barrier needed
plm.muls(tmp_vec, tmp_vec, SCALE)
plm.exp(qk_vec, tmp_vec)
pl.system.bar_v()                          # ← TEXP → TROWSUM
plm.row_sum(reduce_dst, qk_vec, tmp_vec)
pl.system.bar_v()                          # ← TROWSUM → TMULS(copy)
plm.muls(global_sum, reduce_dst, 1.0)

# UPDATE (6 barriers):
plm.row_max(...)
pl.system.bar_v()      # → TMAX
plm.maximum(...)
pl.system.bar_v()      # → TSUB
plm.sub(...)
pl.system.bar_v()      # → TMULS(copy)
plm.muls(global_max_rm, reduce_dst_rm, 1.0)
pl.system.bar_v()      # → TROWEXPANDSUB
# interleaved independent ops (no barrier)...
plm.exp(exp_corr_rm, ...) ; plm.exp(qk_vec, ...) ; plm.cast(...)
pl.system.bar_v()      # → TMUL, TROWSUM
plm.mul(...) ; plm.row_sum(...)
pl.system.bar_v()      # → TADD
plm.add(...)
```

**Key insight**: Independent ops on different tiles can run without barriers. E.g., `TMULS` on [1,64] exp_corr and `TMULS` on [64,128] tmp_vec operate on different data → no barrier needed between them.

### 2. Wrong pv_buf Multi-Core Offset (core_id * 24)

**Bug**: `plm.store(pv_buf, acc, [core_id * 24 + q_mat_idx * TS, 0])` — the `24` was wrong. Each core needs `2 * TS = 256` rows in pv_buf (double-buffered Q tile slots).

**Fix**: `PV_CORE_STRIDE = 2 * TS` → `core_id * PV_CORE_STRIDE + q_mat_idx * TS`

### 3. Vector core_id in Mix Mode (section_cube/section_vector)

When using `pl.section_cube()` / `pl.section_vector()`, `get_block_idx()` returns the same core index (0..num_cores-1) in **both** sections. No `// 2` needed. `get_subblock_idx()` differentiates the two vector sub-blocks (0 or 1).

## Historical Fixes (Previous Sessions)

### Tile Layout Parameters (CRITICAL)
Cube matmul tiles auto-fill correct `blayout`/`slayout` via `TileType` Python wrapper (see `manual_ops.py` `_REQUIRED_LAYOUTS`). No need to specify manually unless overriding (e.g., K DN-layout load uses `blayout=1, slayout=2`).

### Missing wait_cross_core
`wait_cross_core(M, P_READY)` is mandatory in Cube section before loading P written by Vector.

## PLM Parser Convention (CRITICAL)

The AST parser for `plm.*` manual ops: **first positional arg = output tile**. Parser moves arg[0] to last position in IR call.

```python
plm.sub(OUT, lhs, rhs)     # → manual.sub(lhs, rhs, OUT)
plm.muls(OUT, tile, scalar) # → manual.muls(tile, scalar, OUT)
plm.matmul(OUT, left, right) # → manual.matmul(left, right, OUT)
plm.row_max(OUT, tile, tmp)  # → manual.row_max(tile, tmp, OUT)
plm.cast(OUT, tile, target_type=pl.FP16, mode="round")
```

**Exceptions** (parsed as block ops, NO reordering):
- `plm.make_tile(tile_type, addr=X, size=Y)` → `block.make_tile(...)`

## VEC Reduce Tile Pattern

Element-wise TVEC ops (TMAX, TSUB, TEXP, TMUL, TADD) require **RowMajor** blayout. Row-reduce/expand ops (TROWMAX, TROWSUM, TROWEXPAND*) use **ColMajor** [64,1]. Solution: alias same address:
```python
reduce_dst    = plm.make_tile(..., shape=[64, 1], blayout=2, addr=ADDR)  # ColMajor
reduce_dst_rm = plm.make_tile(..., shape=[1, 64], addr=ADDR)             # RowMajor alias
```

## Compilation Flow
```bash
source compile.sh  # sets up environment
python3 tests/ut/frontend/test_fa.py
```

`fe.compile(kernel, arch="a3")` → PTOCodegen → `.pto` MLIR → `ptoas --enable-insert-sync --pto-level=level3 --pto-arch=a3` → `.cpp` → `bisheng` → `.so`

## Known Issues
1. `plm.muls(tile, tile, scalar)` — parser reorders first arg as output. Must write `plm.muls(OUT, input_tile, scalar)`.
2. `math.sqrt()` not supported in kernel body — precompute as module-level constant.
3. `pl.tensor.dim()` inline in expressions produces missing SSA operand — always assign to variable first.
4. `plm.cast` needs explicit `mode="round"` kwarg; default empty string causes codegen error.
5. `get_block_idx()` returns i64; needs `pl.block.index_cast()` for index arithmetic.
