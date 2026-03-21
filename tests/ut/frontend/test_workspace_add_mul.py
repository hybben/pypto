# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for (a + b) * c with workspace (GM intermediate buffer).

Two kernels:
1. workspace_add_mul_kernel  — plain workspace (no double buffer)
2. workspace_add_mul_db_kernel — workspace double buffer (ping/pong)

The workspace double-buffer kernel exercises the indirect-select pattern for
TensorType: scf.if yields a !pto.ptr<f16>, then pto.make_tensor_view rebuilds
the tensor_view after the scf.if block.  This mirrors the TileType
indirect-select (which yields i64 addr).

Event-id rule: each (set_pipe, wait_pipe) combination has its own independent
event_id namespace (0–7), so the same value 0 (or 1) may be reused across
different pipe combinations without conflict.
"""

import pytest
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
N = pl.DynVar('N')

# ---------------------------------------------------------------------------
# Kernel 1: workspace add-mul (no double buffer)
# ---------------------------------------------------------------------------
# Tile layout:
#   UB: tile_a [64,128], tile_b [64,128], tile_ab [64,128], tile_c [64,128], tile_z [64,128]
# Sync sequence per (i,j) block (event_ids are per pipe-combo, all reuse 0):
#   pre-loop: sync_src(MTE3→MTE2, 0)      # prime first iteration
#   loop top: sync_dst(MTE3→MTE2, 0)      # wait last z-store to finish
#   load tile_a, tile_b
#   sync_src/dst(MTE2→V, 0)               # loads done, compute can start
#   add tile_ab = tile_a + tile_b
#   sync_src/dst(V→MTE3, 0)               # add done, store ws can start
#   store ws_view ← tile_ab              # UB→GM
#   sync_src/dst(MTE3→MTE2, 0)            # ws stored, MTE2 can load back
#   load tile_ab ← ws_view; load tile_c
#   sync_src/dst(MTE2→V, 0)
#   mul tile_z = tile_ab * tile_c
#   sync_src/dst(V→MTE3, 0)
#   sync_src(MTE3→MTE2, 0)               # notify next loop's top
#   store z ← tile_z

@fe.kernel
def workspace_add_mul_kernel(
    a: pl.Tensor[[M, N], pl.FP16],
    b: pl.Tensor[[M, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP16],
    workspace: pl.Ptr[pl.FP16],
    z: pl.Tensor[[M, N], pl.FP16],
) -> pl.Tensor[[M, N], pl.FP16]:
    tile_a = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x0000, size=16384,
    )
    tile_b = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x4000, size=16384,
    )
    # tile_ab is reused: first holds a+b result, later holds the ws read-back
    tile_ab = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x8000, size=16384,
    )
    tile_c = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0xC000, size=16384,
    )
    tile_z = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x10000, size=16384,
    )

    # Workspace tensor_view: one [64,128] FP16 block in GM
    ws_view: pl.Tensor[[64, 128], pl.FP16] = pl.make_tensor(workspace, [64, 128], [128, 1])

    with pl.section_vector():
        M_dim = pl.tensor.dim(a, 0)
        N_dim = pl.tensor.dim(a, 1)

        # Prime first iteration: MTE3→MTE2 event_id=0 starts as "done"
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

        for i in pl.range(0, M_dim, 64):
            for j in pl.range(0, N_dim, 128):
                # Wait for previous z-store (MTE3) to finish before MTE2 can write UB
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

                plm.load(tile_a, a, [i, j])
                plm.load(tile_b, b, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                plm.add(tile_ab, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)

                # Store a+b result to workspace (UB → GM)
                plm.store(ws_view, tile_ab, [0, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

                # Load back from workspace and load c (GM → UB)
                plm.load(tile_ab, ws_view, [0, 0])
                plm.load(tile_c, c, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                plm.mul(tile_z, tile_ab, tile_c)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)

                # Notify next iteration that z-store is about to begin
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

                plm.store(z, tile_z, [i, j])

    return z


# ---------------------------------------------------------------------------
# Kernel 2: workspace add-mul with double buffer (ping/pong)
# ---------------------------------------------------------------------------
# Two workspace slices in GM: ws_ping [64,128] at offset 0,
#                              ws_pong [64,128] at offset 64*128 elements.
# buf_idx cycles 0/1 each (i,j) iteration (incremented at the loop bottom).
# ws_buf = (ws_ping, ws_pong); ws_cur = ws_buf[buf_idx]
# event_ids = (0, 1); each pipe-combo uses event_ids[buf_idx] independently.
#
# Sync sequence (each pipe-combo's event_id namespace is independent):
#   pre-loop: sync_src(MTE3→MTE2, 0); sync_src(MTE3→MTE2, 1)  # prime both slots
#   loop top: sync_dst(MTE3→MTE2, event_ids[buf_idx])
#   load tile_a, tile_b
#   sync_src/dst(MTE2→V, event_ids[buf_idx])
#   add tile_ab = tile_a + tile_b
#   sync_src/dst(V→MTE3, event_ids[buf_idx])
#   store ws_cur ← tile_ab
#   sync_src/dst(MTE3→MTE2, event_ids[buf_idx])
#   load tile_ab ← ws_cur; load tile_c
#   sync_src/dst(MTE2→V, event_ids[buf_idx])
#   mul tile_z = tile_ab * tile_c
#   sync_src/dst(V→MTE3, event_ids[buf_idx])
#   sync_src(MTE3→MTE2, event_ids[buf_idx])   # notify next same-slot iteration
#   store z ← tile_z
#   buf_idx = (buf_idx + 1) % 2

@fe.kernel
def workspace_add_mul_db_kernel(
    a: pl.Tensor[[M, N], pl.FP16],
    b: pl.Tensor[[M, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP16],
    workspace: pl.Ptr[pl.FP16],
    z: pl.Tensor[[M, N], pl.FP16],
) -> pl.Tensor[[M, N], pl.FP16]:
    tile_a = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x0000, size=16384,
    )
    tile_b = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x4000, size=16384,
    )
    tile_ab = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x8000, size=16384,
    )
    tile_c = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0xC000, size=16384,
    )
    tile_z = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addr=0x10000, size=16384,
    )

    # Two workspace slices: ping at offset 0, pong at offset 64*128 elements
    ws_ping_ptr: pl.Ptr[pl.FP16] = pl.addptr(workspace, 0)
    ws_pong_ptr: pl.Ptr[pl.FP16] = pl.addptr(ws_ping_ptr, 64 * 128)
    ws_ping = pl.make_tensor(ws_ping_ptr, [64, 128], [128, 1])
    ws_pong = pl.make_tensor(ws_pong_ptr, [64, 128], [128, 1])
    ws_buf = (ws_ping, ws_pong)
    event_ids = (0, 1)

    with pl.section_vector():
        M_dim = pl.tensor.dim(a, 0)
        N_dim = pl.tensor.dim(a, 1)

        # Prime both slots so the first two iterations can proceed immediately
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)

        buf_idx = pl.const(0, pl.INDEX)

        for i in pl.range(0, M_dim, 64):
            for j in pl.range(0, N_dim, 128):
                ws_cur: pl.Tensor[[64, 128], pl.FP16] = ws_buf[buf_idx]

                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2,
                                   event_id=event_ids[buf_idx])

                plm.load(tile_a, a, [i, j])
                plm.load(tile_b, b, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V,
                                   event_id=event_ids[buf_idx])
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V,
                                   event_id=event_ids[buf_idx])

                plm.add(tile_ab, tile_a, tile_b)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3,
                                   event_id=event_ids[buf_idx])
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3,
                                   event_id=event_ids[buf_idx])

                # Store a+b to current workspace slot (UB → GM)
                plm.store(ws_cur, tile_ab, [0, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2,
                                   event_id=event_ids[buf_idx])
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2,
                                   event_id=event_ids[buf_idx])

                # Load back from workspace and load c
                plm.load(tile_ab, ws_cur, [0, 0])
                plm.load(tile_c, c, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V,
                                   event_id=event_ids[buf_idx])
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V,
                                   event_id=event_ids[buf_idx])

                plm.mul(tile_z, tile_ab, tile_c)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3,
                                   event_id=event_ids[buf_idx])
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3,
                                   event_id=event_ids[buf_idx])

                # Notify next same-slot iteration that z-store is about to start
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2,
                                   event_id=event_ids[buf_idx])

                plm.store(z, tile_z, [i, j])

                buf_idx = (buf_idx + 1) % 2

    return z


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@fe.jit()
def test_workspace_add_mul():
    compiled_lib = fe.compile(workspace_add_mul_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:7"
    torch.npu.set_device(device)

    shapes = [[64, 128], [128, 256]]
    torch.manual_seed(42)
    dtype = torch.float16

    for shape in shapes:
        M_val, N_val = shape
        a = torch.rand(shape, device=device, dtype=dtype)
        b = torch.rand(shape, device=device, dtype=dtype)
        c = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        # workspace: one [64,128] block = 8192 bytes
        workspace = torch.empty(64 * 128, device=device, dtype=dtype)

        fe.launch(None, 1, compiled_lib, a, b, c, workspace, z)
        torch.npu.synchronize()

        z_ref = (a + b) * c
        print(f"shape={shape}: max_diff={( z - z_ref).abs().max().item():.4f}")
        torch.testing.assert_close(z, z_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


@fe.jit()
def test_workspace_add_mul_db():
    compiled_lib = fe.compile(workspace_add_mul_db_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:7"
    torch.npu.set_device(device)

    shapes = [[64, 128], [128, 256], [1024, 1024]]
    torch.manual_seed(42)
    dtype = torch.float16

    for shape in shapes:
        M_val, N_val = shape
        a = torch.rand(shape, device=device, dtype=dtype)
        b = torch.rand(shape, device=device, dtype=dtype)
        c = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)
        # workspace: two [64,128] blocks for ping/pong = 2 * 64*128 elements
        workspace = torch.empty(2 * 64 * 128, device=device, dtype=dtype)

        fe.launch(None, 1, compiled_lib, a, b, c, workspace, z)
        torch.npu.synchronize()

        z_ref = (a + b) * c
        print(f"shape={shape}: max_diff={(z - z_ref).abs().max().item():.4f}")
        torch.testing.assert_close(z, z_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_workspace_add_mul()
    test_workspace_add_mul_db()
    print("\nAll tests passed!")
