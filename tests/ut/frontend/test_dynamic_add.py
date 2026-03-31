# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic add tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops."""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
N = pl.DynVar('N')

@fe.kernel
def dynamic_add_kernel(
    x: pl.Tensor[[M, N], pl.FP16],
    y: pl.Tensor[[M, N], pl.FP16],
    z: pl.Tensor[[M, N], pl.FP16]
) -> pl.Tensor[[M, N], pl.FP16]:
    tile_type_a = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1,-1],
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    
    tile_type_b = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1,-1],
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    
    tile_type_c = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1,-1],
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    
    with pl.section_vector():
        M_dim = pl.tensor.dim(x, 0)
        N_dim = pl.tensor.dim(x, 1)
        
        for i in pl.range(0, M_dim, 64):
            for j in pl.range(0, N_dim, 128):
                # Barrier at start: ensure previous iteration's store is complete
                pl.system.bar_all()
                m_size = pl.min(M_dim - i, 64)
                n_size = pl.min(N_dim - j, 128)
                plm.set_validshape(tile_a, m_size, n_size)
                plm.load(tile_a, x, [i, j])
                plm.set_validshape(tile_b, m_size, n_size)
                plm.load(tile_b, y, [i, j])
                # Sync: wait for load (MTE2) to complete before compute (V)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                plm.set_validshape(tile_c, m_size, n_size)
                plm.add(tile_c, tile_a, tile_b)
                # Sync: wait for compute (V) to complete before store (MTE3)
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                plm.store(z, tile_c, [i, j])
    return z


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_dynamic_add():
    compiled_lib = fe.compile(dynamic_add_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:7"
    torch.npu.set_device(device)

    shapes = [
        [64, 128],
        [128, 256],
        [96, 192],
        [16, 16],
        [777, 666],
    ]
    torch.manual_seed(0)
    dtype = torch.float16

    for shape in shapes:

        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        fe.launch(None, 1, compiled_lib, x, y, z)
        torch.npu.synchronize()

        print("***********npu output***********")
        print(z.shape, z.dtype)
        print(z)
        z_ref = x + y
        print("***********golden output***********")
        print(z_ref.shape, z_ref.dtype)
        print(z_ref)
        torch.testing.assert_close(z, z_ref)
        print("result equal!")


if __name__ == "__main__":
    test_dynamic_add()
    print("\nAll tests passed!")