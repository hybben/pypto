# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for pl.getval and pl.setval functions.

Tests that getval and setval work correctly with Tile operations.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel: get first element and set it to another position
@fe.kernel
def tile_getval_setval_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    pl.system.bar_all()
    tensor_value = pl.tensor.getval(a, 100)
    pl.tensor.setval(a, 101, tensor_value)
    pl.system.bar_all()

    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                              addr=0x0000, size=16384)
    with pl.section_vector():
        plm.load(tile_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        value = pl.block.getval(tile_a, 0)
        tile_a = pl.block.setval(tile_a, 1, value)
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(a, tile_a, [0, 0])
    return a


# ---------------------------------------------------------------------------
# Test functions (run with: python test_getval_setval.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_tile_getval_setval():
    device = "npu:5"
    torch.npu.set_device(device)

    shape = [64, 128]
    torch.manual_seed(0)
    dtype = torch.float16
    a = torch.rand(shape, device=device, dtype=dtype)
    
    print("Input tensor:")
    print("a[0,0] =", a[0, 0].item())
    print("a[0,1] =", a[0, 1].item())
    print("a[0,100] =", a[0, 100].item())
    print("a[0,101] =", a[0, 101].item())

    compiled_lib = fe.compile(tile_getval_setval_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 1, compiled_lib, a)
    
    torch.npu.synchronize()
    
    print("\n***********npu output***********")
    print("\n***********block output***********")
    print("a[0,0] =", a[0, 0].item())
    print("a[0,1] =", a[0, 1].item())
    print("\n***********tensor output***********")
    print("a[0,100] =", a[0, 100].item())
    print("a[0,101] =", a[0, 101].item())
    
    block_expected_value = a[0, 0].item()
    block_actual_value = a[0, 1].item()
    
    tensor_expected_value = a[0, 100].item()
    tensor_actual_value = a[0, 101].item()

    print(f"\nblock_expected a[0,0] = {block_expected_value}")
    print(f"block_actual a[0,1] = {block_actual_value}")
    
    print(f"\ntensor_expected a[0,100] = {tensor_expected_value}")
    print(f"tensor_actual a[0,101] = {tensor_actual_value}")

    assert abs(block_expected_value - block_actual_value) < 1e-3, f"block getval/setval failed: expected {block_expected_value}, got {block_actual_value}"
    assert abs(tensor_expected_value - tensor_actual_value) < 1e-3, f"tensor getval/setval failed: expected {tensor_expected_value}, got {tensor_actual_value}"
    print("result equal!")


if __name__ == "__main__":
    test_tile_getval_setval()
    print("\nAll tests passed!")
