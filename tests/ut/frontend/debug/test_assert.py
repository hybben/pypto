# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual frontend runtime demo for plm.assert_.

Expected behavior:
- With flag=True and all three asserts enabled, the first 64x64 tile passes and
  the second tile fails on `offset == 0`, so output should stop after
  ASSERT_TEST_BEFORE_LAUNCH.
- If the third assert is commented out, output should continue through
  ASSERT_TEST_AFTER_LAUNCH, ASSERT_TEST_BEFORE_CHECK, and finally print equal.
- If flag=False, the first or second assert should fail in the first iteration.
"""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


@fe.kernel
def assert_kernel_runtime(
    x: pl.Tensor[[128, 64], pl.INT32],
    y: pl.Tensor[[128, 64], pl.INT32],
    z: pl.Tensor[[128, 64], pl.INT32],
    flag: pl.Scalar[pl.BOOL],
) -> pl.Tensor[[128, 64], pl.INT32]:
    tile_type = plm.TileType(shape=[64, 64], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
    tile_x = plm.make_tile(tile_type, addr=0x0000, size=16384)
    tile_y = plm.make_tile(tile_type, addr=0x4000, size=16384)
    tile_z = plm.make_tile(tile_type, addr=0x8000, size=16384)

    with pl.section_vector():
        for offset in pl.range(0, 128, 64):
            pl.system.bar_all()
            plm.load(tile_x, x, [offset, 0])
            plm.load(tile_y, y, [offset, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.add(tile_z, tile_x, tile_y)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            plm.assert_(flag)
            plm.assert_(flag, "flag is false")
            plm.assert_(offset == 0, "offset=%d", offset)
            plm.store(z, tile_z, [offset, 0])
            pl.system.bar_all()

    return z


@fe.jit()
def test_assert() -> None:
    print("ASSERT_TEST_BEFORE_COMPILE", flush=True)
    compiled_lib = fe.compile(assert_kernel_runtime, arch="a3")
    print("ASSERT_TEST_AFTER_COMPILE", compiled_lib.lib_path, flush=True)

    device = "npu:0"
    torch.npu.set_device(device)

    base = torch.arange(128 * 64, device=device, dtype=torch.int32).reshape(128, 64)
    x = base + 1
    y = base
    z = torch.zeros_like(x)

    print("ASSERT_TEST_BEFORE_LAUNCH", flush=True)
    fe.launch(None, 1, compiled_lib, x, y, z, False)
    torch.npu.synchronize()
    print("ASSERT_TEST_AFTER_LAUNCH", flush=True)

    print("ASSERT_TEST_BEFORE_CHECK", flush=True)
    torch.testing.assert_close(z, x + y)
    print("equal", flush=True)


if __name__ == "__main__":
    test_assert()
    print("\nAll tests passed!")
