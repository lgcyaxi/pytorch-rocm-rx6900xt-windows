# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch import fx
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.post_grad import _decompose_shard_dim_alltoall
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX


class TestDecomposeShardDimAllToAll(TestCase):
    def _make_graph_module(
        self,
        *,
        shape: tuple[int, ...] = (5, 7),
        dtype: torch.dtype = torch.float32,
        gather_dim: int = 0,
        shard_dim: int = 1,
    ) -> fx.GraphModule:
        graph = fx.Graph()
        inp = graph.placeholder("inp")
        inp.meta["val"] = torch.empty(shape, dtype=dtype)
        shard_dim_alltoall = graph.call_function(
            torch.ops._dtensor.shard_dim_alltoall.default,
            args=(inp, gather_dim, shard_dim, "test_pg"),
        )
        graph.output(shard_dim_alltoall)
        return fx.GraphModule({}, graph)

    def _run_pass(
        self,
        gm: fx.GraphModule,
        *,
        group_size: int = 4,
        group_rank: int = 1,
    ) -> None:
        with (
            mock.patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value="resolved_pg",
            ),
            mock.patch(
                "torch.distributed.distributed_c10d._get_group_size_by_name",
                return_value=group_size,
            ),
            mock.patch(
                "torch.distributed.distributed_c10d.get_group_rank",
                return_value=group_rank,
            ),
            mock.patch("torch.distributed.get_rank", return_value=group_rank),
        ):
            _decompose_shard_dim_alltoall(gm)

    def test_decomposes_uneven_shard_dim_alltoall(self) -> None:
        counters.clear()
        gm = self._make_graph_module()

        self._run_pass(gm)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertIn(torch.ops._c10d_functional.wait_tensor.default, targets)

        alltoall_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops._c10d_functional.all_to_all_single.default
        )
        self.assertEqual(alltoall_node.args[1], [2, 2, 2, 2])
        self.assertEqual(alltoall_node.args[2], [2, 2, 2, 1])
        self.assertEqual(alltoall_node.args[3], "test_pg")

        view_shapes = [
            node.args[1]
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.view.default
        ]
        self.assertIn([4, 2, 5], view_shapes)
        self.assertIn([20, 2], view_shapes)
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 1)

    def test_skips_complex_dtype(self) -> None:
        counters.clear()
        gm = self._make_graph_module(dtype=torch.complex64)

        self._run_pass(gm)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertNotIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 0)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
