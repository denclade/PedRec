import numpy as np
import torch

from pedrec.networks.net_pedrec.ehpi_stgcn import EhpiStGcn, get_ehpi_partitioned_adjacency, NUM_JOINTS


def test_adjacency_partitions():
    adjacency = get_ehpi_partitioned_adjacency()
    assert adjacency.shape == (3, NUM_JOINTS, NUM_JOINTS)
    assert np.allclose(adjacency[0], np.eye(NUM_JOINTS))
    rows = adjacency[1:].sum(axis=0).sum(axis=1)
    assert np.all((np.abs(rows - 2) < 1e-6) | (np.abs(rows - 1) < 1e-6))  # in-degree normalized per partition
    # every edge goes either towards or away from the hip, never both
    assert not np.any((adjacency[1] > 0) & (adjacency[2] > 0))


def test_forward_and_backward_on_ehpi_input():
    torch.manual_seed(0)
    net = EhpiStGcn(num_actions=20)
    for frames in (32, 64):
        ehpi = torch.randn(4, 3, 32, frames)  # EHPI image: 26 joints + padding rows
        out = net(ehpi)
        assert out.shape == (4, 20)
    out.sum().backward()
    assert net.blocks[0].gcn.adaptive.grad is not None
    assert sum(p.numel() for p in net.parameters()) < 3_000_000
