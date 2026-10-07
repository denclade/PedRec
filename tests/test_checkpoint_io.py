import torch

from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file


def test_plain_and_wrapped_state_dicts(tmp_path):
    net = torch.nn.Linear(3, 2)
    torch.save(net.state_dict(), tmp_path / "plain.pth")
    torch.save({"state_dict": net.state_dict(), "epoch": 3}, tmp_path / "wrapped.pth.tar")
    torch.save({f"module.{k}": v for k, v in net.state_dict().items()}, tmp_path / "ddp.pth")
    for name in ["plain.pth", "wrapped.pth.tar", "ddp.pth"]:
        state = load_state_dict_file(str(tmp_path / name))
        assert set(state.keys()) == {"weight", "bias"}
        torch.nn.Linear(3, 2).load_state_dict(state)
