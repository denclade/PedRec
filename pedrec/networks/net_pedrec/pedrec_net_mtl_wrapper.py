import torch
import torch.nn as nn


class PedRecNetMTLWrapper(nn.Module):
    """
    Network + multi task loss head. The network may run under (bf16 / fp16) autocast, the loss head always runs in
    fp32 (numerically safer, and BCELoss is not allowed under autocast). The returned outputs are fp32.
    """

    def __init__(self, model: nn.Module, loss_head: nn.Module):
        super(PedRecNetMTLWrapper, self).__init__()
        self.model = model
        self.loss_head = loss_head

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            outputs = tuple(o.float() if torch.is_tensor(o) else o for o in outputs)
            loss = self.loss_head(outputs, targets)
        return outputs, loss
