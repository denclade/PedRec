import torch
import torch.nn as nn


class PedRecNetMTLWrapper(nn.Module):
    """
    Network + multi task loss head. The network may run under (bf16 / fp16) autocast, the loss head always runs in
    fp32 (numerically safer, and BCELoss is not allowed under autocast). The returned outputs are fp32.

    In eval mode (validation) the poses are returned in the joint convention of each sample's dataset if the loss head
    has a convention adapter, i.e. they are validated against the labels in the convention they were annotated in.
    ``adapt_eval_outputs = False`` returns the canonical poses instead (what the inference network outputs).
    """

    def __init__(self, model: nn.Module, loss_head: nn.Module):
        super(PedRecNetMTLWrapper, self).__init__()
        self.model = model
        self.loss_head = loss_head
        self.adapt_eval_outputs = True

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            outputs = tuple(o.float() if torch.is_tensor(o) else o for o in outputs)
            loss = self.loss_head(outputs, targets)
            if not self.training and self.adapt_eval_outputs and hasattr(self.loss_head, "adapt_outputs"):
                outputs = self.loss_head.adapt_outputs(outputs, targets)
        return outputs, loss
