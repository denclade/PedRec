import logging

import numpy as np
import torch
import torch.nn as nn


class YoloLayer(nn.Module):
    """
    Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    """

    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out
        self._decode_cache = {}

    def forward(self, output, target=None):
        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        # return yolo_forward(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),
        #                     scale_x_y=self.scale_x_y)

        return self.yolo_forward_alternative(output, self.thresh, self.num_classes, masked_anchors,
                                             len(self.anchor_mask))

    def _get_decode_tensors(self, H: int, W: int, anchors, num_anchors: int, device, dtype):
        """
        Grid offsets, anchors and normalization factors for a feature map size; cached because they only depend on
        the input resolution (previously rebuilt from numpy and copied to the device on every forward pass).
        """
        key = (H, W, tuple(anchors), num_anchors, str(device), dtype)
        if key not in self._decode_cache:
            # grid_x[h * W + w] = w, grid_y[h * W + w] = h
            grid_x = torch.arange(W, device=device, dtype=dtype).repeat(H).view(1, 1, H * W)
            grid_y = torch.arange(H, device=device, dtype=dtype).repeat_interleave(W).view(1, 1, H * W)
            anchor_tensor = torch.tensor(anchors, device=device, dtype=dtype).view(1, num_anchors, 2, 1)
            normal_tensor = torch.tensor([1.0 / W, 1.0 / H, 1.0 / W, 1.0 / H], device=device, dtype=dtype).view(1, 1, 4)
            self._decode_cache[key] = (grid_x, grid_y, anchor_tensor, normal_tensor)
        return self._decode_cache[key]

    def yolo_forward_alternative(self, output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                                 validation=False):
        # Slice the second dimension (channel) of output into:
        # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
        batch = output.size(0)
        H = output.size(2)
        W = output.size(3)
        grid_x, grid_y, anchor_tensor, normal_tensor = self._get_decode_tensors(H, W, anchors, num_anchors,
                                                                                output.device, output.dtype)

        output = output.view(batch, num_anchors, 5 + num_classes, H * W)
        # Shape: [batch, num_anchors, 2, H * W]
        bxy = torch.sigmoid(output[:, :, 0:2])
        bwh = torch.exp(output[:, :, 2:4])
        # Shape: [batch, num_anchors * H * W, 1]
        det_confs = torch.sigmoid(output[:, :, 4]).reshape(batch, num_anchors * H * W, 1)
        # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
        cls_confs = output[:, :, 5:].permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)
        cls_confs = torch.softmax(cls_confs, dim=2)

        # Apply C-x, C-y, P-w, P-h
        bxy = torch.stack((bxy[:, :, 0] + grid_x, bxy[:, :, 1] + grid_y), dim=2)
        bwh = bwh * anchor_tensor

        # Shape: [batch, num_anchors, 4, H * W] --> [batch, num_anchors * H * W, 4]
        boxes = torch.cat((bxy, bwh), dim=2).permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, 4)
        boxes = boxes * normal_tensor

        # boxes: [batch, num_anchors * H * W, 4] (center x, center y, width, height; normalized to [0, 1])
        # confs: [batch, num_anchors * H * W, num_classes]
        return boxes, cls_confs * det_confs

    def yolo_forward(self, output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                     validation=False):
        # Output would be invalid if it does not satisfy this assert
        # assert (output.size(1) == (5 + num_classes) * num_anchors)

        # print(output.size())

        # Slice the second dimension (channel) of output into:
        # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
        # And then into
        # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
        batch = output.size(0)
        H = output.size(2)
        W = output.size(3)

        bxy_list = []
        bwh_list = []
        det_confs_list = []
        cls_confs_list = []

        for i in range(num_anchors):
            begin = i * (5 + num_classes)
            end = (i + 1) * (5 + num_classes)

            bxy_list.append(output[:, begin: begin + 2])
            bwh_list.append(output[:, begin + 2: begin + 4])
            det_confs_list.append(output[:, begin + 4: begin + 5])
            cls_confs_list.append(output[:, begin + 5: end])

        # Shape: [batch, num_anchors * 2, H, W]
        bxy = torch.cat(bxy_list, dim=1)
        # Shape: [batch, num_anchors * 2, H, W]
        bwh = torch.cat(bwh_list, dim=1)

        # Shape: [batch, num_anchors, H, W]
        det_confs = torch.cat(det_confs_list, dim=1)
        # Shape: [batch, num_anchors * H * W]
        det_confs = det_confs.view(batch, num_anchors * H * W)

        # Shape: [batch, num_anchors * num_classes, H, W]
        cls_confs = torch.cat(cls_confs_list, dim=1)
        # Shape: [batch, num_anchors, num_classes, H * W]
        cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
        # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
        cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

        # Apply sigmoid(), exp() and softmax() to slices
        #
        bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
        bwh = torch.exp(bwh)
        det_confs = torch.sigmoid(det_confs)
        cls_confs = torch.nn.Softmax(dim=2)(cls_confs)

        # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
        grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0),
                                axis=0)
        grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0),
                                axis=0)
        # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
        # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

        anchor_w = []
        anchor_h = []
        for i in range(num_anchors):
            anchor_w.append(anchors[i * 2])
            anchor_h.append(anchors[i * 2 + 1])

        device = None
        cuda_check = output.is_cuda
        if cuda_check:
            device = output.get_device()

        bx_list = []
        by_list = []
        bw_list = []
        bh_list = []

        # Apply C-x, C-y, P-w, P-h
        for i in range(num_anchors):
            ii = i * 2
            # Shape: [batch, 1, H, W]
            bx = bxy[:, ii: ii + 1] + torch.tensor(grid_x, device=device,
                                                   dtype=torch.float32)  # grid_x.to(device=device, dtype=torch.float32)
            # Shape: [batch, 1, H, W]
            by = bxy[:, ii + 1: ii + 2] + torch.tensor(grid_y, device=device,
                                                       dtype=torch.float32)  # grid_y.to(device=device, dtype=torch.float32)
            # Shape: [batch, 1, H, W]
            bw = bwh[:, ii: ii + 1] * anchor_w[i]
            # Shape: [batch, 1, H, W]
            bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

            bx_list.append(bx)
            by_list.append(by)
            bw_list.append(bw)
            bh_list.append(bh)

        ########################################
        #   Figure out bboxes from slices     #
        ########################################

        # Shape: [batch, num_anchors, H, W]
        bx = torch.cat(bx_list, dim=1)
        # Shape: [batch, num_anchors, H, W]
        by = torch.cat(by_list, dim=1)
        # Shape: [batch, num_anchors, H, W]
        bw = torch.cat(bw_list, dim=1)
        # Shape: [batch, num_anchors, H, W]
        bh = torch.cat(bh_list, dim=1)

        # Shape: [batch, 2 * num_anchors, H, W]
        bx_bw = torch.cat((bx, bw), dim=1)
        # Shape: [batch, 2 * num_anchors, H, W]
        by_bh = torch.cat((by, bh), dim=1)

        # normalize coordinates to [0, 1]
        bx_bw /= W
        by_bh /= H

        # Shape: [batch, num_anchors * H * W, 1]
        bx = bx_bw[:, :num_anchors].view(batch, num_anchors * H * W, 1)
        by = by_bh[:, :num_anchors].view(batch, num_anchors * H * W, 1)
        bw = bx_bw[:, num_anchors:].view(batch, num_anchors * H * W, 1)
        bh = by_bh[:, num_anchors:].view(batch, num_anchors * H * W, 1)

        # Shape: [batch, num_anchors * h * w, 4]
        boxes = torch.cat((bx, by, bw, bh), dim=2).view(batch, num_anchors * H * W, 4)

        # boxes:     [batch, num_anchors * H * W, num_classes, 4]
        # cls_confs: [batch, num_anchors * H * W, num_classes]
        # det_confs: [batch, num_anchors * H * W]

        det_confs = det_confs.view(batch, num_anchors * H * W, 1)
        confs = cls_confs * det_confs

        # boxes: [batch, num_anchors * H * W, 4]
        # confs: [batch, num_anchors * H * W, num_classes]

        return boxes, confs
