import torch

def initialize_weights_with_same_name_and_shape(net, p2d_coco_only_weights_path: str, prefix: str = None):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        # if name == "loss_head.sigmas":
        #     # init p2d and p3d head sigmas
        #     net_weights[name][:param.shape[0]] = param[:]
        #     continue
        fixed_name = name.replace("conv_transpose_2d", "conv_transpose_shared")
        if prefix is not None:
            fixed_name = fixed_name.replace(prefix, "")
        if name in net_weights:
            net_weights[name] = param
        elif fixed_name in net_weights:
            net_weights[fixed_name] = param
        else:
            print(f"No weight found for '{name}'")
    net.load_state_dict(net_weights)


def initialize_p2d_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name in net_weights:
            net_weights[name] = param
        else:
            print(f"No weight found for '{name}'")
    net.load_state_dict(net_weights)


def initialize_p3d_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        net_name = name.replace("2d", "3d")
        if net_name in net_weights:
            net_weights[net_name] = param
        else:
            print(f"No weight found for '{name}', tried mapping '{net_name}'")
    net.load_state_dict(net_weights)


def initialize_p2d_p3d_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name in net_weights:
            net_weights[name] = param
        # try to initialize 3d from 2d
        net_name = name.replace("2d", "3d")
        if net_name in net_weights:
            net_weights[net_name] = param
        else:
            print(f"No weight found for '{name}', tried mapping '{net_name}'")
    net.load_state_dict(net_weights)


def initialize_p2d_p3d_shared_conv_from_p2d_coco(net, p2d_coco_only_weights_path: str):
    p2d_coco_only_weights = torch.load(p2d_coco_only_weights_path)
    net_weights = net.state_dict()
    for name, param in p2d_coco_only_weights.items():
        if name in net_weights:
            net_weights[name] = param
        else:
            # try to initialize 3d from 2d
            net_name = name.replace("2d", "shared")
            if net_name in net_weights:
                net_weights[net_name] = param
            else:
                net_name = name.replace("shared", "3d")
                if net_name in net_weights:
                    net_weights[net_name] = param
                else:
                    print(f"No weight found for '{name}', tried mapping '{net_name}'")
    net.load_state_dict(net_weights)

def initialize_pose_resnet(net, pose_resnet_weights_path: str):
    """
    Initializes a PedRecNet (wrapped in the MTL wrapper, thus the ``model.`` prefix) from the Microsoft
    "Simple Baselines" pose-resnet weights: backbone + shared deconvs + 2D head.
    """
    pose_resnet_state_dict = torch.load(pose_resnet_weights_path)
    net_weights = net.state_dict()
    for name, param in pose_resnet_state_dict.items():
        if name.startswith("final"):
            net_name = name.replace("final_layer.", "model.head_pose_2d.pose_heatmap_layer.")
        elif name.startswith("deconv_layers.6") or name.startswith("deconv_layers.7"):
            net_name = f"model.head_pose_2d.{name.replace('deconv_layers', 'deconv_head').replace('6', '0').replace('7', '1')}"
        elif name.startswith("deconv"):
            net_name = f"model.conv_transpose_shared.{name}"
        else:
            net_name = f"model.feature_extractor.{name}"
        if net_name in net_weights:
            if net_weights[net_name].shape != param.shape and len(param.shape) == 1:
                net_weights[net_name][:17] = param
            elif net_weights[net_name].shape != param.shape and len(param.shape) == 4:
                net_weights[net_name][:17, :, :, :] = param
            elif net_weights[net_name].shape != param.shape:
                print(f"Shape mismatch in {net_name}: {net_weights[net_name].shape} <-> {param.shape}")
            else:
                net_weights[net_name] = param
        else:
            print(f"Skipped: {name}, tried: {net_name}")
    net.load_state_dict(net_weights)
