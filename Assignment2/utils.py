import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

def deprocess_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = x.squeeze(0)  # (3, H, W)

    # De-normalize from [-1, 1] → [0, 1]
    x = (x * 0.5) + 0.5  # Now in [0, 1]
    x = np.clip(x, 0, 1)

    # Convert to uint8 for saving, and transpose to (H, W, C)
    x = (x * 255).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))  # (H, W, C)

    return x

def normalize(x, eps=1e-8):
    return x / (x.pow(2).mean().sqrt() + eps)

def init_coverage_tables(model1, model2):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    return model_layer_dict1, model_layer_dict2

def init_dict(model, model_layer_dict):
    # Conv 레이어만 대상으로 처리
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            out_channels = module.out_channels
            for idx in range(out_channels):
                model_layer_dict[(name, idx)] = False

def update_coverage(input_data, model, model_layer_dict, threshold=0):
    model.eval()
    
    # hook을 저장할 딕셔너리
    activation = {}
    
    def get_hook(name):
        def hook(module, input, output):
            activation[name] = output.detach().cpu()
        return hook
    
    # Conv2d 레이어에 hook 등록
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    with torch.no_grad():
        _ = model(input_data)

    # hook 해제
    for hook in hooks:
        hook.remove()

    # 각 레이어별 활성값 처리
    for layer_name, act in activation.items():
        # act: shape = (1, C, H, W)
        # act.squeeze(0): shape = (C, H, W)
        act = act.squeeze(0)
        act = act.numpy()
        act = scale(act)
        
        for ch in range(act.shape[0]):
            if np.mean(act[ch]) > threshold and not model_layer_dict[(layer_name, ch)]:
                model_layer_dict[(layer_name, ch)] = True
    
    return model_layer_dict

def scale(x, rmin=0, rmax=1):
    # Normalize tensor x to range [rmin, rmax]
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val == 0:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val) * (rmax - rmin) + rmin

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / total_neurons

# 커버되지 않은 뉴런 중 하나를 무작위로 선택하고, 모두 커버되었다면 전체 중에서 아무 뉴런이나 선택
def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(list(model_layer_dict.keys()))
    return layer_name, index

def constraint_light(gradients):
    gradients = gradients.detach().cpu().numpy()
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads

def constraint_occl(gradients, start_point, rect_shape):
    """
    gradients: shape = (1, C, H, W)
    start_point: (x, y)
    rect_shape: (h, w)
    """
    x, y = start_point
    h, w = rect_shape
    
    gradients = gradients.detach().cpu().numpy()
    new_grads = np.zeros_like(gradients)
    new_grads[:, :, y:y+h, x:x+w] = gradients[:, :, y:y+h, x:x+w]
    return new_grads

def constraint_black(gradients, rect_shape=(50, 50)):
    """
    gradients: shape = (1, C, H, W)
    rect_shape: (w, h)
    """
    _, _, H, W = gradients.shape
    w, h = rect_shape
    
    x = random.randint(0, W - w)
    y = random.randint(0, H - h)
    
    patch = gradients[:, :, y:y + h, x:x + w] # shape: (1, C, h, w)
    
    gradients = gradients.detach().cpu().numpy()
    new_grads = np.zeros_like(gradients)
    
    if patch.mean().item() < 0:
        new_grads[:, :, y:y + h, x:x + w] = -1.0
    
    return new_grads

def compute_joint_loss(cfg, model1, model2, gen_img, orig_label, layer_name1, index1, layer_name2, index2):
    # Compute the activations for the specified layers
    activation1 = {}
    activation2 = {}

    def get_activation_hook(name, container):
        def hook(module, input, output):
            container[name] = output.squeeze(0).detach()
        return hook

    hooks1 = []
    hooks2 = []

    for name, module in model1.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks1.append(module.register_forward_hook(get_activation_hook(name, activation1)))

    for name, module in model2.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks2.append(module.register_forward_hook(get_activation_hook(name, activation2)))

    # Forward pass
    with torch.no_grad():
        _ = model1(gen_img)
        _ = model2(gen_img)

    # Remove hooks
    for hook in hooks1:
        hook.remove()
    for hook in hooks2:
        hook.remove()

    # Compute the joint loss
    if cfg.target_model == 0:
        loss1 = -cfg.weight_diff * model1(gen_img)[0, orig_label] # model1: lower prediction
        loss2 = model2(gen_img)[0, orig_label] # model2: maintain prediction
    else:
        loss1 = model1(gen_img)[0, orig_label]
        loss2 = -cfg.weight_diff * model2(gen_img)[0, orig_label]

    loss1_neuron = activation1[layer_name1][index1].mean()
    loss2_neuron = activation2[layer_name2][index2].mean()
    
    return (loss1 + loss2) + cfg.weight_nc * (loss1_neuron + loss2_neuron)