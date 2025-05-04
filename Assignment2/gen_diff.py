import random

from imageio import imwrite

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Model1 import Model1
from Model2 import Model2
from utils import *

class Config:
    transformation: str = 'light'         # realistic transformation type: 'light', 'occl', or 'blackout'
    weight_diff: float = 10             # weight hyperparm to control differential behavior
    weight_nc: float = 10               # weight hyperparm to control neuron coverage
    step: float = 1e-2                    # step size of gradient descent
    seeds: int = 10000                       # number of seeds of input
    grad_iterations: int = 50             # number of iterations of gradient descent
    threshold: float = 0.5               # threshold for determining neuron activated
    target_model: int = 0                 # target model that we want it predicts differently (0/1)
    start_point: tuple = (0, 0)           # occlusion upper left corner coordinate (used if transformation == 'occl')
    occlusion_size: tuple = (50, 50)      # occlusion size (width, height)

cfg = Config()

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = Model1(train=False, device=device)
model2 = Model2(train=False, device=device)
model1.eval()
model2.eval()

# init coverage table
model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

# start gen inputs
for _ in range(cfg.seeds):
    orig_img, _ = random.choice(testset) # random select image (image, label)
    orig_img = orig_img.unsqueeze(0).to(device) # add batch dimension
    gen_img = orig_img.clone().requires_grad_() # clone and set requires_grad=True
    
    label1, label2 = model1(orig_img).argmax(dim=1).item(), model2(orig_img).argmax(dim=1).item()
    if label1 != label2:
        print(f'Input already causes different outputs: {label1}, {label2}')

        model_layer_dict1 = update_coverage(gen_img, model1, model_layer_dict1, cfg.threshold)
        model_layer_dict2 = update_coverage(gen_img, model2, model_layer_dict2, cfg.threshold)
        
        covered1, total1, percent1 = neuron_covered(model_layer_dict1)
        covered2, total2, percent2 = neuron_covered(model_layer_dict2)

        print("=== Neuron Coverage Report ===")
        print(f"Model 1: {covered1}/{total1} neurons covered ({percent1:.2%})")
        print(f"Model 2: {covered2}/{total2} neurons covered ({percent2:.2%})")
        print("=================================")
        
        gen_img_deprocessed = deprocess_image(gen_img)
        
        save_path = f'./generated_inputs/already_differ_{label1}_{label2}.png'
        imwrite(save_path, gen_img_deprocessed)
        continue
    
    # if all label agrees
    print("AA")
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    
    for iters in range(cfg.grad_iterations):
        loss = compute_joint_loss(cfg, model1, model2, gen_img, orig_label, layer_name1, index1, layer_name2, index2)
        grads = normalize(torch.autograd.grad(loss, gen_img)[0])
        
        if cfg.transformation == 'light':
            grads = constraint_light(grads)
        elif cfg.transformation == 'occl':
            grads = constraint_occl(grads, cfg.start_point, cfg.occlusion_size)
        elif cfg.transformation == 'blackout':
            grads = constraint_black(grads)
        
        grads = torch.from_numpy(grads).to(gen_img.device)
        gen_img = gen_img + cfg.step * grads
        label1 = model1(gen_img).argmax(dim=1).item()
        label2 = model2(gen_img).argmax(dim=1).item()
        
        if label1 != label2:
            model_layer_dict1 = update_coverage(gen_img, model1, model_layer_dict1, cfg.threshold)
            model_layer_dict2 = update_coverage(gen_img, model2, model_layer_dict2, cfg.threshold)
            
            covered1, total1, percent1 = neuron_covered(model_layer_dict1)
            covered2, total2, percent2 = neuron_covered(model_layer_dict2)

            print("=== Neuron Coverage Report ===")
            print(f"Model 1: {covered1}/{total1} neurons covered ({percent1:.2%})")
            print(f"Model 2: {covered2}/{total2} neurons covered ({percent2:.2%})")
            print("=================================")
            
            averaged_nc = (covered1 + covered2) / (total1 + total2)
            print(f"Averaged covered neurons: {averaged_nc:.3f}")
            
            orig_img_deprocessed = deprocess_image(orig_img)
            gen_img_deprocessed = deprocess_image(gen_img)
            
            save_path = f"./generated_inputs/{cfg.transformation}_{label1}_{label2}"
            imwrite(f"{save_path}_orig.png", orig_img_deprocessed)
            imwrite(f"{save_path}.png", gen_img_deprocessed)
            break