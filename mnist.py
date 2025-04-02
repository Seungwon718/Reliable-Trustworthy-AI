import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
def train_model(model, device, train_loader, optimizer, epoch=10):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# Evaluation function
def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Test accuracy: {correct / len(test_loader.dataset):.4f}')


# Adversarial attacks
def fgsm_targeted(model, data, target, eps):
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    
    perturbed_data = data - eps * data.grad.data.sign()

    return torch.clamp(perturbed_data, 0, 1)

def fgsm_untargeted(model, data, label, eps):
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    perturbed_data = data + eps * data.grad.data.sign()

    return torch.clamp(perturbed_data, 0, 1)

def pgd_targeted(model, data, target, k, eps, eps_step):
    perturbed_data = data + torch.empty_like(data).uniform_(-eps, eps)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    for _ in range(k):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        perturbed_data = perturbed_data - eps_step * perturbed_data.grad.data.sign()
        eta = torch.clamp(perturbed_data - data, min=-eps, max=eps)
        perturbed_data = torch.clamp(data + eta, 0, 1).detach()
    return perturbed_data

def pgd_untargeted(model, data, label, k, eps, eps_step):
    perturbed_data = data + torch.empty_like(data).uniform_(-eps, eps)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    for _ in range(k):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()

        perturbed_data = perturbed_data + eps_step * perturbed_data.grad.data.sign()
        eta = torch.clamp(perturbed_data - data, min=-eps, max=eps)
        perturbed_data = torch.clamp(data + eta, 0, 1).detach()
    return perturbed_data

def test_adversarial(model, device, test_loader, attack_fn, isTargeted=False, **kwargs):
    model.eval()

    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # For targeted, randomly select new_target from possible outputs excluding the label
        if isTargeted:
            rand = torch.randint(0, 9, target.shape, device=target.device)
            new_target = rand + (rand >= target).long()

            perturbed_data = attack_fn(model, data, new_target, **kwargs)
            output = model(perturbed_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            perturbed_data = attack_fn(model, data, target, **kwargs)
            output = model(perturbed_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'After Adversarial accuracy: {correct / len(test_loader.dataset):.4f}')

    plt.figure(figsize=(5, 5))
    plt.imshow(data[0].cpu().detach().numpy().squeeze(), cmap='gray')
    plt.title(f"Pred: {pred[0].cpu().detach().numpy().squeeze()}\nLabel: {target[0].cpu().detach().numpy().squeeze()}", fontsize=12)
    plt.axis('off')
    plt.savefig(f'mnist_{attack_fn.__name__}.png', bbox_inches='tight')
    plt.close()