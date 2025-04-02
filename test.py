import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import mnist
import cifar

'''
MNIST Dataset과 CIFAR10 Dataset에 대해 CNN을 각각 학습시켜 Test Accuracy 계산한한 후,
Input Data에 각 공격이 적용되었을 때, 정확도가 어떻게 변화하는지 출력하는 코드
'''

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)

# Train CNN model
model = mnist.CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("MNIST Dataset")

mnist.train_model(model, device, train_loader, optimizer)
mnist.test_model(model, device, test_loader)

print("\nUntargeted FGSM Attack")
mnist.test_adversarial(model, device, test_loader, mnist.fgsm_untargeted, eps=0.25)

print("\nTargeted FGSM Attack")
mnist.test_adversarial(model, device, test_loader, mnist.fgsm_targeted, isTargeted=True, eps=0.25)

print("\nUntargeted PGD Attack")
mnist.test_adversarial(model, device, test_loader, mnist.pgd_untargeted, eps=0.25, eps_step=0.01, k=50)

print("\nTargeted PGD Attack")
mnist.test_adversarial(model, device, test_loader, mnist.pgd_targeted, isTargeted=True, eps=0.25, eps_step=0.01, k=50)

cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=1000, shuffle=False)

model = cifar.CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nCIFAR Dataset")

cifar.train_model(model, device, train_loader, optimizer, epoch=10)
cifar.test_model(model, device, test_loader)

print("\nUntargeted FGSM Attack")
cifar.test_adversarial(model, device, test_loader, cifar.fgsm_untargeted, eps=0.25)

print("\nTargeted FGSM Attack")
cifar.test_adversarial(model, device, test_loader, cifar.fgsm_targeted, isTargeted=True, eps=0.25)

print("\nUntargeted PGD Attack")
cifar.test_adversarial(model, device, test_loader, cifar.pgd_untargeted, eps=0.25, eps_step=0.01, k=50)

print("\nTargeted PGD Attack")
cifar.test_adversarial(model, device, test_loader, cifar.pgd_targeted, isTargeted=True, eps=0.25, eps_step=0.01, k=50)
