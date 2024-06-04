from PIL import Image
from torchvision import models, transforms
import torch

resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mpath_01 = r'D:\Project\PyProject\pytorch_example\IMG\bobby.jpg'
mpath_02 = r'D:\Project\PyProject\pytorch_example\IMG\horse.jpg'
img = Image.open(mpath_02)
# img.show()
img = preprocess(img)
#
batch_t = torch.unsqueeze(img, 0)
#
out = resnet(batch_t)
#
with open(r'D:\Project\PyProject\pytorch_example\LAB\imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
#
_, index = torch.max(out, 1)
print(index)
#
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()
#
print(labels[index[0]], percentage[index[0]].item())
#
_, indices = torch.sort(out, dim=1, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
