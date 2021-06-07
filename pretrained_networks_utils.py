from torchvision import models
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional


class PreTrainedNetworkUtil:
    @staticmethod
    def image_recognition():
        resnet = models.resnet101(pretrained=True)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        img = Image.open(Path('resources/horse.jpg'))
        img_transform = preprocess(img)
        batch_t = torch.unsqueeze(img_transform, 0)
        resnet.eval()
        out = resnet(batch_t)
        with open('resources/imagenet_classes.txt') as file:
            labels = [line.strip() for line in file.readlines()]
        # index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        indices = torch.sort(out, descending=True)
        label = [(labels[idx], percentage[idx].item()) for idx in indices.indices[0][:5]]
        print(label)

