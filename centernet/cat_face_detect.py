# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_model(name):
    """ load pth model """
    model = torch.load(
        name,
        map_location=torch.device('cpu')
        )

    return model


if __name__ == '__main__':
    # load model
    name = '/home/lz/hutpond/deeplearn/source/CenterNet/exp/multi_pose/dla_1x_catface/model_last.pth'
    model = load_model(name)

    # open image
    img_name = '/home/lz/hutpond/deeplearn/source/CenterNet/data/coco/test2017/6029.jpg'
    image = Image.open(img_name)
    image = np.array(image)
    trans = transforms.Compose([transforms.ToTensor()])
    image = trans(image)
    image = image.unsqueeze(0)
    target = model(image)
    print(target)