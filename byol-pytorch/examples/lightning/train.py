import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL

from tqdm import tqdm

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 90
# EPOCHS     = 1000
# LR         = 3e-4
# NUM_GPUS   = 1
IMAGE_SIZE = 128
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()




class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
            # transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)
        # return 10
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# main

if __name__ == '__main__':
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    resnet = models.resnet50(pretrained=False).to('cuda')

    learner = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


    for ep in range(100):
        loss_t = []
        for images in tqdm(train_loader):
            # images = sample_unlabelled_images()
            images = images.to('cuda')
            loss = learner(images)
            loss_t.append(loss.cpu().detach().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of target encoder
        print('epoch: ',ep,'loss:',np.mean(loss_t))
        if ep %5 ==0 :
            torch.save(resnet.state_dict(), './lightning_logs/improved-net_%d.pt' %ep)
