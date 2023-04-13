import csv
import json
import os
from os.path import isdir, dirname, basename
from sys import argv

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm


class fine_tune_model(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.n = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256, bias=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=65, bias=True),
        )
    def forward(self,x):
        out = self.n(x)
        return out


class ImagesDataset(Dataset):
    def __init__(self, csv_path,img_path):
        super().__init__()
        self.data_dic = data_dic
        self.csv_path = csv_path
        self.img_path = img_path
        img_name = self.read_csv(self.csv_path)
        self.file_name = [os.path.join(self.img_path, i) for i in img_name]

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
            # transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):

        img = Image.open(self.file_name[index])
        img = img.convert('RGB')
        return self.transform(img),basename(self.file_name[index])
    def read_csv(self,path):
        with open(path, newline='') as csvfile:
            rows = csv.reader(csvfile,delimiter=',')
            img_name = []
            for row in rows:
                img_name.append(row[1])

            img_name = img_name[1:]
            # print(label)

        return img_name

if __name__ == '__main__':
    csv_path,img_path,output = argv[1],argv[2],argv[3]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device ',device)

    resnet = models.resnet50(pretrained=False).to(device)

    resnet.load_state_dict(torch.load('/home/alex/Desktop/hw4-s2137127/hw4_data/pretrain_model_SL.pt'))
    # print(resnet.fc.in_features)
    fine = fine_tune_model(resnet.fc.in_features).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    fine.load_state_dict(torch.load('/home/alex/Desktop/hw4-s2137127/byol-pytorch/examples/lightning/fine/fine_best_D.pth'))
    # print(fine)

    with open("output.json") as f:
        # 讀取 JSON 檔案
        data_dic = json.load(f)
    test_ds = ImagesDataset(csv_path,img_path)
    test_ld = DataLoader(test_ds, batch_size=16, num_workers=4, shuffle=False)
    fine.eval()
    resnet.eval()
    prediction = []
    name = []
    with torch.no_grad():
        for images,names in tqdm(test_ld):
            images = images.to(device)
            feature = resnet(images).view(images.size(0), -1)
            out = fine(feature)
            _, pred_label = torch.max(out, 1)
            for i in range(len(pred_label)):
                prediction.append(pred_label[i])
                name.append(names[i])
    pred_name = []
    for i in prediction:
        tmp = [k for k,v in data_dic.items() if v == i.data]
        pred_name.append(tmp[0])
    if not isdir(dirname(output)):
        os.mkdir(dirname(output))
    with open(output, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','filename', 'label'])
        for i in range(len(name)):
            writer.writerow([i, name[i],pred_name[i]])
