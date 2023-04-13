import csv
import os

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm


class fine_tune_model(nn.Module):
    def __init__(self,net,input_dim):
        super().__init__()
        self.net = net
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
        feature = self.net(x)
        feature = feature.view(feature.size(0),-1)
        out = self.n(feature)
        return out

class ImagesDataset(Dataset):
    def __init__(self, rootdir,datatype,data_dic):
        super().__init__()
        self.data_dic = data_dic
        self.label_u = None
        self.root_dir = rootdir
        img_name,self.label = self.read_csv(datatype)
        self.file_name = [os.path.join(self.root_dir,datatype, i) for i in img_name]

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
        return self.transform(img),self.label[index]
    def read_csv(self,datatype):
        path = os.path.join(self.root_dir,'%s.csv' %datatype)
        with open(path, newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile,delimiter=',')
            # print(rows)
            img_name,label = [],[]
            for row in rows:
                img_name.append(row[1])
                label.append(row[2])
            img_name = img_name[1:]
            label = label[1:]
            label = [self.data_dic[i] for i in label]
            # print(label)

        return img_name,label

if __name__ == '__main__':
    root_dir = '../../../hw4_data/office'
    path = os.path.join(root_dir, 'train.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device ',device)
    resnet = models.resnet50(pretrained=False).to(device)

    # resnet.load_state_dict(torch.load('/home/alex/Desktop/hw4-s2137127/hw4_data/pretrain_model_SL.pt'))
    # print(resnet.fc.in_features)
    in_fea = resnet.fc.in_features
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    fine = fine_tune_model(resnet,in_fea).to(device)
    
    print(fine)
    BATCH_SIZE=64
    NUM_WORKERS=4
    epoch=50
    # optimizer = optim.SGD(fine.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(resnet.parameters(), lr=3e-4)
    # optimizer2 = optim.SGD(fine.parameters(), lr=0.01, momentum=0.9)
    # optimizer2 = optim.Adam(fine.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        l = [i[2] for i in rows]
        l = np.unique(l[1:])
    data_dic = {name:idx for idx,name in enumerate(l)}
    train_ds,val_ds = ImagesDataset(root_dir,'train',data_dic),ImagesDataset(root_dir,'val',data_dic)
    train_ld,val_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,pin_memory=True),DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,pin_memory=True)
    # resnet.eval()
    best_acc = 0
    cnt = 0
    for ep in range(epoch):
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        for images,labels in tqdm(train_ld):
            images = images.to(device)
            labels = labels.to(device)
            fine.train()
            optimizer.zero_grad()
            out = fine(images)
            # print(out.shape,labels.shape)
            loss = criterion(out, labels)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += len(labels)
            correct_cnt += (pred_label == labels).sum().item()
            # print(out.shape)
        acc = correct_cnt / total_cnt
        ave_loss = total_loss / total_cnt
        print('Training epoch: {}, train loss: {:.6f}, acc: {:.3f}'.format(
            ep, ave_loss, acc))
        fine.eval()
        loss_val = 0
        total_correct = 0
        total_cnt = 0
        accuracy = 0
        valid_acc,valid_loss = [],[]
        with torch.no_grad():
            for images, labels in tqdm(val_ld):
                images, labels = images.to(device), labels.to(device)
                out = fine(images)
                loss = criterion(out, labels)
                loss_val += loss.item()
                _, pred_label = torch.max(out, 1)
                total_correct += (pred_label == labels).sum().item()
                total_cnt += len(labels)

            acc = total_correct/total_cnt
            if acc > best_acc:
                cnt = 0
                best_acc = acc
                torch.save(fine.state_dict(), './fine/fine_best_A.pth')
            elif best_acc > acc:
                cnt += 1
            print("current loss: ", loss_val/total_cnt, "   valid_acc: ", acc)
            print("best_acc ", best_acc)
            if cnt > 10:
                print("early stop!!!")
                break




