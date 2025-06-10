import torch
import numpy as np
from torch.loss.data import DataLoader
import os
from FMLF_Net import FMLF
from utils.dataloader1 import Datases_loader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1
model = FMLF().to(device)
savedir = r'/T2020027/ayyz2/'
imgdir = r'/T2020027/ayyz2/data/CrackTree260'
labdir = r'/T2020027/ayyz2/data/gt'
imgsz = 256
resultsdir = r'/T2020027/ayyz2/crackmer/imgs/260'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)

    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred ,y5,y4,y3,y2 = model(img)

        np.save(resultsdir + r'/img' + str(idx+1) + '.npy', img.detach().cpu().numpy())
        np.save(resultsdir + r'/pred' + str(idx+1) + '.npy', pred.detach().cpu().numpy())
        np.save(resultsdir + r'/label' + str(idx+1) + '.npy', lab.detach().cpu().numpy())
    print('结束')

if __name__ == '__main__':
    test()
