import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-29.pth')

# for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
# data_path = './data/CVCpolyp/pred/{}/'.format(_data_name)
# save_path = './results/PraNet/{}/'.format(_data_name)
data_path = './data/CVCpolyp'
save_path = './data/CVCpolyp/pout/'
opt = parser.parse_args()
model = PraNet()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = '{}/pred/'.format(data_path)
gt_root = '{}/predmasks/'.format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()

    res5, res4, res3, res2 = model(image)
    res = res2
    print(gt.shape)
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    imageio.imwrite(save_path+name, res)
