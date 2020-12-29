import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
from lib import PraNet, U2PraNet_plus_plus, U2NET, U2NET_plus
from utils.dataloader import test_dataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/archive/U2PraNet++(Ultimate).pth')

# data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
data_path = './data/PolypData/test/Kvasir'
save_path = './data/PolypData/test/Kvasir/U2PraNet++/'
# data_path = './data/PolypData/test/test'
# save_path = './data/PolypData/test/test/pred/'
opt = parser.parse_args()
model = U2PraNet_plus_plus()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = '{}/imgs/'.format(data_path)
gt_root = '{}/masks/'.format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()

    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(image)
    # lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, before_SA, after_SA, SA_add = model(image)
    # Sg, R5, S5, crop_4, R4, S4, crop_3, R3, S3, crop_2, lateral_map_2 = model(image)

    res = lateral_map_2
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    imageio.imwrite(save_path+name, res)

    # res = res > 0.5
    # res = Image.fromarray((res * 255).astype(np.uint8))
    # res.save(save_path+name)
