import torch
import torch.nn.functional as F
from tqdm import tqdm
from lib import U2PraNet_plus_plus, PraNet_plus_plus, PraNet, U2NET, U2NET_plus
from utils.dataloader import get_test_loader

from dice_loss import dice_coeff


def eval_net(net, loader, device, network_name):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                if network_name == '4returns':
                    masks_pred_4, masks_pred_3, masks_pred_2, masks_pred = net(imgs)
                else:
                    masks_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(masks_pred, true_masks).item()
            else:
                pred = torch.sigmoid(masks_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val


if __name__ == '__main__':
    # data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    image_root = 'data/PolypData/test/ETIS-LaribPolypDB/imgs/'
    gt_root = 'data/PolypData/test/ETIS-LaribPolypDB/masks/'
    # model = U2PraNet_plus_plus().cuda()
    # model = PraNet_plus_plus().cuda()
    # model = PraNet().cuda()
    # model = U2NET_plus().cuda()
    model = U2PraNet_plus_plus().cuda()
    model.load_state_dict(torch.load('snapshots/archive/U2PraNet++-69.pth'))
    test_loader = get_test_loader(image_root, gt_root)
    test_dice = eval_net(model, test_loader, 'cuda', '4returns')
    print(test_dice)
