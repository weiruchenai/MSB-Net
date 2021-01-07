import torch
import logging

import os
import argparse
import torch.nn.functional as F

from datetime import datetime
from torch.autograd import Variable
from lib import PraNet, PraNet_plus_plus, U2PraNet_plus_plus, U2NET, U2NET_plus, U_Net
from utils.dataloader import get_loader
from eval import eval_net
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from torch.utils.tensorboard import SummaryWriter


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, val_loader, model, optimizer, epoch, writer, global_step, return_num):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            if return_num == 4:
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
            if return_num == 1:
                lateral_map_2 = model(images)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = loss2  # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                if return_num == 4:
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)
                if return_num == 1:
                    loss_record2.update(loss2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            val_score = eval_net(model, val_loader, 'cuda', return_num)
            if return_num == 4:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}, dice: {:0.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(), val_score))
                writer.add_scalar('Loss5/train', loss_record5.show().item(), i + global_step * total_step)
            if return_num == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-2: {:.4f}, dice: {:0.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record2.show(), val_score))
            writer.add_scalar('Loss2/train', loss_record2.show().item(), i + global_step * total_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i + global_step * total_step)
            writer.add_scalar('Dice/test', val_score, i + global_step * total_step)

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth' % epoch)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=60, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/PolypData/', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='U2PraNet(NoSEASPP)')
    parser.add_argument('-n', '--network', metavar='N', type=str, default="U2PraNet_plus_plus",
                        help='choice of network: U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, ResUnetPlusPlus, '
                             'PraNet, PraNet_plus_plus, U2Net, U2NET_plus, U2PraNet_plus_plus', dest='network')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-r', '--return_num', dest='return_num', type=int, default=4,
                        help='return number of the model')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device

    if opt.network == 'U_Net':
        #     # model = UNet(n_channels=3, n_classes=1, bilinear=False)
        model = U_Net().cuda()
    # if opt.network == 'R2U_Net':
    #     model = R2U_Net(n_channels=3, n_classes=1, bilinear=False).cuda()
    # if opt.network == 'AttU_Net':
    #     model = AttU_Net(n_channels=3, n_classes=1, bilinear=False).cuda()
    # if opt.network == 'R2AttU_Net':
    #     model = R2AttU_Net(n_channels=3, n_classes=1, bilinear=False).cuda()
    # if opt.network == 'NestedUNet':
    #     model = NestedUNet(n_channels=3, n_classes=1, bilinear=False).cuda()
    # if opt.network == 'ResUnetPlusPlus':
    #     model = ResUnetPlusPlus(n_channels=3, n_classes=1, bilinear=False).cuda()
    if opt.network == 'PraNet':
        model = PraNet().cuda()
    if opt.network == 'PraNet_plus_plus':
        model = PraNet_plus_plus().cuda()
    if opt.network == 'U2Net':
        model = U2NET().cuda()
    if opt.network == 'U2NET_plus':
        model = U2NET_plus().cuda()
    if opt.network == 'U2PraNet_plus_plus':
        model = U2PraNet_plus_plus().cuda()

    if opt.load:
        model.load_state_dict(torch.load(opt.load))
        logging.info(f'Model loaded from {opt.load}')

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/imgs/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_loader = loader[0]
    val_loader = loader[1]
    n_train = loader[2]
    n_val = 100
    total_step = len(train_loader)

    # tensorboard
    writer = SummaryWriter(comment=f'_BS={opt.batchsize}_Epoch={opt.epoch}')

    logging.info(f'''Starting training:
        Network:         {opt.network}
        Epochs:          {opt.epoch}
        Batch size:      {opt.batchsize}
        Dataset:         {opt.train_path}
        Learning rate:   {opt.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Image size:      {opt.trainsize}
    ''')
    global_step = 0
    return_num = opt.return_num
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        global_step += 1
        train(train_loader, val_loader, model, optimizer, epoch, writer, global_step, return_num)

    writer.close()
