# utils.py
from pathlib import Path
import os
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch
import math
import h5py


def confusion_matrix_2d(x, y, th=0.5, reduce=True):
    x_ = x.gt(th).to(torch.float)
    y_ = y.gt(th).to(torch.float)

    c = 2 * y_ - x_

    if reduce:
        dim = [0, 1, 2, 3, 4]
    else:
        dim = [1, 2, 3, 4]

    tp = (c == 1).sum(dim=dim, dtype=torch.float)
    tn = (c == 0).sum(dim=dim, dtype=torch.float)
    fp = (c == -1).sum(dim=dim, dtype=torch.float)
    fn = (c == 2).sum(dim=dim, dtype=torch.float)

    return tp, tn, fp, fn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val.mean() * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix(object):
    def __init__(self):
        self.tp = AverageMeter()
        self.tn = AverageMeter()
        self.fp = AverageMeter()
        self.fn = AverageMeter()
        self.dice = AverageMeter()
        self.jcc = AverageMeter()

        self.f1 = 0
        self.f05 = 0
        self.f2 = 0

        self.precision = 0
        self.recall = 0

    def update(self, cm, n):
        self.tp.update(cm[0], n=n)
        self.tn.update(cm[1], n=n)
        self.fp.update(cm[2], n=n)
        self.fn.update(cm[3], n=n)

        self.dice.update(2 * cm[0] / (2 * cm[0] + cm[2] + cm[3] + 0.00001), n=n)
        self.jcc.update(cm[0] / (cm[2] + cm[3] + cm[0] + 0.00001), n=n)

        self.total = self.tp.sum + self.tn.sum + self.fp.sum + self.fn.sum

        self.prec = self.tp.sum / (self.tp.sum + self.fp.sum + 0.00001)
        self.recall = self.tp.sum / (self.tp.sum + self.fn.sum + 0.00001)

        self.f1 = (2 * self.prec * self.recall / (self.prec + self.recall + 0.00001)).mean()
        self.f05 = ((1 + 0.25) * self.prec * self.recall / (0.25 * self.prec + self.recall + 0.00001)).mean()
        self.f2 = ((1 + 4) * self.prec * self.recall / (4 * self.prec + self.recall + 0.00001)).mean()


def pearson_correlation_coeff(x, y):
    std_x = np.std(x)
    std_y = np.std(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    vx = (x - mean_x) / (std_x + 0.0001)
    vy = (y - mean_y) / (std_y + 0.0001)

    return np.mean(vx * vy)


def psnr(x, y):
    mse = np.linalg.norm(y - x)

    if mse == 0:
        return 100

    PIXEL_MAX = 1.

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_save_dir(arg):
    save_path = './' + arg.output_dir + arg.inifile

    '''
    if arg.phase == 'infer':
        save_path = arg.output_dir
    else:
        save_path = "./outs/" + arg.inifile
    '''

    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    return save_path


def RVD(output, target):
    # Relative Volume Difference
    output_sum = output.sum()
    target_sum = target.sum()
    if output_sum == target_sum:
        return 1

    score = (output_sum - target_sum) / target_sum
    # Higher is Better
    return -score


def torch_downsample(img, scale):
    # Create grid
    out_size = img.shape[-1] // scale
    x = torch.linspace(-1, 1, out_size).view(-1, 1).repeat(1, out_size)
    y = torch.linspace(-1, 1, out_size).repeat(out_size, 1)
    grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2).cuda()
    # grid.unsqueeze_(0)

    return F.grid_sample(img, grid)


def get_roc_pr(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 1
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else 1
    recall = tp / (tp + fn) if (tp + fn) != 0 else 1

    # f1 = 2 * precision * recall / (precision + recall) # if (precision + recall) != 0 else 1
    f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) != 0 else 1
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 1

    return sensitivity, 1 - specificity, precision, recall, f1, jaccard


def slice_threshold(_np, th):
    return (_np >= th).astype(int)


def image_save(save_path, **args):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hf = h5py.File(save_path + '.hdf', 'w')
    for k, v in args.items():
        hf.create_dataset(k, data=v, compression='gzip')
    hf.close()


def slack_alarm(send_id, send_msg="Train Done"):
    """
    send_id : slack id. ex) zsef123
    """
    from slackclient import SlackClient
    slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
    if slack_client.rtm_connect(with_team_state=False):
        ret = slack_client.api_call("chat.postMessage", channel="@" + send_id, text=send_msg, as_user=True)
        resp = "Send Failed" if ret['ok'] == False else "To %s, send %s" % (send_id, send_msg)
        print(resp)
    else:
        print("Client connect Fail")


if __name__ == "__main__":
    # x = torch.Tensor([[1,2,3,4,5],[6,7,8,9,10]])
    # y = torch.Tensor([[1,2,3,4,5],[6,7,8,9,10]])

    #x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    #y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    #print(pearson_correlation_coeff(x, y))

    x = torch.randn(4, 1, 16, 224, 224)
    y = torch.randn(4, 1, 16, 224, 224)

    print(np.sum(confusion_matrix_2d(x, y)))
    print(4*1*16*224*224)