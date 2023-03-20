import torch.nn.functional as F

def get_loss(output, sample):
    y_pred = output['y_pred']
    y, mask_hr, mask_lr = (sample[k] for k in ('y', 'mask_hr', 'mask_lr'))

    l1_loss = l1_loss_func(y_pred, y, mask_hr)
    mse_loss = mse_loss_func(y_pred, y, mask_hr)

    loss = l1_loss*10

    return loss, {
        'l1_loss': l1_loss.detach().item(),
        'mse_loss': mse_loss.detach().item(),
        'optimization_loss': loss.detach().item(),
    }
        

def mse_loss_func(pred, gt, mask):
    return F.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    return F.l1_loss(pred[mask == 1.], gt[mask == 1.])

