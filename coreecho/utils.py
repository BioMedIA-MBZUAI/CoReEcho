import numpy as np
import random
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model

def save_model(model, regressor, opt, epoch, save_file, best_error=None):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'regressor': regressor.state_dict(),
        'epoch': epoch,
        'best_error': best_error,
    }
    torch.save(state, save_file)
    del state

def set_optimizer(opt, model):
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.learning_rate,
            momentum=opt.momentum, weight_decay=opt.weight_decay
        )
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt.learning_rate,
            weight_decay=opt.weight_decay
        )
    
    return optimizer

def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True