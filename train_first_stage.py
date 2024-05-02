import argparse
import os
import sys
import logging
import torch
import time
import wandb

from coreecho import get_feature_extractor
from coreecho.dataloader import set_loader
from coreecho.loss import RnCLoss
from coreecho.regressor import get_shallow_mlp_head
from coreecho.utils import AverageMeter, save_model, set_seed, set_optimizer
from coreecho.validation import validate
from coreecho.viz import HelperTSNE, HelperUMAP

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    
    eta_min = lr * args.lr_decay_rate
    
    if args.lr_step_epoch != -1:
        if epoch >= args.lr_step_epoch:
            lr = eta_min
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

print = logging.info

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_step_epoch', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--trial', type=int, default=0, help='id for recording multiple runs')
    
    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--model', type=str, default='uniformer_small', choices=['uniformer_small'])
    parser.add_argument('--aug', action='store_true', help='whether to use augmentations')
    
    # RnCLoss Parameters
    parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')
    
    parser.add_argument('--frames', type=int)
    parser.add_argument('--frequency', type=int)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project_name', type=str, default='echonet-ef')
    
    opt = parser.parse_args()
    
    opt.optim = 'adamw'
    
    opt.model_path = './save'
    opt.model_name = 'RnC+L1SG_{}_ep_{}_lr_{}_d_{}_wd_{}_bsz_{}_aug_{}_temp_{}_label_{}_feature_{}_trial_{}'. \
        format(
            opt.model, opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.weight_decay,
            opt.batch_size, opt.aug, opt.temp, opt.label_diff, opt.feature_sim, opt.trial
        )
    
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder exist.')
    
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])
    
    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")
    
    return opt

def set_model(opt):
    model = get_feature_extractor(opt.model, opt.pretrained_weights)
    if opt.model == 'uniformer_small':
        dim_in = model.head.in_features
    else:
        dim_in = model.fc.in_features
    dim_out = 1
    
    regressor = get_shallow_mlp_head(dim_in, dim_out)
    
    criterion = RnCLoss(temperature=opt.temp, label_diff=opt.label_diff, feature_sim=opt.feature_sim)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            regressor = torch.nn.DataParallel(regressor)
        model = model.cuda()
        regressor = regressor.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True
    
    return model, criterion, regressor

def train(train_loader, model, criterion, optimizer, epoch, opt, regressor, optimizer_regressor):
    model.train()
    regressor.train()
    
    criterion_mse = torch.nn.L1Loss()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        views1, views2 = batch
        images = torch.cat([views1["image"], views2["image"]], dim=0)
        labels = views1["label"]
        bsz = labels.shape[0]
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        _, features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        features = features.detach()
        y_preds = regressor(torch.cat((features[:,0], features[:,1]), dim=0))
        loss_reg = criterion_mse(y_preds, labels.repeat(2, 1))
        
        optimizer_regressor.zero_grad()
        loss_reg.backward()
        optimizer_regressor.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            sys.stdout.flush()

def main():
    opt = parse_option()
    
    # set wandb
    if opt.wandb:
        wandb.login()
        wandb.init(
            name=opt.model_name,
            project=opt.wandb_project_name,
            config={key: val for key, val in opt.__dict__.items()},
        )
    
    # Set seed (for reproducibility)
    set_seed(opt.trial)
    
    # build data loader
    train_loader, train_no_aug_loader, val_loader, test_loader = set_loader(opt)
    
    # build model and criterion
    model, criterion, regressor = set_model(opt)
    
    # build optimizer
    optimizer = set_optimizer(opt, model)
    optimizer_regressor = set_optimizer(opt, regressor)
    
    start_epoch = 0
    
    # training routine
    best_error = 1e5
    save_file_best = os.path.join(opt.save_folder, 'best.pth')
    for epoch in range(start_epoch, opt.epochs + 1):
        lr_cur_val = adjust_learning_rate(opt, optimizer, epoch)
        _ = adjust_learning_rate(opt, optimizer_regressor, epoch)
        
        train(train_loader, model, criterion, optimizer, epoch, opt, regressor, optimizer_regressor)
        
        valid_metrics, valid_aux  = validate(val_loader, model, regressor)
        valid_tsne = HelperTSNE(valid_aux['embeddings'], n_components=2, perplexity=5, random_state=7)
        valid_umap = HelperUMAP(valid_aux['embeddings'], n_components=2, n_neighbors=5, init='random', random_state=0)
        
        valid_error = valid_metrics['l1']
        is_best = valid_error <= best_error
        best_error = min(valid_error, best_error)
        print(f"Best MAE: {best_error:.3f}")
        
        if is_best:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'regressor': regressor.state_dict(),
                'best_error': best_error,
            }, save_file_best)
        
        dict_repr = {
            'epoch': epoch,
            'lr': lr_cur_val,
            'Val R2': valid_metrics['r2'].item(),
            'Val L2': valid_metrics['l2'].item(),
            'Val L1': valid_metrics['l1'].item(),
            **{f'Val UMAP ({key})': valid_umap(val) for key, val in valid_aux['aux'].items()},
            **{f'Val TSNE ({key})': valid_tsne(val) for key, val in valid_aux['aux'].items()},
        }
        
        if opt.wandb:
            wandb.log(dict_repr)
        
        save_file = os.path.join(opt.save_folder, 'last.pth')
        save_model(model, regressor, opt, epoch, save_file, best_error)
    
    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    model.load_state_dict(checkpoint['model'])
    regressor.load_state_dict(checkpoint['regressor'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}")
    
    set_seed(opt.trial)
    test_metrics, test_aux = validate(test_loader, model, regressor)
    print('Test R2: {:.3f}'.format(test_metrics['r2']))
    print('Test L2: {:.3f}'.format(test_metrics['l2']))
    print('Test L1: {:.3f}'.format(test_metrics['l1']))
    
    set_seed(opt.trial)
    val_metrics, val_aux = validate(val_loader, model, regressor)
    print('Val R2: {:.3f}'.format(val_metrics['r2']))
    print('Val L2: {:.3f}'.format(val_metrics['l2']))
    print('Val L1: {:.3f}'.format(val_metrics['l1']))
    
    set_seed(opt.trial)
    train_metrics, train_aux = validate(train_no_aug_loader, model, regressor)
    print('Train R2: {:.3f}'.format(train_metrics['r2']))
    print('Train L2: {:.3f}'.format(train_metrics['l2']))
    print('Train L1: {:.3f}'.format(train_metrics['l1']))
    
    train_tsne = HelperTSNE(train_aux['embeddings'], n_components=2, perplexity=5, random_state=7)
    train_umap = HelperUMAP(train_aux['embeddings'], n_components=2, n_neighbors=5, init='random', random_state=0)
    
    val_tsne = HelperTSNE(val_aux['embeddings'], n_components=2, perplexity=5, random_state=7)
    val_umap = HelperUMAP(val_aux['embeddings'], n_components=2, n_neighbors=5, init='random', random_state=0)
    
    test_tsne = HelperTSNE(test_aux['embeddings'], n_components=2, perplexity=5, random_state=7)
    test_umap = HelperUMAP(test_aux['embeddings'], n_components=2, n_neighbors=5, init='random', random_state=0)
    
    dict_repr = {
        '(Best) Train R2': train_metrics['r2'].item(),
        '(Best) Train L2': train_metrics['l2'].item(),
        '(Best) Train L1': train_metrics['l1'].item(),
        '(Best) Val R2': val_metrics['r2'].item(),
        '(Best) Val L2': val_metrics['l2'].item(),
        '(Best) Val L1': val_metrics['l1'].item(),
        '(Best) Test R2': test_metrics['r2'].item(),
        '(Best) Test L2': test_metrics['l2'].item(),
        '(Best) Test L1': test_metrics['l1'].item(),
        **{f'(Best) Train UMAP ({key})': train_umap(val) for key, val in train_aux['aux'].items()},
        **{f'(Best) Val UMAP ({key})': val_umap(val) for key, val in val_aux['aux'].items()},
        **{f'(Best) Test UMAP ({key})': test_umap(val) for key, val in test_aux['aux'].items()},
        **{f'(Best) Train TSNE ({key})': train_tsne(val) for key, val in train_aux['aux'].items()},
        **{f'(Best) Val TSNE ({key})': val_tsne(val) for key, val in val_aux['aux'].items()},
        **{f'(Best) Test TSNE ({key})': test_tsne(val) for key, val in test_aux['aux'].items()},
    }
    
    if opt.wandb:
        wandb.log(dict_repr)

if __name__ == '__main__':
    main()