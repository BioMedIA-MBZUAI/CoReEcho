import argparse
import copy
import pandas as pd
import torch
import pickle

from torcheval.metrics.functional import r2_score

from coreecho import get_feature_extractor
from coreecho.dataset import EchoNetTest
from coreecho.regressor import get_shallow_mlp_head
from coreecho.utils import load_model
from coreecho.validation import validate

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--path_test_start_indexes', type=str)
    parser.add_argument('--path_save_test_files', type=str)
    
    parser.add_argument('--model', type=str, default='uniformer_small', choices=['uniformer_small'])
    
    parser.add_argument('--frames', type=int)
    parser.add_argument('--frequency', type=int)
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    
    opt = parser.parse_args()
    
    return opt

def set_model(opt):
    model = get_feature_extractor(opt.model, None)
    if opt.model == 'uniformer_small':
        dim_in = model.head.in_features
    else:
        dim_in = model.fc.in_features
    dim_out = 1
    
    regressor = get_shallow_mlp_head(dim_in, dim_out)
    
    checkpoint = torch.load(opt.pretrained_weights, map_location='cpu')
    model = load_model(model, checkpoint['model'])
    regressor = load_model(regressor, checkpoint['regressor'])
    
    if torch.cuda.is_available():
        model = model.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True
    
    return model, regressor

def set_test_loader(opt):
    test_ds = EchoNetTest(
            root=opt.data_folder,
            frames=opt.frames,
            frequency=opt.frequency,
            path_test_start_indexes=opt.path_test_start_indexes,
            trial=opt.trial,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=opt.num_workers,
    )
    
    return test_loader

if __name__ == '__main__':
    opt = parse_option()
    
    model, regressor = set_model(opt)
    
    df = pd.read_pickle(opt.path_test_start_indexes)
    list_trial = list(range(len(df[list(df.keys())[0]])))
    
    list_outputs = []
    best_r2 = -1_000_000
    for trial in list_trial:
        opt.trial = trial
        test_loader = set_test_loader(opt)
        test_metrics, test_aux = validate(test_loader, model, regressor)
        if best_r2 <= test_metrics['r2']:
            best_r2 = max(best_r2, test_metrics['r2'])
            best_metrics = copy.deepcopy(test_metrics)
            best_aux = copy.deepcopy(test_aux)
        list_outputs.append(test_aux['outputs'])
        
        print('-'*10)
        print('Trial ', trial)
        print(test_metrics)
        print('')
    
    outputs = torch.cat(list_outputs, dim=1).mean(dim=1)[:,None]
    labels = test_aux['labels']
    
    metrics = {
        'r2': r2_score(outputs, labels),
        'l1': torch.nn.L1Loss()(outputs, labels),
        'l2': torch.sqrt(torch.nn.MSELoss()(outputs, labels)),
    }
    
    print('-'*30)
    print(f'Metrics from {len(list_trial)}x clips')
    print(metrics)
    
    dict_test_files = {
        'N clips': len(list_trial),
        'metrics xN clips': metrics,
        'best_metrics x1 clip': best_metrics,
        'best_aux x1 clip': best_aux,
    }
    
    with open(opt.path_save_test_files, 'wb') as f:
        pickle.dump(dict_test_files, f)