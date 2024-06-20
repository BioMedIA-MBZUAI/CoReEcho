import copy
import numpy as np
import torch

from torcheval.metrics.functional import r2_score

def validate(val_loader, model, regressor, n_clips_per_sample=1):
    model.eval()
    regressor.eval()
    
    best_r2 = - np.Inf
    loutputs = []
    for _ in range(n_clips_per_sample):
        list_outputs = []
        list_labels = []
        list_embeddings = []
        dict_list_aux = {}
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                images = batch["image"]
                labels = batch["label"]
                
                for key in batch.keys():
                    if key in ["id", "image", "label"]:
                        continue
                    if key not in dict_list_aux:
                        dict_list_aux[key] = []
                    key, torch.is_tensor(batch[key])
                    if torch.is_tensor(batch[key]):
                        dict_list_aux[key].append(batch[key].cpu().detach().numpy())
                    else:
                        dict_list_aux[key].append(batch[key])
                
                images = images.cuda()
                labels = labels.cuda()
                
                _, embeddings = model(images)
                outputs = regressor(embeddings)
                
                list_outputs.append(outputs.cpu().detach())
                list_labels.append(labels.cpu().detach())
                list_embeddings.append(embeddings.cpu().detach())
        
        outputs = torch.cat(list_outputs, dim=0)
        labels  = torch.cat(list_labels, dim=0)
        embeddings = torch.cat(list_embeddings, dim=0)

        loutputs.append(outputs)
        
        metrics = {
            'r2': r2_score(outputs, labels),
            'l1': torch.nn.L1Loss()(outputs, labels),
            'l2': torch.sqrt(torch.nn.MSELoss()(outputs, labels)),
        }
        
        for key in dict_list_aux.keys():
            dict_list_aux[key] = [element for innerList in dict_list_aux[key] for element in innerList]
        
        if metrics['r2'] >= best_r2:
            best_r2 = copy.deepcopy(metrics['r2'])
            flabels = copy.deepcopy(labels)
            fembeddings = copy.deepcopy(embeddings)
            fdict_list_aux = copy.deepcopy(dict_list_aux)
    
    foutputs = torch.stack(loutputs).mean(dim=0)
    fmetrics = {
        'r2': r2_score(foutputs, flabels),
        'l1': torch.nn.L1Loss()(foutputs, flabels),
        'l2': torch.sqrt(torch.nn.MSELoss()(foutputs, flabels)),
    }
    
    return fmetrics, {'outputs': foutputs, 'labels': flabels, 'embeddings': fembeddings, 'aux': fdict_list_aux}
