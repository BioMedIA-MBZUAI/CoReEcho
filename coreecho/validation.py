import torch

from torcheval.metrics.functional import r2_score

def validate(val_loader, model, regressor):
    model.eval()
    regressor.eval()
    
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
    
    metrics = {
        'r2': r2_score(outputs, labels),
        'l1': torch.nn.L1Loss()(outputs, labels),
        'l2': torch.sqrt(torch.nn.MSELoss()(outputs, labels)),
    }
    
    for key in dict_list_aux.keys():
        dict_list_aux[key] = [element for innerList in dict_list_aux[key] for element in innerList]
    
    return metrics, {'outputs': outputs, 'labels': labels, 'embeddings': embeddings, 'aux': dict_list_aux}
