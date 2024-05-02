import torch

from .uniformer import uniformer_small

def get_feature_extractor(model_name, pretrained_weights=None):
    if model_name == 'uniformer_small':
        model = uniformer_small()
        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            model.load_state_dict(state_dict)
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1)
        model.head.bias.data[0] = 55.6
    else:
        raise ValueError
    return model