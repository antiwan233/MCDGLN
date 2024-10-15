import logging
from collections import defaultdict
from typing import List
from omegaconf import DictConfig
import torch


# 此函数接收一个 torch.nn.Module 类型的模型以及两个可选字符串参数 match_rule 和 except_rule。
# 它会遍历模型的所有模块和参数，并根据规则分类参数，分为需要权重衰减和不需要权重衰减的两组。
def get_param_group_no_wd(model: torch.nn.Module, match_rule: str = None, except_rule: str = None):
    param_group_no_wd = []
    names_no_wd = []
    param_group_normal = []

    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        if match_rule is not None and match_rule not in name:
            continue
        if except_rule is not None and except_rule in name:
            continue
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.BatchNorm2d) \
                or isinstance(m, torch.nn.BatchNorm1d):
            if m.weight is not None:
                param_group_no_wd.append(m.weight)
                names_no_wd.append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1

    for name, p in model.named_parameters():
        if match_rule is not None and match_rule not in name:
            continue
        if except_rule is not None and except_rule in name:
            continue
        if name not in names_no_wd:
            param_group_normal.append(p)

    params_length = len(param_group_normal) + len(param_group_no_wd)
    logging.info(f'Parameters [no weight decay] length [{params_length}]')
    return [{'params': param_group_normal}, {'params': param_group_no_wd, 'weight_decay': 0.0}], type2num


# 此函数接收一个 torch.nn.Module 类型的模型和一个 DictConfig 类型的优化器配置
# 它创建并返回一个优化器实例。
def optimizer_factory(model: torch.nn.Module, optimizer_config: DictConfig) -> torch.optim.Optimizer:
    parameters = {
        'lr': optimizer_config.lr,
        'weight_decay': optimizer_config.weight_decay
    }

    # 如果optimizer_config.no_weight_decay为True，那么就会执行下面的语句
    # 从而将参数分为需要权重衰退和不需要权重衰退的两组
    if optimizer_config.no_weight_decay:
        params, _ = get_param_group_no_wd(model,
                                          match_rule=optimizer_config.match_rule,
                                          except_rule=optimizer_config.except_rule)

    # 否则，打印params的长度
    else:
        params = list(model.parameters())
        logging.info(f'Parameters [normal] length [{len(params)}]')

    parameters['params'] = params

    optimizer_type = optimizer_config.name
    if optimizer_type == 'SGD':
        parameters['momentum'] = optimizer_config.momentum
        parameters['nesterov'] = optimizer_config.nesterov

    # getattr() 函数用于返回一个对象属性值。
    # 例如，getattr(torch.optim, 'SGD')，就是返回torch.optim.SGD
    # 然后利用**parameters，将parameters中的所有参数传递给torch.optim.SGD，并返回一个 optimizer
    return getattr(torch.optim, optimizer_type)(**parameters)


def optimizers_factory(model: torch.nn.Module, optimizer_configs: List[DictConfig]) -> List[torch.optim.Optimizer]:
    if model is None:
        return None
    return [optimizer_factory(model=model, optimizer_config=single_config) for single_config in optimizer_configs]
