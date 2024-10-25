import hydra
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.model_selection import KFold
from source.model import model_factory
from source.training import training_factory
import datetime
from datetime import timezone, timedelta
from source.dataset import dataset_factory, dataloader_factory
from source.components import optimizers_factory, lr_scheduler_factory
import torch
from source.utils.util import set_seed
from source.components.logger import logger_factory
import os
import pandas as pd

os.environ["WANDB_CACHE_DIR"] = '/home/user/data/tmp'
os.environ['WANDB_DIR'] = '/home/user/data/tmp'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HYDRA_FULL_ERROR'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def model_training(cfg: DictConfig, dataloaders, fold=None):
    device = torch.device('cuda:' + str(cfg.cuda) if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = model_factory(cfg).to(device)

    # logger_factory返回的是一个logging.logger对象
    logger = logger_factory(cfg)

    # 检查模型是否在GPU上
    # if next(model.parameters()).is_cuda:
    #     print("Model is on GPU")
    # else:
    #     print("Model is on CPU")

    # 加载优化器，返回的是一个List[torch.optim.Optimizer]对象
    optimizers = optimizers_factory(model=model, optimizer_configs=[cfg.optimizer])

    # 加载学习率调度器，返回的是一个List[torch.optim.lr_scheduler._LRScheduler]对象
    lr_schedulers = lr_scheduler_factory(lr_configs=[cfg.optimizer], cfg=cfg)

    # 加载训练器
    training = training_factory(cfg=cfg,
                                model=model,
                                optimizers=optimizers,
                                lr_schedulers=lr_schedulers,
                                dataloaders=dataloaders,
                                logger=logger,
                                fold=fold)

    # 开始训练
    results = training.train()

    return results


# 使用hydra的配置文件读取配置组件，hydra会自动将conf/config.yaml中的配置参数传入main函数
# 利用hydra.main()装饰器，将main函数转换为hydra应用
# 直接从命令行运行python -m source，即可运行main函数
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 设置随机种子
    set_seed(cfg.seed)

    # 生成time_id唯一标识
    SHA_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai')
    time_id = datetime.datetime.now().astimezone(SHA_TZ).strftime('%Y-%m-%d-%H-%M-%S')

    with open_dict(cfg):
        cfg.time_id = time_id
        cfg.ckpt_path = f"{cfg.ckpt_path}/{cfg.time_id}"

    if os.path.exists(cfg.ckpt_path) is False:
        os.makedirs(cfg.ckpt_path)

    # 这里要写wandb的命名
    if cfg.use_wandb:
        args_of_wandb = {}
        tags = []
        notes = ""
        wandb.init(project=cfg.wandb.project,
                   name=cfg.wandb.name,
                   config=args_of_wandb,
                   tags=tags,
                   entity=cfg.wandb.entity,
                   notes=notes,
                   reinit=False)

        # 这里要把wandb中的同名参数传给cfg

    # 加载数据集，数据集的元素是torch_geometric.data.Data对象，
    # 返回labels和sites用于分割数据集
    dataset, labels, sites = dataset_factory(cfg)

    results_df = pd.DataFrame(columns=[
        "Accuracy", "AUC", "F1", "Precision", "Recall", "Specificity", "Sensitivity"
    ])

    # 使用交叉验证
    if cfg.cross_validation:

        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

        for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
            dataloaders = dataloader_factory(cfg, dataset, labels, sites, train_index, test_index)
            results = model_training(cfg, dataloaders, fold)
            results_df.loc[fold] = results

    # 不使用交叉验证
    else:

        dataloaders = dataloader_factory(cfg, dataset, labels, sites)
        results = model_training(cfg, dataloaders)
        results_df.loc[0] = results

    print(results_df)

    # 打印最后的结果
    print("\n Final Results: \n")

    print(" | ".join([f'Accuracy: {results_df.Accuracy.mean():.4f}±{results_df.Accuracy.std():.4f}',
                      f'AUC: {results_df.AUC.mean():.4f}±{results_df.AUC.std():.4f}',
                      f'F1: {results_df.F1.mean():.4f}±{results_df.F1.std():.4f}',
                      f'Precision: {results_df.Precision.mean():.4f}±{results_df.Precision.std():.4f}',
                      f'Recall: {results_df.Recall.mean():.4f}±{results_df.Recall.std():.4f}',
                      f'Specificity: {results_df.Specificity.mean():.4f}±{results_df.Specificity.std():.4f}',
                      f'Sensitivity: {results_df.Sensitivity.mean():.4f}±{results_df.Sensitivity.std():.4f}'
                      ]))

    # 如果使用wandb，结束wandb进程
    if cfg.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
