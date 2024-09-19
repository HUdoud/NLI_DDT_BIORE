import os
from typing import List, Optional


import hydra
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src import utils

# 使用 utils 模块中的 get_logger 函数创建一个名为 log 的日志记录器对象，用于记录该模块中的日志信息。
log = utils.get_logger(__name__)


# 函数接受一个字典类型的配置信息 cfg，其中包含了需要实例化的回调函数的配置。
@rank_zero_only
# 用于根据配置信息创建并返回 PyTorch Lightning 回调函数列表。
def get_pl_callbacks(cfg: DictConfig) -> List[Callback]:
    # 函数首先创建一个空列表 callbacks，用于存储实例化后的回调函数。
    callbacks: List[Callback] = []
    # 检查配置对象cfg中是否包含一个名为"callbacks"的键。 函数检查配置中是否存在回调函数，并遍历配置中的每个回调函数
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg["callbacks"].items():
            # 对于每个回调函数，函数会检查其配置是否包含 _target_ 属性，如果存在，则说明需要实例化该回调函数
            if type(cb_conf) is DictConfig and "_target_" in cb_conf:
                log.info(f"Instantiating callback {cb_name} <{cb_conf._target_}>")
                # 如果回调的目标类名包含"ModelCheckpoint"，则进行进一步的检查。
                if "ModelCheckpoint" in cb_conf._target_:
                    # 检查回调配置中的"monitor"参数是否设置为"PLACEHOLDER"，这可能是一个占位符，用于提示用户输入正确的监控指标。
                    if cb_conf["monitor"] == "PLACEHOLDER":
                        # 如果"monitor"参数是"PLACEHOLDER"，则将其设置为"valid/loss_epoch"，这是一个可能的监控指标。
                        cb_conf["monitor"] = "valid/loss_epoch"
                    # 设置回调配置中的"filename"参数，这是一个用于保存检查点的文件名模板。
                    cb_conf["filename"] = r"epoch{epoch:02d}-vl{valid/loss_epoch:.3f}"
                # 函数会根据配置信息实例化回调函数，并将其添加到 callbacks 列表中。
                callbacks.append(hydra.utils.instantiate(cb_conf))
    # 返回创建的回调列表。 回调（Callback）是一个扩展PyTorch Lightning训练循环的工具。
    return callbacks


@rank_zero_only
# 它接受一个类型为 DictConfig 的参数 cfg，并返回一个 LightningLoggerBase 类型的列表。
def get_pl_logger(cfg: DictConfig) -> List[LightningLoggerBase]:
    loggers: List[LightningLoggerBase] = []  # 创建一个空列表 loggers，用于存储实例化的日志记录器对象。
    # 检查配置 cfg 中是否包含键为 "logger" 的项。
    if "logger" in cfg:
        # 遍历配置中 "logger" 键对应的项，其中 lg_conf 是每个日志记录器的配置。
        for _, lg_conf in cfg["logger"].items():
            # 检查日志记录器的配置是否为 DictConfig 类型，并且是否包含 "target" 属性。
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                # 记录日志，表示正在实例化日志记录器。
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                # 使用 Hydra 的 instantiate 函数根据配置实例化日志记录器对象。
                logger = hydra.utils.instantiate(lg_conf)
                # 将实例化的日志记录器对象添加到 loggers 列表中。
                loggers.append(logger)
                # 尝试访问实例化的日志记录器的实验属性，直到成功为止。这样做是为了确保实验对象已经创建，以便后续使用。
                while True:
                    try:
                        # sometimes fail for unknown reason
                        print(logger.experiment)
                        break
                    except BaseException:
                        pass

                # will not be in debug mode as in debug mode logger is deleted
                # if "wandb" in lg_conf["_target_"]:
                #     # will upload this run to cloud in the end of the run
                #     log.info(f"wandb url in {logger.experiment.url}")
                #     project = logger.experiment.project
                #     # get id from x-y-id
                #     id = logger.experiment.name.rsplit('-', 1)[1]
                #
                #     with open_dict(cfg):
                #         cfg.output_dir = os.path.join(
                #             cfg.output_dir, project, id
                #         )
                #         if cfg.get("callbacks") and cfg.callbacks.get("model_checkpoint"):
                #             cfg.callbacks.model_checkpoint.dirpath = os.path.join(
                #                 cfg.output_dir, "ckpt"
                #             )

    return loggers


# 用于测试模型的函数
def test(config, trainer, model, datamodule):
    """
    by default (test_after_training), call after .fit, using best ckpt
    OR
    if test_without_fit:
        skip .fit and use un-tuned ckpt
    if infer_mode
        skip .fit and use provided ckpt (ckpt_path)
    """
    # do not call test in DDP, will lead to in-accurate results
    # 它检查训练器是否在多 GPU 情况下运行，如果是，则销毁进程组，因为在分布式数据并行训练时进行测试会导致不准确的结果。
    if trainer.num_devices > 1:
        torch.distributed.destroy_process_group()
    # 它检查当前进程是否是全局主进程（global zero）。如果不是，则退出。
    if not trainer.is_global_zero:
        import sys
        sys.exit(0)

    # 根据配置和训练器状态确定要使用的检查点路径
    if config.get("test_without_fit"):
        # 如果配置中设置了 test_without_fit，则使用未调整的检查点；如果设置了 infer_mode，则使用提供的检查点路径；否则，默认使用在训练过程中保存的最佳模型检查点。
        ckpt_path = None
    elif config.get("infer_mode"):
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path
    else:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(ckpt_path)
        # assert os.path.exists(ckpt_path)
        log.info(
            f"Best model ckpt at {ckpt_path}")
        # 确定了要使用的检查点路径，则加载模型的参数和超参数，并记录日志。
        if trainer.logger is not None:
            trainer.logger.log_hyperparams({"best_model_path": ckpt_path})
    if ckpt_path:
        # assert os.path.exists(ckpt_path)
        model = type(model).load_from_checkpoint(ckpt_path)  # load hparams etc
    log.info(f"Starting testing using ckpt={ckpt_path}!")

    # recreate new trainer thus get rid of old (potential DDP) trainer, but keep logger so that wandb still continue
    # trainer = Trainer(gpus=1, logger=trainer.logger, limit_test_batches=0.03)
    # 重新创建一个新的训练器对象，以确保之前的训练器对象被销毁。然后使用新的训练器对象执行测试，传入模型、数据模块和检查点路径。
    trainer = Trainer(gpus=1, logger=trainer.logger)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


# 函数的主要功能是使用给定的trainer来预测数据，并可能从检查点文件中恢复模型。
# trainer（一个PyTorch Lightning的Trainer实例）
def predict(config, trainer, model, datamodule):
    # 从配置对象config中获取一个名为"ckpt_path"的键，并将其值赋给变量ckpt_path。
    ckpt_path = config.get("ckpt_path")
    # 检查ckpt_path是否不为None。
    if ckpt_path:
        assert ckpt_path  # 断言ckpt_path不为None
        assert os.path.exists(ckpt_path)   # 断言ckpt_path指定的路径确实存在。
    # 如果ckpt_path为None，则执行以下代码。
    else:
        ckpt_path = None
    print("ckpt_path is", ckpt_path)  # 打印当前的ckpt_path值
    predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)
    # preds = predictions[0] + predictions[1]
    # for split, idx in zip(["train", "test"], [0, 1]):
    #     pred = torch.cat(predictions[idx]) # (#samples, d)
    # torch.save(preds, os.path.join(f"/shares/perception/yufeng/project/EMNLP22/data/{out}/emb",
    #                                f"{config.model.modality}_emb_temporal.pt"))
    print()


# 执行模型的训练过程 函数接收一个DictConfig类型的配置对象，并返回一个可选的浮点数，这个浮点数可能是用于超参数优化的度量分数。
def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    # 检查配置对象config中是否包含一个名为"seed"的键。
    if config.get("seed"):
        # 如果存在"seed"键，则使用该种子值初始化所有PyTorch、NumPy和Python标准库中的随机数生成器。
        seed_everything(config.seed, workers=True)
    # 在日志中记录正在实例化数据模块的信息，并打印其目标类名。
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    # 使用hydra.utils.instantiate函数根据config.datamodule中的配置实例化一个LightningDataModule对象。
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule)

    # 调用get_pl_logger函数，它根据配置对象config返回一个LightningLoggerBase类型的列表，这些是PyTorch Lightning的日志记录器。
    # Init lightning loggers
    logger: List[LightningLoggerBase] = get_pl_logger(config)

    # 用get_pl_callbacks函数，它根据配置对象config返回一个Callback类型的列表，这些是PyTorch Lightning的回调。
    # Init lightning callbacks
    callbacks: List[Callback] = get_pl_callbacks(config)

    # 在日志中记录正在实例化模型的信息，并打印其目标类名。
    log.info(f"Instantiating model <{config.model._target_}>")
    # 使用hydra.utils.instantiate函数根据config.model中的配置实例化一个LightningModule对象。这里使用了_recursive_=False参数，这意味着不会递归地解析模型配置中的优化器和调度器等配置。
    model: LightningModule = hydra.utils.instantiate(
        # non recursive so that optim and scheulder can be passed as DictConfig
        config.model, _recursive_=False
    )

    # 在日志中记录正在实例化训练器的信息，并打印其目标类名。
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # 使用hydra.utils.instantiate函数根据config.trainer中的配置实例化一个Trainer对象。这里使用了_convert_="partial"参数，这可能意味着它不会尝试自动解析callbacks和logger中的配置，而是直接使用传入的实例。
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # 在日志中记录正在记录超参数的信息
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    # 调用log_hyperparameters函数，它将配置、模型、数据模块、训练器、回调和日志记录器的信息记录下来。
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 检查配置对象config中是否包含一个名为"infer_mode"的键。
    if config.get("infer_mode"):
        # 如果存在"infer_mode"键并且值为True，则执行以下代码。
        if config.get("test_only"):
            # 调用test函数，它可能用于在训练完成后进行测试。
            test(config, trainer, model, datamodule)
        else:
            # 调用predict函数，它可能用于在训练完成后进行预测。
            predict(config, trainer, model, datamodule)
        return None  # 返回None，因为训练过程没有进行，或者超参数优化不需要度量分数。

    # Train the model
    # 在日志中记录开始训练的信息
    log.info("Starting training!")
    # 初始化一个变量ckpt_path，用于保存检查点路径。
    ckpt_path = None
    # 检查配置对象config中是否包含一个名为"resume_from_ckpt"的键。
    if config.get("resume_from_ckpt"):
        # 如果存在"resume_from_ckpt"键，则将其值赋给ckpt_path。
        ckpt_path = config.get("resume_from_ckpt")
        assert os.path.exists(ckpt_path)   # 断言ckpt_path指定的路径确实存在。
    # 使用trainer实例调用fit方法，传递model、datamodule和ckpt_path。这将开始模型的训练过程。
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # 从配置对象config中获取一个名为"optimized_metric"的键，并将其值赋给变量optimized_metric。
    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    # 检查optimized_metric是否不为None，并且是否在训练器的回调度量中。
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        # 如果optimized_metric在训练器的回调度量中不存在，则抛出一个异常。
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    # 从训练器的回调度量中获取optimized_metric对应的值
    score = trainer.callback_metrics.get(optimized_metric)

    # 测试模型
    # Test the model
    # 检查配置对象config中是否包含一个名为"test_after_training"的键，并且是否设置了为True，以及是否设置了"fast_dev_run"为False。如果这两个条件都满足，则执行以下代码。
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        # 调用test函数，它可能用于在训练完成后进行测试
        test(config, trainer, model, datamodule)

    # Make sure everything closed properly
    # 在日志中记录正在完成资源清理的信息
    log.info("Finalizing!")
    # 用finish函数，它可能用于在训练完成后进行一些清理工作，例如关闭文件、释放内存等。
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,

    )

    # Return metric score for hyperparameter optimization
    # 返回从训练器的回调度量中获取的optimized_metric对应的值。如果optimized_metric不为None，则返回其对应的值；否则，返回None。
    return score
