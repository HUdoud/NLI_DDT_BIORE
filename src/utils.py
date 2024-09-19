import logging
import os
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.utilities import rank_zero_only

try:
    import transformers  # 尝试导入transformers 库

    transformers.logging.set_verbosity_error()
except:
    pass


# 其作用是检查配置字典中是否存在缺失的配置项，并在发现缺失项时抛出异常。
def fail_on_missing(cfg: DictConfig) -> None:
    #  使用 isinstance 函数检查参数 cfg 是否是 ListConfig 类型的对象。如果是，则表示当前配置项是一个列表类型。
    if isinstance(cfg, ListConfig):
        # 如果配置项是一个列表，就遍历列表中的每个元素 x。
        for x in cfg:
            # 对列表中的每个元素递归调用 fail_on_missing 函数，以检查列表中的每个元素是否存在缺失项。
            fail_on_missing(x)
    # 如果配置项不是列表类型，而是 DictConfig 类型的对象，表示当前配置项是一个字典类型。
    elif isinstance(cfg, DictConfig):
        # 遍历字典中的每一对键值对，其中 _ 表示键，v 表示对应的值。
        for _, v in cfg.items():
            # 对每个值递归调用 fail_on_missing 函数，以检查字典中的每个值是否存在缺失项。
            fail_on_missing(v)

# 作用是创建一个多 GPU 友好的 Python 命令行日志记录器
def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    # 这是一个 for 循环，遍历了一个包含不同日志级别的字符串列表。
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # 这是一个 for 循环，遍历了一个包含不同日志级别的字符串列表。
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        # 对于循环中的每个日志级别，使用 setattr 函数动态地将 logger 对象的属性设置为与该日志级别对应的函数，该函数经过 rank_zero_only 装饰器修饰。这样做是为了确保在多GPU设置中，日志级别不会因为每个GPU进程而重复记录。
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

# 用于在 Weights & Biases (wandb) 平台上查找特定的运行（run）记录
def get_wandb_run(entity, project, id):
    import wandb
    api = wandb.Api(timeout=20)
    run_path = None
    for run in api.runs(f"{entity}/{project}"):
        if id in run.name:
            run_path = os.path.join(*run.path)
            break
    assert run_path is not None, f"{entity}/{project}/{id} is not in the wandb"
    return run_path

# 使用 Weights & Biases (wandb) 工具从云存储中检索并恢复先前的实验配置，并与当前的配置覆盖合并。
def restore(config: DictConfig) -> DictConfig:
    """
    assume using wandb, fetch cfg saved in online wandb and merge with current override
    use whole config stored in wandb cloud plus current override
    NOTE non-override field load from "default" config file, thus if
        config file changed, can't reproduce
        when config file change, can download code sync in wandb
    return saved cfg
    """
    assert "wandb" in config.logger
    # e.g. <project>:<id>
    project, id = config.restore_from_run.split(":")
    run_path = get_wandb_run(config.logger.wandb.entity, project, id)
    orig_override: ListConfig = OmegaConf.load(
        wandb.restore("hydra/overrides.yaml", run_path=run_path))
    current_overrides = HydraConfig.get().overrides.task
    # concatenating the original overrides with the current overrides
    # current override orig if has conflict since it's in 2nd part
    overrides: DictConfig = orig_override + current_overrides

    # getting the config name from the previous job.
    hydra_config = OmegaConf.load(
        wandb.restore("hydra/hydra.yaml", run_path=run_path))
    orig_config = OmegaConf.load(
        wandb.restore("hydra/config.yaml", run_path=run_path))
    orig_config.merge_with(hydra_config)
    config_name: str = orig_config.hydra.job.config_name

    cfg = compose(config_name, overrides=overrides)

    return cfg


def extras(config: DictConfig) -> DictConfig:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    # append necessary config
    # 它允许修改配置字典中的值而不会触发配置文件重写。在这段代码中，它的作用是修改了配置字典中的 hydra_dir 键值对，将其设为当前工作目录的绝对路径。
    with open_dict(config):
        config['hydra_dir'] = to_absolute_path(os.getcwd())

    # restore if needed
    # 检查配置对象config中是否包含一个名为"restore_from_run"的键，并且该键的值不是None。如果是，则执行以下代码。
    if config.restore_from_run is not None:
        # 调用restore函数，它可能用于从之前的运行中恢复配置。这可能是为了继续之前的训练过程。
        config = restore(config)
    # ensure no ??? inside config
    # 调用fail_on_missing函数，它可能用于检查配置中是否有未定义的键，并在发现未定义的键时抛出错误
    fail_on_missing(config)

    # 它可能用于获取一个日志对象，用于记录日志信息。这里使用当前模块的名称作为参数。
    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    # 检查配置对象config中是否包含一个名为"ignore_warnings"的键，并且该键的值为True。如果是，则执行以下代码。
    if config.get("ignore_warnings"):
        # 在日志中记录正在禁用Python警告的信息。
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        # 使用warnings.filterwarnings函数禁用Python警告。这通常用于在调试过程中，以避免警告信息干扰调试过程
        warnings.filterwarnings("ignore")

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    # 检查配置对象config中是否包含一个名为"fast_dev_run"的键，并且该键的值为True。如果是，则执行以下代码。
    if config.trainer.get("fast_dev_run"):
        # 检查数据模块配置中是否包含一个名为"pin_memory"的键。
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False  # 如果"pin_memory"键存在并且值为True，则将其设置为False。这是为了确保调试器友好的配置，因为调试器可能不支持GPU内存对齐。
        # 检查数据模块配置中是否包含一个名为"num_workers"的键。
        if config.datamodule.get("num_workers"):
            # 如果"num_workers"键存在，则将其设置为0。这是为了确保调试器友好的配置，因为调试器可能不支持多进程。
            config.datamodule.num_workers = 0

    # 检查配置对象config中是否包含一个名为"debug_mode"的键，并且该键的值为True
    if config.get("debug_mode"):
        # in debug mode filter out PIL DEBUG log
        # 设置PIL（Python Imaging Library）日志记录器的级别为WARNING。这是为了在调试模式下过滤掉PIL的DEBUG日志，以避免干扰调试过程。
        logging.getLogger('PIL').setLevel(logging.WARNING)

    """
    by default in infer mode, even if logger exist, will switch to local mode
    unless resume_wandb_run is set
    """
    # 检查配置对象config中是否包含一个名为"infer_mode"的键
    if config.get("infer_mode"):
        # 检查配置对象config中是否包含一个名为"logger"的键，并且该键的值中是否包含"wandb"
        if config.get("logger") and "wandb" in config.logger:
            # 将config.logger.wandb.offline设置为True。这可能意味着在推断模式下，Wandb日志记录器将使用离线模式。
            config.logger.wandb.offline = True

    # 检查配置对象config中是否包含一个名为"logger"的键
    if "logger" in config:
        # 检查配置对象config.logger中是否包含一个名为"resume_wandb_run"的键，并且是否包含一个名为"wandb"的键。如果是，则执行以下代码。
        if config.logger.get("resume_wandb_run") and config.logger.get("wandb"):
            # 将config.logger.resume_wandb_run的值按照冒号分割，得到project和id。
            project, id = config.logger.resume_wandb_run.split(":")
            # entity/project/uuid
            # 调用get_wandb_run函数，它可能用于获取Wandb运行的路径。
            run_path = get_wandb_run(config.logger.wandb.entity, project, id)
            # 检查config.logger.wandb.offline是否为True。
            if config.logger.wandb.offline:
                # 如果config.logger.wandb.offline为True，则将其设置为False。
                config.logger.wandb.offline = False
            # 将run_path的值按照斜杠分割，并获取最后一个元素作为config.logger.wandb.id的值。
            config.logger.wandb.id = run_path.split("/")[-1]
            # 使用open_dict函数打开config对象的可变字典。这可能意味着接下来将修改config对象。
            with open_dict(config):
                # 将config.logger.wandb.resume设置为"must"。这可能意味着Wandb日志记录器将尝试恢复之前的运行。
                config.logger.wandb.resume = "must"
    return config


# 其作用是打印 DictConfig 的内容，并使用 Rich 库的树形结构进行展示。
@rank_zero_only
def print_config(
        # Hydra 组合的配置。
        config: DictConfig,
        # 确定要打印的主字段及其顺序，默认为一组字段。
        fields: Sequence[str] = (
                "trainer",
                "callbacks",
                "logger",
                "model",
                "datamodule",
        ),
        # 是否解析 DictConfig 的引用字段，默认为 True。
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    # 设置树形结构的样式。
    style = "dim"
    # 创建了一个名为 "CONFIG" 的树形结构对象。
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    # 遍历指定的字段列表
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style) # 在树形结构中添加一个分支，表示字段名。

        # 获取指定字段名对应的配置段。
        config_section = config.get(field)
        # 将配置段转换为字符串。
        branch_content = str(config_section)
        # 如果配置段是 DictConfig 类型，则将其转换为 YAML 格式。
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        # 在分支下添加配置段的内容，使用 YAML 语法高亮显示。
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    # others defined in root
    # 在树形结构中添加一个名为 "others" 的分支，用于存放除了指定字段之外的其他字段。
    others = tree.add("others", style=style, guide_style=style)
    # 遍历配置中的其他字段，将其添加到 "others" 分支中。
    for var, val in OmegaConf.to_container(config, resolve=True).items():
        if not var.startswith("_") and var not in fields:
            others.add(f"{var}: {val}")

    rich.print(tree)   # 使用 Rich 库打印树形结构。

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


# 用于记录超参数和模型参数信息，并将其传递给 Lightning 的日志记录器。
@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of model parameters
    """
    if trainer.logger is None:
        # only log when has logger
        return

    if config.logger.get("resume_wandb_run") and config.logger.get("wandb"):
        # don't log if resume
        return
    hparams = dict(config)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()