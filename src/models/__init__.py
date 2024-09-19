import math
import os
from typing import Any, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric

#  定义了BaseMixin 的类，它继承自 LightningModule
class BaseMixin(LightningModule):
    def __init__(    #  一个 DictConfig 类型的配置，用于实例化模型。
            self, model: DictConfig, optim: DictConfig, sch: Optional[DictConfig] = None,
            infer_output_path: str = ".",   # 一个字符串，指定推理输出的路径，默认为当前目录。
            **kwargs  # 接受其他关键字参数。
    ):
        super().__init__()   # 调用父类 LightningModule 的构造函数。
        import os
        # disable tokenizer fork
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 设置环境变量 TOKENIZERS_PARALLELISM 为 false，以禁用分词器的并行处理。

        self.mean_losses = torch.nn.ModuleDict({    # 创建一个 torch.nn.ModuleDict 对象，用于存储三种不同模式的平均损失（训练、验证和测试）。
            "train_losses": MeanMetric(),
            "valid_losses": MeanMetric(),
            "test_losses": MeanMetric(),
        })
        self.model = hydra.utils.instantiate(   # 使用 Hydra 工具的 instantiate 函数根据提供的 model 配置实例化模型。
            model, _recursive_=False,   # _recursive_=False 参数表示不递归地实例化配置中的所有对象
        )
        os.makedirs(infer_output_path, exist_ok=True)  # 创建推理输出路径的目录，如果目录已存在，则不会引发错误。
        # path to {output_dir}/project/id
        self.infer_output_path = infer_output_path  # 将推理输出路径保存为实例变量。

        self.save_hyperparameters(logger=False)   # 调用 save_hyperparameters 方法，将超参数保存到模型的 hparams 属性中
# 模型的前向传播
    def forward(self, batch):
        return self.model(**batch)   # 它接受一个批次的数据 batch，并将其作为关键字参数传递给模型的 forward 方法。
# 计算损失
    def compute_step(self, batch: dict, split: str):   # 计算一个批次的损失
        loss = self(batch)
        self.log(f"{split}/loss_step", loss, on_step=True,
                 on_epoch=False, prog_bar=False, sync_dist=True, rank_zero_only=True)
        return {"loss": loss}
# 日志记录的功能
    def compute_step_end(self, outputs, split: str):  # 在每个步骤结束时调用
        """
        log, since DDP logging must be put in *_step_end method
        """
        losses = outputs["loss"]  # 从 outputs 字典中提取损失
        self.mean_losses[f"{split}_losses"](losses)  # 使用 mean_losses 字典中的相应指标对象来更新平均损失
        if losses.numel() > 1:  # DP mode  如果损失是一个张量且包含多个元素（这通常发生在数据并行（DP）模式下），它返回损失的平均值
            return losses.mean()
# 用于从一批次的输出中检索特定的键值对，并将它们整理成一个列表或张量
    def retrieve(self, outputs, key: Union[str, List[str]]):
        """
        outputs: 
            [ {"key": value in the whole batch} ] if key is str
            or tuple of above for str-list
        """
        if type(key) is str:   # 检查 key 是否为字符串类型。
            ret = [value for o in outputs for value in o[key]]   # 如果 key 是字符串，那么使用列表推导式从每个批次的输出中提取所有与 key 相关联的值，并将它们组成一个列表。
            if isinstance(ret[0], torch.Tensor) and ret[0].numel() == 1:  # 检查列表中的第一个元素是否是单个元素的张量
                ret = torch.stack(ret).detach().cpu().numpy()  # 用torch.stack 将这些张量堆叠成一个新张量，然后将其从计算图中分离（detach），移动到 CPU（cpu），并将其转换为 NumPy 数组（numpy）。
            if isinstance(ret[0], torch.Tensor) and ret[0].numel() > 1:   # 检查列表中的第一个元素是否是多元素的张量
                ret = torch.cat(ret).detach().cpu().numpy()   # 如果是，则使用 torch.cat 将这些张量连接成一个新张量，然后执行与上面相同的操作。
            return ret
        return [
            self.retrieve(outputs, k)
            for k in key
        ]   # 如果 key 是字符串列表，则对列表中的每个字符串递归调用 retrieve 方法，并将结果组成一个列表返回。
# 在每个 epoch 结束时调用，用于计算和记录一个 epoch 的平均损失
    def agg_epoch(self, outputs: List[Any], split: str):
        loss = self.mean_losses[f"{split}_losses"].compute()   # 调用 mean_losses 字典中对应于当前数据集分割的平均损失对象（例如 train_losses、valid_losses 或 test_losses）的 compute 方法来计算当前 epoch 的平均损失。
        self.mean_losses[f"{split}_losses"].reset()  # 重置 mean_losses 字典中对应于当前数据集分割的平均损失对象，为下一个 epoch 的计算做准备。
        self.log(f"{split}/loss_epoch",
                 loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)   # 使用 self.log 方法记录当前 epoch 的平均损失。这个损失将在整个 epoch 中记录（on_epoch=True），在进度条中显示（prog_bar=True），在分布式训练中同步（sync_dist=True），并且只在 rank 0 进程中记录（rank_zero_only=True）。
# training_step 方法在每次训练批次结束时被调用。它将批次数据和 “train” 字符串传递给 compute_step 方法，后者计算损失并记录。
    def training_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "train")
# training_step_end 方法在每次训练批次结束后被调用。它将批次输出和 “train” 字符串传递给 compute_step_end 方法，后者用于记录和汇总损失。
    def training_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "train")
# training_epoch_end 方法在每个训练 epoch 结束时被调用。它将整个 epoch 的输出列表和 “train” 字符串传递给 agg_epoch 方法，后者计算并记录平均损失。
    def training_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "train")
# validation_step 方法在每次验证批次结束时被调用。它将批次数据和 “valid” 字符串传递给 compute_step 方法，后者计算损失并记录。
    def validation_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "valid")
# validation_step_end 方法在每次验证批次结束后被调用。它将批次输出和 “valid” 字符串传递给 compute_step_end 方法，后者用于记录和汇总损失。
    def validation_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "valid")
# validation_epoch_end 方法在每个验证 epoch 结束时被调用。它将整个 epoch 的输出列表和 “valid” 字符串传递给 agg_epoch 方法，后者计算并记录平均损失。
    def validation_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "valid")
# test_step 方法在每次测试批次结束时被调用。它将批次数据和 “test” 字符串传递给 compute_step 方法，后者计算损失并记录
    def test_step(self, batch: Any, batch_idx: int):
        return self.compute_step(batch, "test")
# test_step_end 方法在每次测试批次结束后被调用。它将批次输出和 “test” 字符串传递给 compute_step_end 方法，后者用于记录和汇总损失。
    def test_step_end(self, outputs: Any):
        return self.compute_step_end(outputs, "test")
# test_epoch_end 方法在每个测试 epoch 结束时被调用。它将整个 epoch 的输出列表和 “test” 字符串传递给 agg_epoch 方法，后者计算并记录平均损失。
    def test_epoch_end(self, outputs: List[Any]):
        return self.agg_epoch(outputs, "test")
# 在 PyTorch Lightning 的训练阶段开始时被调用
    def setup(self, stage: Optional[str] = None) -> None:   # 它接受一个可选参数 stage，该参数指示当前阶段是 “fit”（训练和验证）、“validate”（仅验证）、“test”（仅测试）中的哪一个。
        if stage == "fit":  # 检查 stage 参数是否为 “fit”，即是否处于训练和验证阶段。
            train_loader = self.trainer.datamodule.train_dataloader()   # 从训练器的 datamodule 中获取训练数据加载器 train_loader

            # Calculate total steps
            effective_batch_size = (self.trainer.datamodule.hparams.batch_size *
                                    max(1, self.trainer.num_devices) * self.trainer.accumulate_grad_batches)   # 计算有效批处理大小 effective_batch_size。这是实际传递给模型的批处理大小，它考虑了数据并行（self.trainer.num_devices）和梯度累积（self.trainer.accumulate_grad_batches）。
            self.total_steps = int(
                (len(train_loader.dataset) // effective_batch_size) * float(self.trainer.max_epochs))  # 计算总训练步骤 self.total_steps。这是在整个训练过程中模型将看到的总批次数。它是通过将数据集的长度除以有效批处理大小，并乘以最大训练周期数 self.trainer.max_epochs 来得到的。
# 用于配置模型优化器和学习率调度器。
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]   # 定义了一个列表 no_decay，包含不应应用权重衰减（weight decay）的参数名称。
        wd = self.hparams.optim.pop("weight_decay")  # 从优化器配置中弹出 weight_decay 参数的值，并将其存储在变量 wd 中
        optimizer_grouped_parameters = [   # 创建一个参数组列表 optimizer_grouped_parameters，用于指定哪些参数应该应用权重衰减，哪些不应该。第一个字典包含所有不在 no_decay 列表中的参数，并将应用权重衰减 wd；第二个字典包含所有在 no_decay 列表中的参数，不应用权重衰减。
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": wd},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = hydra.utils.instantiate(   # 使用 hydra.utils.instantiate 方法来实例化优化器  instantiate 方法根据传入的配置创建一个对象。
            self.hparams.optim, params=optimizer_grouped_parameters,   # 这是一个包含优化器配置的对象，通常是从配置文件中读取的。之前创建的 optimizer_grouped_parameters 列表作为参数传递给优化器。
            _convert_="partial"   # Hydra 的一个选项，表示在实例化对象时，只转换那些在配置中明确指定的参数，而不是全部参数。
        )
        if self.hparams.sch is not None:   # 检查是否提供了学习率调度器配置 self.hparams.sch
            ratio = 0.5  # 设置学习率预热比例 ratio，默认为 0.5。如果调度器配置中提供了 warmup_ratio，则使用该值。
            if self.hparams.sch.get("warmup_ratio"):
                ratio = self.hparams.sch.pop("warmup_ratio")
            scheduler = hydra.utils.instantiate(   # 使用 Hydra 工具的 instantiate 方法根据 self.hparams.sch 配置实例化学习率调度器，传递优化器 optimizer 和计算出的预热步骤数 num_warmup_steps 以及总训练步骤数 self.total_steps。
                self.hparams.sch, optimizer=optimizer,
                num_warmup_steps=math.ceil(self.total_steps * ratio), num_training_steps=self.total_steps
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}  # 如果定义了学习率调度器，则返回一个包含优化器和学习率调度器的字典。
        return optimizer  # 如果没有定义学习率调度器，则只返回优化器。

# 为生成模型提供了额外的支持，使得在评估或推理阶段可以通过生成方式进行。
class GenerateMixin(BaseMixin):
    """
    additionally, support eval/infer by generation
    since generator model, will not use traditional logic for {val,test}_step
    """
# 定义了 should_do_generate 方法，它接受一个参数 split，表示当前的数据集分割（例如 “train”、“valid” 或 “test”），并返回一个布尔值，指示是否应该在正向传播期间进行生成。
    def should_do_generate(self, split) -> bool:
        """
        if true, generate during forward
        """
        if split == "test":  # 如果 split 是 “test” 或 “valid”，则返回 True，表示在测试或验证阶段应该进行生成。否则，返回 False。
            return True
        elif split == "valid":
            return True
        return False
# 接受一个参数 batch，预期返回一个由字典组成的列表 (List[dict])。
    def generate_step(self, batch) -> List[dict]:
        """
        return list of (dict of generated results)
        """
        generated: List[dict] = self.model.generate(**batch)  # 首先定义了一个变量 generated，并且指明了其类型为 List[dict]。它通过调用这个类的 model 属性的 generate 方法来获得值。**batch 是一个参数展开操作，意味着这个字典中的键值对将被作为独立的参数传递给 generate 方法。
        return {"generated": generated}
# 该方法接收两个参数：outputs 和 split，并且不返回任何值（None）。
    def generate_epoch(self, outputs, split) -> None:
        """
        eg save generated result or log
        """
        raise NotImplementedError   # 会报错
# 以支持生成模型在验证和测试阶段的特殊行为
    def compute_step(self, batch: dict, split: str):  # 定义了 compute_step 方法，它覆盖了 BaseMixin 类中的同名方法。这个方法在每次训练、验证或测试批次结束时被调用。它接受批次数据 batch 和数据集分割 split 作为参数。
        """
        disable forward for valid / test
        """
        if split in ["valid", "test"]:  # 如果 split 是 “valid” 或 “test”，则检查是否应该进行生成。
            if self.should_do_generate(split):
                return self.generate_step(batch)  # 如果是，则调用 self.generate_step(batch) 方法进行生成。
            return  # 如果不需要生成，则直接返回 None，从而跳过传统的损失计算和指标记录。
        return super().compute_step(batch, split)   # 如果 split 是 “train”，则调用父类 BaseMixin 的 compute_step 方法，执行传统的训练步骤逻辑。

    def compute_step_end(self, outputs, split: str):
        if split in ["valid", "test"]:
            return  # 如果 split 是 “valid” 或 “test”，则直接返回，不做任何处理
        return super().compute_step_end(outputs, split)   # 如果 split 是 “train”，则调用父类 BaseMixin 的 compute_step_end 方法，执行传统的训练步骤结束逻辑。这些方法使得 GenerateMixin 类能够适应生成模型在验证和测试阶段的行为，提供了在需要时切换到生成模式的能力。
# 用于聚合一个训练周期中生成的结果的方法。
    def agg_epoch(self, outputs: List[Any], split: str):
        """
        outputs: List[output from *_step], len=num_batches
        """
        # by default skip eval in trainer, so if indeed in eval mode:
        #   1. should_do_generate(split) => False but only need to save ckpt
        #   2. should_do_generate(split) => True, will eval (eg get adaptive neg sample) but still save ckpt 
        if split == "valid":  # 如果 split 参数为 "valid"，表明当前正在处理验证集的数据。
            os.makedirs(os.path.join(self.infer_output_path, "ckpt"), exist_ok=True)   # 创建一个目录路径，该路径是 self.infer_output_path 和 "ckpt" 的组合。exist_ok=True 参数确保如果目录已存在，不会抛出错误。
            path = os.path.join(self.infer_output_path, "ckpt",
                                f"epoch{self.trainer.current_epoch}-step{self.trainer.global_step}.ckpt")   # 构造一个文件路径，该路径包括检查点文件的名称，其中包含当前的训练周期和全局步骤数。
            self.trainer.save_checkpoint(path)   # 调用 trainer 的 save_checkpoint 方法，将模型的状态保存到上一步构造的路径中。
        if self.should_do_generate(split):  # test or sometimes valid  检查是否应该在当前的数据分割（split）上执行生成操作。
            # go = self.all_gather(outputs)
            # generated = self.all_gather(outputs)
            # # wait untill all processes are sync
            # self.trainer.strategy.barrier()
            if self.trainer.is_global_zero:  # 检查当前的处理器（在分布式训练环境中）是否是全局索引为0的处理器，这通常是控制节点。
                generated = self.retrieve(outputs, "generated")  # 从输出中检索名为 "generated" 的数据。
                # generated = [o["generated"] for o in outputs]
                # (world size, ...) -> (..., )
                # new_outputs = []
                # for output in outputs:
                #     new_outputs.extend(output)
                # outputs = new_outputs
                self.generate_epoch(generated, split)  # 调用 generate_epoch 方法，将生成的数据和分割类型作为参数。
            # wait untill all processes are sync
            # self.trainer.strategy.barrier()
            return  # 如果条件满足，方法会提前返回，不执行后续的代码。
        return super().agg_epoch(outputs, split)  # 如果以上条件不满足，这行代码会调用父类的 agg_epoch 方法，将控制权交给父类处理，可能进行更一般的聚合操作。
