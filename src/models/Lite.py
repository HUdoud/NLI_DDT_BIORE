import json
import os
from collections import defaultdict
from itertools import chain
from typing import Any, List, Optional

import torch

from .components import LITE

from . import GenerateMixin

# 似乎不包含验证或测试阶段，但会保存置信度，这意味着可能需要在训练后对结果进行后处理。
class LiteModule(GenerateMixin):
    """
    only training, no validation / test
        but will save confidence, need postprocess

    https://arxiv.org/abs/2202.06167

    https://github.com/luka-group/lite
    """

    def __init__(self, adaptive_neg_sample: bool = False, num_negs: int = 1, **kwargs):  # 接受以下参数：adaptive_neg_sample：一个布尔值，指示是否自适应地选择难负样本。num_negs：一个整数，表示每个正样本的负样本数量。**kwargs：一个字典，包含其他传递给父类GenerateMixin的参数。
        """
        @adaptive_neg_sample: whether choose hard neg sample adaptively
        """
        super().__init__(**kwargs)  # 调用父类的__init__方法，传递**kwargs参数
        self.adaptive_neg_sample = adaptive_neg_sample  # 将adaptive_neg_sample参数保存为类的属性。
        # store (latest gen from trainset (through val dataloader), latest gen's epoch)
        # only used in adaptive_neg_sample=True
        self.current_train_gen = (None, None)  # 初始化一个元组self.current_train_gen，用于存储最新的训练集生成的数据和对应的周期号。这个属性只在adaptive_neg_sample=True时使用。
        self.num_negs = num_negs  # 将num_negs参数保存为类的属性。
        self.model: LITE  # 声明了一个名为model的属性，其类型为LITE。
# 用于控制何时生成输出
    def should_do_generate(self, split) -> bool:  # 定是否在模型的前向传播过程中生成输出。它接受一个参数split，表示当前的数据集分割（例如，训练、验证或测试），并返回一个布尔值。
        """
        if true, generate during forward
        by default, only generate when
            1. test
            2. doing adaptive neg sampling (only in valid)
        """
        if split == "test":  # 如果split是"test"（测试集），则返回True。
            return True
        if split == "valid" and self.adaptive_neg_sample:  # 如果split是"valid"（验证集）且self.adaptive_neg_sample（自适应负采样）为True，也返回True。否则，返回False。
            return True
        return False
# 如何处理和保存生成的输出
    def generate_epoch(self, outputs: List[dict], split) -> None:   # 它接受两个参数：outputs，一个包含多个批次输出的字典列表；split，当前的数据集分割。
        """
        save to @self.infer_output_path
        outputs: List[dict], len=num_batches
            might need to transfer to torch cpu again by `all_gather`
        """
        gen_by_dataset = defaultdict(list)  # 创建一个默认字典gen_by_dataset，用于按数据集名称存储生成的输出。
        for d in outputs:
            """
            convert to python object
            """
            if d["confidence"]:   # 如果d中包含"confidence"键，则将其值转换为Python对象。
                if type(d["confidence"][0]) is torch.Tensor:
                    d["confidence"] = list(map(lambda t: t.detach().item(), d["confidence"]))   # 如果"confidence"的第一个元素是张量，则将其转换为Python浮点数。
            if type(d["id"]) is torch.Tensor:   # 如果d中的"id"键对应的值是张量，则将其转换为Python整数。
                d["id"] = d["id"].detach().item()
            gen_by_dataset[d["dataset_name"]].append(d)   # 将处理后的d添加到gen_by_dataset字典中，按照d["dataset_name"]分类。
        """
        save results
        """
        if self.adaptive_neg_sample and split == "valid":   # 检查是否启用了自适应负采样（self.adaptive_neg_sample）并且当前的数据集分割是验证集（split == "valid"）。
            # not true valid, input data is in fact train
            # so that can calc adaptive neg example
            save_path = os.path.join(self.infer_output_path, "adaptive")  # 创建一个保存路径，用于保存自适应负样本的生成数据。
            """
            update current_train_gen
            {id => dict}
            """
            assert len(gen_by_dataset) == 1, "should only have 1 dataset in training"  # 代码断言gen_by_dataset字典中只有一个数据集。这是因为在训练中通常只有一个数据集。
            train_gen: List[dict] = gen_by_dataset[next(iter(gen_by_dataset))]  # 这行代码获取gen_by_dataset字典中的第一个数据集，并将其转换为一个列表。
            self.current_train_gen = ({  # 更新self.current_train_gen属性，其中包含了一个字典，键是数据集的ID，值是数据集的生成数据，以及当前的周期号。
                                          g["id"]: g for g in train_gen
                                      }, self.trainer.current_epoch)
        else:
            save_path = os.path.join(self.infer_output_path, "infer")  # 上述条件不满足，即不是在验证集上进行自适应负采样，则创建一个不同的保存路径，用于保存推断数据。
        os.makedirs(save_path, exist_ok=True)  # 创建保存路径的目录，如果目录已经存在，则不会引发错误。
        for name, gen in gen_by_dataset.items():  # 循环遍历gen_by_dataset字典中的每个数据集名称和对应的生成数据。
            path = os.path.join(save_path, f"{name}_generated{self.trainer.current_epoch}_{split}.jsonl")
            print("saving to ", path)   # 对于每个数据集，它创建一个文件路径，并将生成的数据写入该文件。每个文件以jsonl格式保存，这意味着每行都是一个JSON对象。
            with open(path, "w") as f:
                for g in gen:
                    f.write(json.dumps(g) + "\n")
# 用于处理每个批次数据的一个训练步骤
    def training_step(self, batch: Any, batch_idx: int):   # 这行定义了一个名为 training_step 的方法，它接受两个参数：batch 是一个包含批次数据的变量，batch_idx 是当前批次的索引。
        """
        add neg samples to @batch
        """
        if self.adaptive_neg_sample:  # 检查一个类属性 adaptive_neg_sample 是否为 True。如果是，表示使用自适应负样本生成策略；否则，使用常规的随机负样本生成策略。
            cur_epoch = self.trainer.current_epoch  # 0 based  获取当前训练周期的索引（基于0）。
            train_gen, last_epoch = self.current_train_gen  # 从类属性 current_train_gen 中提取当前训练生成器 train_gen 和上一个使用该生成器的训练周期 last_epoch。
            if train_gen is None:  # no val in the beginning  如果 train_gen 为空（例如在训练开始时还没有初始化），则使用数据模块的 random_neg_sample 方法生成负样本。
                batch = self.trainer.datamodule.random_neg_sample(batch, self.num_negs)  # 调用数据模块的 random_neg_sample 方法添加指定数量 self.num_negs 的随机负样本到批次 batch 中。
            else:  # 如果 train_gen 不为空，说明已经有可用的负样本生成策略。
                train_gen: dict
                last_epoch: int
                ith = cur_epoch - last_epoch - 1  # eg last_epoch + 1 (next) use 0th  计算从上一次使用负样本生成器以来过了多少个训练周期，并据此决定使用哪一组负样本配置。
                batch = self.trainer.datamodule.adaptive_neg_sample(batch, ith, train_gen, self.num_negs)  # 调用数据模块的 adaptive_neg_sample 方法，根据当前周期和负样本生成器 train_gen，为批次 batch 添加自适应负样本。
        else:  # normal random neg sample  如果 adaptive_neg_sample 属性为 False，则总是使用随机负样本生成策略。
            batch = self.trainer.datamodule.random_neg_sample(batch, self.num_negs)

        return super().training_step(batch, batch_idx)  # 调用父类的 training_step 方法进行后续的训练处理，将更新后的批次数据和批次索引传递给它。