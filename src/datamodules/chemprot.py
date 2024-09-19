import json
import os
import random
from collections import defaultdict
from typing import List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


# 定义一个名为ChemProt_Dataset的类，它继承自torch.utils.data.Dataset。这意味着这个类可以被用于PyTorch的数据加载器（DataLoader）。
class ChemProt_Dataset(Dataset):
    # 定义类的初始化方法__init__，它接受两个参数：data_dir（数据集的目录）和split（数据集的分割，如’train’、‘dev’或’test’）
    def __init__(self, data_dir, split: str):
        # 检查data_dir是否包含"_new"。如果不包含，则将data_dir添加"_new"。
        if "_new" not in data_dir:
            data_dir = data_dir + "_new"
        # 打开data_dir目录下split.json文件，其中split是’train’、‘dev’或’test’。
        with open(os.path.join(data_dir, f"{split}.json")) as f:
            """
            'id': '23264615.T14.T42'
            'sentence': 'In study 1, cyclic ewes received vehicle, cortisol, PF 915275 (PF; a selective inhibitor of HSD11B1), cortisol and PF, meloxicam (a selective inhibitor of PTGS2), cortisol and meloxicam, recombinant ovine IFNT, or IFNT and PF into the uterus from day 10 to day14 after estrus.', 
            'label': '0'
            """
            # 读取文件中的每一行，将其解析为JSON对象，并将所有对象存储在self.data列表中。
            self.data = [
                json.loads(line)
                for line in f
            ]
            # 遍历self.data列表中的每个条目
            for d in self.data:
                # 检查条目的"label"是否为’0’。
                if d["label"] == '0':
                    d["label"] = 'no answer'  # 如果"label"为’0’，则将其替换为’no answer’。
                # 检查条目的"sentence"是否同时包含"@CHEMICAL$"和"@GENE$"。
                if "@CHEMICAL$" in d["sentence"] and "@GENE$" in d["sentence"]:
                    # always chem
                    # 如果包含"@CHEMICAL$"，则找到它在"sentence"中的开始位置。
                    start = d["sentence"].find("@CHEMICAL$")
                    # 计算"@CHEMICAL$"在"sentence"中的结束位置。
                    end = start + len("@CHEMICAL$")
                    # 检查d["sentence"]中从start位置到end位置的字符串是否等于"@CHEMICAL$"。
                    assert d["sentence"][start:end] == "@CHEMICAL$"
                    # 将"entity1"设置为包含"@CHEMICAL$"、'CHEMICAL’和位置的列表。
                    d["entity1"] = ["@CHEMICAL$", "CHEMICAL", start, end]

                    # 如果包含"@GENE$"，则找到它在"sentence"中的开始位置。
                    # always gene
                    start = d["sentence"].find("@GENE$")
                    # 计算"@GENE$"在"sentence"中的结束位置。
                    end = start + len("@GENE$")
                    assert d["sentence"][start:end] == "@GENE$"
                    # 将"entity2"设置为包含"@GENE$"、'GENE’和位置的列表。
                    d["entity2"] = ["@GENE$", "GENE", start, end]
                else:
                    # in fact in dev and test always no answer
                    # 15 non-'no answer' (CPR:9) in train
                    assert "@CHEM-GENE$" in d["sentence"]
                    # 找到"@CHEM-GENE$"在"sentence"中的开始位置
                    start = d["sentence"].find("@CHEM-GENE$")
                    end = start + len("@CHEM-GENE$")
                    d["entity1"] = ["@CHEM-GENE$", "CHEM-GENE", start, end]  # d["entity1"]设置为一个包含上述信息的列表
                    d["entity2"] = [None, None, None, None]
                d["is_none"] = d["label"] == 'no answer'   # d["is_none"] 则用于标记这个样本的标签是否为 "no answer"，如果是则为 True，否则为 False。

    # 定义了如何从数据集中获取单个样本。另一个特殊的方法__len__定义了数据集的长度，即数据集中包含的样本数量。
    def __getitem__(self, idx) -> dict:
        # 从self.data列表中获取索引为idx的条目，并将其赋值给变量ret
        ret: dict = self.data[idx]
        # 将"chemprot"添加到ret字典中，作为键"dataset_name"的值。
        ret["dataset_name"] = "chemprot"
        # 返回修改后的ret字典，其中包含原始数据条目以及新添加的"dataset_name"键。
        return ret

    # 它返回一个整数值，表示数据集的长度
    def __len__(self):
        # 返回self.data列表的长度，即数据集中的样本数量
        return len(self.data)

    # 如何将多个样本组合成一个批次
    @staticmethod
    # 定义collate_fn方法，它接受一个List[dict]类型的参数，其中每个字典代表一个样本。
    def collate_fn(samples: List[dict]):
        # 创建一个defaultdict对象，它的默认值是一个列表。这将用于存储所有样本中相同键的值。
        ret = defaultdict(list)
        # 遍历samples列表中的每个样本
        for sample in samples:
            # 遍历当前样本中的每个键值对
            for k, v in sample.items():
                # 将当前样本的键k对应的值v添加到ret字典中k对应的列表中。
                ret[k].append(v)
        # 将DataModule类的LABEL_VERBALIZER属性添加到ret字典中，键为"LABEL_VERBALIZER"。
        ret["LABEL_VERBALIZER"] = DataModule.LABEL_VERBALIZER
        # 返回修改后的ret字典，其中包含了所有样本中相同键的值以及"LABEL_VERBALIZER"键。
        return ret    # ret是字典


# 目的是根据输入的单词（w1 和 w2）、描述性文本（verbalized）、是否避免做出直接结论（abstain）以及假设的版本（hypo_version）来生成假设模板
def hypo_template(w1, w2, verbalized, abstain=False, hypo_version="v1"):
    if abstain:
        # 这个参数用于描述这两个实体之间的关系，例如 "no answer" 表示没有明确的答案
        if verbalized == "no answer":
            hypo = f"There is no relation between {w1} and {w2}."  # 输出将表示两个单词或实体之间没有关系
        else:
            hypo = f"Relation exists between {w1} and {w2}."   # 否则，输出表示它们之间存在某种关系。
        # 将字符串 hypo 的首字母转换为大写，并保持其余部分不变，然后返回这个结果。
        return hypo[0].upper() + hypo[1:]

    # hypernoym  如果version=v1 则执行以下代码
    if hypo_version == "v1": # hypernoym
        # only used in use_new_mask mode
        # w1: "@CHEM-GENE$"   # 代码检查变量 w2 是否为 None。如果 w2 是 None，则认为只有一个实体 w1 需要处理，没有第二个实体来比较或关联。
        if w2 is None:  # w1: "@CHEM-GENE$"
            # 描述性文本 verbalized 为 "no answer"，则表示没有关于 w1 的具体关系描述
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."   # 表明在 w1 中没有任何关系
            else:
                hypo = f"Relation within {w1} is {verbalized}."  # 代码生成一个描述 w1 内部存在的具体关系类型的字符串
        # 如果 w2 不是 None，即存在第二个实体与 w1 进行比较或关联。
        else:
            # w1 always chem, w2 always gene
            if verbalized == "no answer":  # 如果描述性文本 verbalized 为 "no answer"，则表示两个实体之间没有明确的关系。
                hypo = f"There is no relation between {w1} and {w2}."
            else:
                hypo = f"{w1} is a {verbalized} to {w2}."# 否则关系是...

    # 如果hypo_version等于"v2"，则执行以下代码，这是上下文相关的假设生成。
    elif hypo_version == "v2": # contextual
        # w1: "@CHEM-GENE$"  检查w2是否为None。
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."  # 如果w2为None且verbalized为"no answer"，则生成一个假设句子，表示在w1中没有关系。
            else:
                hypo = f"Relation within {w1} is {verbalized}."  # 否则 有关系....表示w1中的关系是verbalized
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."  # 如果w2不为None且verbalized为"no answer"，则生成一个假设句子，表示w1和w2之间没有关系。
            # 如果verbalized等于"upregulator"
            elif verbalized == "upregulator":
                hypo = f"Upregulator {w1} is activated by {w2}."  # 如果w2不为None且verbalized为"upregulator"，则生成一个假设句子，表示w1是w2的上调剂。
            elif verbalized == "downregulator":
                hypo = f"Downregulator {w1} is designed as an inhibitor of {w2}."  # 生成一个假设句子，表示w1是w2的下调剂
            elif verbalized == "agonist":
                hypo = f"Activity of agonist {w1} is mediated by {w2}."   # 生成一个假设句子，表示w1是w2的激动剂
            elif verbalized == "antagonist":
                hypo = f"{w1} is identified as an antagonist of {w2}."   # 生成一个假设句子，表示w1是w2的拮抗剂。
            else: # substrate
                hypo = f"{w1} is a substrate for {w2}."   # 如果w2不为None且verbalized表示w1是w2的底物

    #  上下文相关的假设生成。
    elif hypo_version == "v3": # in-context
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "@CHEMICAL$ selectively induced @GENE$ in four studied HCC cell lines."'  # {w1}与{w2}的关系类似于“@CHEMICAL$选择性诱导的@GENE$在四种HCC细胞系中的关系”。
            elif verbalized == "downregulator":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "@CHEMICAL$, a new @GENE$ inhibitor for the management of obesity."'   # {w1}与{w2}的关系类似于“@CHEMICAL$，一种新的管理肥胖的@GENE$抑制剂”中描述的关系。
            elif verbalized == "agonist":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "Pharmacology of @CHEMICAL$, a selective @GENE$/MT2 receptor agonist: a novel therapeutic drug for sleep disorders."'   # {w1}与{w2}的关系类似于“@CHEMICAL$的药理学，一种选择性的@GENE$/MT2受体激动剂:一种治疗睡眠障碍的新型药物”。
            elif verbalized == "antagonist":
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "@CHEMICAL$ is an @GENE$ antagonist that is metabolized primarily by glucuronidation but also undergoes oxidative metabolism by CYP3A4."'  # 描述 w1 作为拮抗剂对 w2 的具体影响
            else: # substrate
                hypo = f'Relation of {w1} to {w2} is similar to relation described in "For determination of [@GENE$+Pli]-activity, @CHEMICAL$ was added after this incubation."'   # {w1}与{w2}的关系类似于“为了测定[@GENE$+Pli]-活性，在孵育后加入@CHEMICAL$”中描述的关系。
    elif hypo_version == "v4": # in-context + contextual
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f'''Upregulator {w1} is activated by {w2}, similar to relation described in "@CHEMICAL$ selectively induced @GENE$ in four studied HCC cell lines."'''   # 上调因子{w1}被{w2}激活，类似于“@CHEMICAL$选择性诱导的@GENE$在四种研究的HCC细胞系中的关系”。
            elif verbalized == "downregulator":
                hypo = f'Downregulator {w1} is designed as an inhibitor of {w2}, similar to relation described in "@CHEMICAL$, a new @GENE$ inhibitor for the management of obesity."'   # 下调因子{w1}被设计为{w2}的抑制剂，类似于“@CHEMICAL$”中描述的关系，一种新的管理肥胖的@GENE$抑制剂。
            elif verbalized == "agonist":
                hypo = f'Activity of agonist {w1} is mediated by {w2}, similar to relation described in "Pharmacology of @CHEMICAL$, a selective @GENE$/MT2 receptor agonist: a novel therapeutic drug for sleep disorders."'   # 激动剂{w1}的活性由{w2}介导，类似于“@CHEMICAL$药理学，一种选择性的@GENE$/MT2受体激动剂:一种新的睡眠障碍治疗药物”中描述的关系。
            elif verbalized == "antagonist":
                hypo = f'{w1} is identified as an antagonist of {w2}, similar to relation described in "@CHEMICAL$ is an @GENE$ antagonist that is metabolized primarily by glucuronidation but also undergoes oxidative metabolism by CYP3A4."'  # {w1}被鉴定为{w2}的拮抗剂，类似于“@CHEMICAL$是一种@GENE$拮抗剂，主要通过葡萄糖醛酸化代谢，但也通过CYP3A4进行氧化代谢”中描述的关系。
            else: # substrate
                hypo = f'{w1} is a substrate for {w2}, similar to relation described in "For determination of [@GENE$+Pli]-activity, @CHEMICAL$ was added after this incubation."'  # {w1}是{w2}的底物，类似于“为了测定[@GENE$+Pli]-活性，在孵育后加入@CHEMICAL$”中描述的关系。
    else: # v5 autoamtic
        if w2 is None:  # w1: "@CHEM-GENE$"
            if verbalized == "no answer":
                hypo = f"There is no relation in {w1}."
            else:
                hypo = f"Relation within {w1} is {verbalized}."
        else: # w1 always chem, w2 always gene
            if verbalized == "no answer":
                hypo = f"{w1} and {w2} have no relation."
            elif verbalized == "upregulator":
                hypo = f"{w1} is activated by {w2}."
            elif verbalized == "downregulator":
                hypo = f"{w1} activity inhibited by {w2}."
            elif verbalized == "agonist":
                hypo = f"{w1} agonist actions of {w2}."
            elif verbalized == "antagonist":
                hypo = f"{w1} identified are antagonists {w2}."
            else: # substrate
                hypo = f"{w1} is substrate for {w2}."
    return hypo[0].upper() + hypo[1:]


# 用于定义数据加载和预处理的逻辑
class DataModule(LightningDataModule):
    # 定义一个名为LABEL_VERBALIZER的字典，它将标签ID（如"CPR:3"）映射到相应的标签名称（如"upregulator"）。
    LABEL_VERBALIZER = {
        "CPR:3": "upregulator",
        "CPR:4": "downregulator",
        "CPR:5": "agonist",
        "CPR:6": "antagonist",
        "CPR:9": "substrate",
        "no answer": "no answer"
    }

    # 定义类的初始化方法__init__，它接受几个参数，包括data_dir（数据集的目录）、batch_size（批量大小）、num_workers（数据加载器的工作进程数）、pin_memory（是否将数据加载到内存中）和一些可选的kwargs。
    def __init__(self, data_dir,
                 # dataloader
                 batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
                 **kwargs):
        # 调用父类的初始化方法，这是PyTorch Lightning DataModule的约定
        super().__init__()
        assert os.path.exists(data_dir)
        # 创建一个空的字典self.datasets，用于存储不同分割的数据集。
        self.datasets = {}
        # 保存超参数，但不在日志中记录
        self.save_hyperparameters(logger=False)

    # 定义setup方法，它接受一个可选的stage参数，用于指定执行哪些数据预处理步骤
    def setup(self, stage: Optional[str] = None):
        for split in ["train", "dev", "test"]:   # 遍历数据集的三个分割：训练、验证和测试。
            # 为每个分割创建一个ChemProt_Dataset实例，并将其存储在self.datasets字典中。
            self.datasets[split] = ChemProt_Dataset(self.hparams.data_dir, split)

    # 定义load_dataloader方法，它接受一个split参数，用于指定要加载的数据集分割。  使用数据集、工作进程数、内存固定设置、批处理大小和是否打乱数据（仅训练数据）来构建 DataLoader。
    def load_dataloader(self, split: str):
        return DataLoader(
            self.datasets[split],   # 指定数据集  从split中获取，split是什么这里就是什么
            num_workers=self.hparams.num_workers,  # 设置用于加载数据的子进程数量 从cpr的yaml文件中获得
            pin_memory=self.hparams.pin_memory,    # 数据加载器会将数据拷贝到CUDA固定内存中
            collate_fn=ChemProt_Dataset.collate_fn,   # 指定如何将样本列表打包成一个小批次
            batch_size=self.hparams.batch_size,    # hparams查看yaml文件
            shuffle=(split == "train"),   # 是否对数据进行随机打乱。通常在训练数据集上进行打乱，以提高模型的泛化能力，但在验证和测试集上则不打乱数据顺序。
        )

    # 定义train_dataloader方法，它返回用于训练的数据加载器。
    def train_dataloader(self):
        return self.load_dataloader("train")

    # 定义val_dataloader方法，它返回用于验证的数据加载器  它使用self.datasets["dev"]作为数据集，并应用了self.hparams中的超参数。
    def val_dataloader(self):
        return self.load_dataloader("dev")

    # 使用self.datasets["test"]作为数据集，并应用了self.hparams中的超参数。
    def test_dataloader(self):
        return self.load_dataloader("test")

    # 定义predict_dataloader方法，它返回用于预测的数据加载器
    def predict_dataloader(self):
        # 返回一个包含两个DataLoader实例的列表，第一个是用于训练的数据加载器，第二个是用于测试的数据加载器。这可能意味着在预测过程中，模型将使用训练和测试数据。
        return [self.train_dataloader(), self.test_dataloader()]

    # 静态方法，它定义了如何随机从所有可能的标签中选择一个标签作为负样本
    @staticmethod
    # 定义random_neg_sample方法，它接受一个字典类型的batch和一个可选的整数类型的num_negs参数。batch可能包含多个样本，每个样本都有一个"label"字段，表示正样本的标签。num_negs表示每个正样本应该生成的负样本数量。
    def random_neg_sample(batch: dict, num_negs: int = 1) -> dict:
        """
        randomly sample 1 from all possible labels
        :return: new batch with field "neg_label" List[List[str]] where List[str] is num_negs-len neg samples
        """
        # 开始一个列表推导式，用于生成负样本。
        neg = [
            # 对于batch中的每个标签label，使用random.sample函数从所有可能的标签中随机选择num_negs个不同的标签作为负样本。   会产生不包含现在标签的负样本列表
            random.sample(list(filter(  # 将filter函数返回的迭代器转换成列表。
                lambda l: l != label,  # 意思是保留所有不等于label的元素。
                DataModule.LABEL_VERBALIZER  # DataModule.LABEL_VERBALIZER（假设这是一个包含标签的列表或集合）中移除等于label的元素。
            )), k=num_negs)
            # 遍历batch中的每个样本的"label"字段。
            for label in batch["label"]
        ]
        assert len(neg) == len(batch["label"])   # 断言生成的负样本数量与batch中的样本数量相等。
        # enforce no answer to appear in neg_sample
        # 遍历生成的负样本和相应的正样本标签。
        for n, label in zip(neg, batch["label"]):
            # 如果当前的正样本标签不是"no answer"，并且生成的负样本中不包含"no answer"，则执行以下代码。
            if label != "no answer" and "no answer" not in n:
                # 将生成的第一个负样本设置为"no answer"。
                n[0] = "no answer"
        # 将生成的负样本添加到batch字典中，键为"neg_label"。
        batch["neg_label"] = neg
        # 返回修改后的batch字典，其中包含了新的"neg_label"字段。
        return batch