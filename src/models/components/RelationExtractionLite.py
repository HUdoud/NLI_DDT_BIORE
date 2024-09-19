from typing import List, Tuple

import torch
from transformers import PreTrainedModel

from . import LITE
from ... import utils
import argparse
from data_utils import (
    RE_Processor
)




logger = utils.get_logger(__name__)
# mention, type, start, end
Entity = Tuple[str, str, int, int]  # Entity是一个元组类型，用于表示实体，包含实体的提及（mention）、类型（type）、起始位置（start）和结束位置（end）


class RELite(LITE):
    def __init__(self, model_path: str,   # model_path是模型路径，用于加载预训练的模型。
                abstain=False, use_loss_weight=False,   # abstain是一个布尔参数，默认设置为False，表示是否预测两个实体之间是否有关系。
                # LITE   use_loss_weight是一个布尔参数，默认设置为False，表示是否使用损失权重。
                **kwargs):  # 是一个关键字参数列表，用于传递给父类LITE的初始化方法。
        """
        Lite with ED task
        :param entity_marker: whether adding entity marker to sentence
        :param abstain: if true, predict whether or not two entities have none relation
        """
        super().__init__(model_path=model_path, additional_tokens=None, **kwargs)  # 调用父类的__init__方法，并将model_path和additional_tokens传递给父类。
        self.use_loss_weight = use_loss_weight  # 保存use_loss_weight和abstain参数的值。
        self.abstain = abstain
        self.model: PreTrainedModel  # 它存储了加载的预训练模型。

    def hypo_template(self, w1, w2, verbalized, ds):  # hypo_template方法用于生成假设模板，用于将实体和关系转换为模型的输入。 w1和w2是实体的提及。verbalized是关系的文本表示。ds是数据集名称。
        if ds == "chemprot":  # hypo_template方法根据数据集名称从不同的数据模块中导入对应的函数，并返回一个假设模板。
            from ...datamodules.chemprot import hypo_template
        elif ds == "GAD":
            from ...datamodules.GAD import hypo_template
        else: #
            from ...datamodules.DDI import hypo_template
        return hypo_template(w1, w2, verbalized, self.abstain, hypo_version=self.hypo_version)
# 用于处理实体关系（Entity Relationship, ER）任务
    def format_input(self, sentence, entity1, entity2, label, LABEL_VERBALIZER: dict, dataset_name):   # format_input方法是RELite类的一个成员方法，它接受以下参数：sentence：输入句子，可以是列表或单个字符串。entity1：实体1的信息，可以是列表或单个元组。entity2：实体2的信息，可以是列表或单个元组。label：标签信息，可以是列表或单个字符串。LABEL_VERBALIZER：一个字典，用于将标签转换为可读的文本。dataset_name：数据集名称，可以是列表或单个字符串。
        """
        all input can be a list or single str
        return two lists
        """
        if type(sentence) is list:  # 这行代码检查sentence是否是一个列表。如果是，它将检查所有输入的长度是否相等。如果不是，它将每个输入转换为一个包含单个元素的列表。
            assert len(sentence) == len(entity1) == len(entity2) == len(label) == len(dataset_name)
        else:  # change to list
            sentence, entity1, entity2, label, dataset_name = map(
                lambda a: [a], [sentence, entity1, entity2, label, dataset_name])
        formatted_sentence, formatted_hypothesis = [], []  # 初始化两个空列表：formatted_sentence用于存储格式化的句子，formatted_hypothesis用于存储格式化的假设。
        for s, (e1, *_), (e2, *_), l, ds in zip(sentence, entity1, entity2, label, dataset_name):  # 遍历sentence、entity1、entity2、label和dataset_name的列表，
            verbalized = LABEL_VERBALIZER[l]   # 转换标签
            fh = self.hypo_template(e1, e2, verbalized, ds)  # 并将每个元素传递给hypo_template方法
            formatted_sentence.append(s)
            formatted_hypothesis.append(fh)
        return formatted_sentence, formatted_hypothesis  # 它将原始句子和生成的假设分别添加到formatted_sentence和formatted_hypothesis列表中。最后，它返回这两个列表。
# 模型的前向传播过程。
    def forward(self, sentence: List[str], entity1: List[Entity], entity2: List[Entity],
                label: List[str], neg_label: List[List[str]], LABEL_VERBALIZER: dict, dataset_name: List[str], **kwargs):   # sentence：输入句子列表。entity1：实体1信息列表，每个实体是一个元组。entity2：实体2信息列表，每个实体是一个元组。label：标签列表。neg_label：负样本标签列表，每个标签是一个字符串列表。LABEL_VERBALIZER：一个字典，用于将标签转换为可读的文本。dataset_name：数据集名称列表。**kwargs：其他关键字参数。
        """
        sentence: list of sentence
        entity1, entity2: list of Entity, [mention, type, start, end]
        label: list of gt label
        neg_label: list of num_negs-len neg samples
        LABEL_VERBALIZER Dict[label -> verbalized label]
        """
        formatted_sentence, formatted_hypothesis = self.format_input(sentence, entity1, entity2, label,
                                                                     LABEL_VERBALIZER, dataset_name)  # format_input方法，将输入的句子、实体、标签和数据集名称转换为模型可以处理的格式。

# #        ——————————————————————————————START—————————————————————————————————— 修改
        self.formatted_sentence = formatted_sentence
# # ——————————————————————————————————————————结束————————————————————————————————————————
        # (N, )
        pos_entailment = self.get_entailment_score(formatted_sentence, formatted_hypothesis)  # 调用get_entailment_score方法，计算正例的蕴含概率，即正样本的得分。

        class_weights = None   # 是否启用了损失权重（self.use_loss_weight）。如果是，它从sklearn.utils.class_weight.compute_class_weight计算损失权重，并将它们转换为张量。
        if self.use_loss_weight: # from https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
            loss_weights = {
                "chemprot": {"no answer": 0.21588977, "CPR:3": 3.97597002, "CPR:4": 1.3497231, "CPR:5": 17.37475915, "CPR:6": 13.12590975, "CPR:9": 4.1345713},
                "DDI": {"no answer": 0.23266038, "DDI-advise": 6.9782069, "DDI-effect": 3.40457604, "DDI-int": 28.58305085, "DDI-mechanism": 4.35012898}
            }
            lw_dict = loss_weights[dataset_name[0]]  # 从loss_weights字典中获取与当前数据集名称（dataset_name[0]）对应的损失权重字典。
            class_weights = [   # 用列表推导式创建一个新的列表class_weights，其中包含与每个正样本标签相对应的损失权重。它通过遍历label列表，并从lw_dict字典中提取每个标签的损失权重。
                lw_dict[l]
                for l in label
            ]
            class_weights = torch.tensor(class_weights).to(pos_entailment.device)  # 将class_weights列表转换为张量，并将其移动到与pos_entailment相同的设备上。

#  _____计算负样本示例_____
        neg_entailments = []  # 初始化 neg_entailments 列表，用于存储每个负样本的蕴含得分。
        for i in range(self.num_negs):  # 循环每个负样本的索引
            # ith neg label
            neg_label_i = list(map(lambda l: l[i], neg_label))  # 使用 map 函数和 lambda 表达式从 neg_label 中提取第 i 个负样本标签列表。
            formatted_sentence, formatted_hypothesis = self.format_input(sentence, entity1, entity2, neg_label_i,
                                                                         LABEL_VERBALIZER, dataset_name)
            self.formatted_sentence = formatted_sentence
            neg_entailment = self.get_entailment_score(formatted_sentence, formatted_hypothesis)  # 计算负例子得分
            neg_entailments.append(neg_entailment)   # 将计算得到的负样本蕴含得分neg_entailment添加到列表neg_entailments中
        # (N, num_negs)
        neg_entailments = torch.stack(neg_entailments, dim=1)   # 使用 torch.stack 将列表中的所有负样本蕴含得分堆叠成一个张量，按第二维（负样本维）堆叠。

        none_location = []  # 初始化 none_location 列表，用于记录每个样本的无答案位置。
        for l, nl in zip(label, neg_label):  # 循环遍历正样本标签和负样本标签列表。
            if l == "no answer":   # 如果当前样本的标签是"no answer"，则将该样本的none_location值设置为0。
                none_location.append(0)
            else:  # 当前样本的标签不是"no answer"，则检查其负样本标签列表中是否包含"no answer"
                if "no answer" in nl:
                    none_location.append(1 + nl.index("no answer"))  # 包含，则将该样本的none_location值设置为1加上no answer在负样本标签列表中的索引
                else:
                    none_location.append(-1)   # 不包含，则将该样本的none_location值设置为-1。

        none_location = torch.tensor(none_location).to(pos_entailment.device)   # 将 none_location 列表转换为张量，并确保其位于与正样本得分相同的设备上。
        loss = self.compute_loss(pos_entailment, neg_entailments, none_location, class_weights=class_weights)  # 调用 compute_loss 方法，计算基于正样本得分、负样本得分、无答案位置和类别权重的损失。
        return loss  # 返回计算得到的损失。
    # 定义了一个名为predict的方法，它是RELite类的一个成员方法，用于预测给定句子中实体之间的关系
    @torch.no_grad()
    def predict(self, sentence: List[str], label: List[str], id: List[str], **kwargs) -> List[dict]:   # predict方法是RELite类的一个成员方法，它接受以下参数：sentence：输入句子列表。label：标签列表，尽管这可能不是必需的，因为预测方法通常不需要标签。id：标识符列表，每个句子都有一个唯一的ID。**kwargs：其他关键字参数。方法返回一个包含字典的列表，每个字典包含句子、ID、标签和嵌入向量。
        # BioLINKBERT directly on sentence
        model_inputs = self.tokenizer(sentence,   # self.tokenizer将输入句子转换为模型可以处理的格式。它设置填充（padding）、截断（truncation）和最大长度（max_length），并将结果移动到self.device上。
                                      padding=True, truncation=True,
                                      max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        outputs = self.model(**model_inputs, output_hidden_states=True)   # self.model进行前向传播，并设置输出隐藏状态（output_hidden_states=True），以便在后续步骤中使用。
        embs = outputs.hidden_states[-1]   # (B, L, d)  从模型输出中提取最后一个隐藏状态，它是一个三维张量，其中B是批量大小，L是句子长度，d是嵌入向量的维度。
        embs = embs[:, 0, :].detach().cpu().numpy()   # [CLS] -> (B, d)  提取的嵌入向量转换为只包含第一个词元（通常是[CLS]标记）的二维张量，并将它移动到CPU上，然后转换为NumPy数组。
        ret = []  # 创建一个空列表ret，用于存储预测结果
        for s, i, l, emb in zip(sentence, id, label, embs):   # 遍历输入的句子、ID、标签和嵌入向量，并为每个元素创建一个字典，包含句子、ID、标签和嵌入向量，然后将这个字典添加到ret列表中。
            ret.append({
                "sentence": s, "id": i, "label": l, "emb": emb
            })
        return ret  # 最后，它返回ret列表。

# 用于处理文本数据，评估句子中实体之间的关系
    @torch.no_grad()
    def generate(self, sentence: List[str], entity1: List[Entity], entity2: List[Entity], dataset_name: List[str],
                 label: List[str], id: List[str], LABEL_VERBALIZER: dict, **kwargs) -> List[dict]:   # 参数包括句子、实体(entity1和entity2)列表、数据集名称、标签、ID和一个标签转义字典。此方法还接受额外的关键字参数(**kwargs)。
        """
        label: non-verbalized eg CPR:6
        :return: list of dict (generated output)
        """
        if self.abstain:   # only has relation or no relation   检查是否启用了abstain模式。如果启用，则表示模型只能预测两个实体之间是否有关系（即存在关系或不存在关系）。
            LABEL_VERBALIZER = {   # 定义了一个字典LABEL_VERBALIZER，用于将标签转换为可读的文本。在这个模式下，只有两个标签：“no answer"和"has answer”。
                "no answer": "no answer",
                "has answer": "has answer"
            }
        all_labels = sorted(LABEL_VERBALIZER)   # 创建两个列表：all_labels，包含所有标签的排序列表；all_labels_verbalized，包含所有标签的可读文本列表。
        all_labels_verbalized = list(map(LABEL_VERBALIZER.get, all_labels))
        ret = []   # 创建一个空列表ret，用于存储最终的预测结果。
        # list of list
        formatted_sentence, formatted_hypothesis = [], []   # 创建两个空列表formatted_sentence和formatted_hypothesis，用于存储格式化的句子和假设。
        for s, e1, e2, ds in zip(sentence, entity1, entity2, dataset_name):   # 循环遍历句子、实体1、实体2和数据集名称，并为每个元素生成所有可能的标签。
            # all possible label
            for l in all_labels:   # 然后，它使用format_input方法为每个可能的标签生成格式化的句子和假设，并将它们添加到相应的列表中。
                # append list
                fs, fh = self.format_input(s, e1, e2, l, LABEL_VERBALIZER, ds)
                formatted_sentence += fs
                formatted_hypothesis += fh

        entailment = self.get_entailment_score(formatted_sentence, formatted_hypothesis)   # 调用get_entailment_score方法，计算所有格式化句子和假设的蕴含概率。然后，它将结果转换为列表。
        entailment = entailment.detach().cpu().numpy().tolist()
        assert len(entailment) == len(formatted_sentence) == len(formatted_hypothesis)   # 断言entailment、formatted_sentence和formatted_hypothesis的长度相等。
        start = 0  # 初始化一个变量start，用于记录预测开始的位置。
        for s, e1, e2, i, gt_l, name in zip(sentence, entity1, entity2, id, label, dataset_name):  # 循环遍历句子、实体1、实体2、标识符、标签和数据集名称。对于每个元素，它创建一个切片span，该切片包含与当前实体对相关的所有可能的标签。
            span = slice(start, start + len(LABEL_VERBALIZER))
            confidence = entailment[span]
            fs = formatted_sentence[span]   # 它从entailment列表中提取相应的蕴含概率，从formatted_sentence和formatted_hypothesis列表中提取相应的句子和假设。
            fh = formatted_hypothesis[span]

            assert len(confidence) == len(LABEL_VERBALIZER) == len(fh) == len(fs)   # 断言confidence、LABEL_VERBALIZER、fh和fs的长度相等。
            start += len(LABEL_VERBALIZER)
            ret.append({    # 当前实体对的预测结果添加到ret列表中。它包含句子、ID、实体1和2、前提、假设、置信度、所有标签、实际标签和标签的可读文本、数据集名称等信息。如果实际标签在LABEL_VERBALIZER中不存在，则将其替换为"has relation"。
                "sentence": s, "id": i,
                "entity1": e1, "entity2": e2,
                "premise": fs, "hypo": fh,
                "confidence": confidence,
                "all_labels": all_labels, "all_labels_verbalized": all_labels_verbalized,
                 # only 1 gt
                "label": gt_l, "label_verbalized": LABEL_VERBALIZER.get(gt_l, "has relation"), # can keyerror if abstain, thus return "has relation"
                # metainfo
                "dataset_name": name,
            })

        return ret   # 返回ret列表，其中包含每个实体对的预测结果。
