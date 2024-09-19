from typing import List, Optional
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Optional

from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


from ... import utils

from yilatree import yilai

Words = List[str]   # Words是一个包含字符串的列表。
logger = utils.get_logger(__name__)  # 从某个工具模块（utils）中获取一个日志记录器（logger），这个记录器通常用于记录程序运行时的信息。

# 定义一个名为LITE的神经网络模块，用于自然语言处理中的文本蕴含任务
class LITE(nn.Module):   # 代码定义了LITE类的初始化方法，接受多个参数，包括预训练模型的路径、损失函数的参数等。
    def __init__(self, model_path: str, margin: float = 0.1, use_softmax: bool = True, hypo_version="v1",
                 num_negs: int = 1, temperature: float = 1.0,
                 use_loss: str ="NCE+AC", NCE_strength: float = 1.0, AC_strength: float = 1.0,
                 use_ranking_sum_as_InfoNCE=False, additional_tokens: Optional[List[str]] = None):
        """
        :param model_path: huggingface path
        :param num_negs: if 1, use LITE vanilla MarginRankingLoss; else use InfoNCE
        :param margin: only used in MarginRankingLoss
        :param temperature: only used in InfoNCE
        :param additional_tokens: add new token to pretrained tokenizer
        """
        super().__init__()  # 代码调用父类（nn.Module）的初始化方法，确保LITE类能够继承nn.Module的所有功能和属性
        self.temperature = temperature  # 代码将初始化方法中的参数保存为LITE类的属性。
        self.NCE_strength = NCE_strength
        self.AC_strength = AC_strength
        self.margin = margin
        self.use_loss = use_loss
        self.hypo_version = hypo_version
        self.num_negs = num_negs
        self.use_softmax = use_softmax
        self.use_ranking_sum_as_InfoNCE = use_ranking_sum_as_InfoNCE
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # 从预训练模型路径加载一个分词器（tokenizer），用于将文本转换为模型可以理解的数字表示。
        if model_path in ["roberta-large-mnli"]:  # 检查model_path是否是"roberta-large-mnli"。如果是，那么它将self.entailment_index设置为2。
            self.entailment_index = 2  # last label is entailment
        else:
            self.entailment_index = 0  # first label is entailment  # 如果model_path不是"roberta-large-mnli"，那么它将self.entailment_index设置为0。
        logger.info(f"entailment index is {self.entailment_index}")  # 使用日志记录器logger来记录一条信息，这条信息表明了蕴含标签的索引是什么。
# 这段代码检查是否提供了额外的特殊标记，如果有，则将这些标记添加到分词器中。
        if additional_tokens:
            num_added_toks = self.tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})
            logger.info(f"tokenizer adding {num_added_toks} new tokens")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)  # 从预训练模型路径自动加载一个用于序列分类的模型。
        self.model.resize_token_embeddings(len(self.tokenizer))  # 代码调整模型的嵌入层大小，以适应新的分词器的大小。
# 定义了一个前向传播过程，用于计算给定前提（sentence）和假设（hypothesis）的蕴含概率。
    def get_entailment_score(self, sentence, hypothesis):
        """
        one forward pass
        return (bsz, )
        """   # 代码使用分词器（tokenizer）将前提和假设转换为模型可以理解的输入格式。它首先将文本进行填充（padding）和截断（truncation），以确保它们的长度适合模型的输入要求。然后，它将输入转换为PyTorch张量，并将其移动到定义的设备上（例如，CPU或GPU）。
        model_inputs = self.tokenizer(sentence, hypothesis,
                                      padding=True, truncation=True,
                                      max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
        # self.lines = nn.Linear(946 , 512).to(device=self.device)
        # input_ids = model_inputs['input_ids']
        # token_type_ids = model_inputs['token_type_ids']
        # attention_mask = model_inputs['attention_mask']
        # input_ids = self.lines(input_ids.type(torch.float))
        # token_type_ids = self.lines(token_type_ids.type(torch.float))
        # attention_mask = self.lines(attention_mask.type(torch.float))
        # data = {
        #     "input_ids": input_ids.type(torch.int64),
        #     "token_type_ids": token_type_ids.type(torch.int64),
        #     "attention_mask": attention_mask.type(torch.int64),
        # }
        # # 使用 BatchEncoding 封装这些数据
        # from transformers import BatchEncoding
        # model_inputs = BatchEncoding(data)
        outputs = self.model(**model_inputs)  # 将处理好的输入传递给模型，进行前向传播，得到模型的输出。
        dep_order_matrix = yilai(self.tokenizer,self.formatted_sentence)
        logits = outputs.logits # (bsz, 3)   # 从模型的输出中提取 logits。Logits 是模型在未应用激活函数之前的原始分数，通常用于多类分类问题。在这里，logits 的形状是 (bsz, 3)，其中 bsz 是批量大小，3 表示三个类别（蕴含、矛盾、中性）的分数。
        dep_order_matrix = dep_order_matrix.to(device='cuda:0')
        logits=(dep_order_matrix*0.5) + logits
        if self.use_softmax:  # 这行代码检查是否使用了softmax函数。如果self.use_softmax为True，则对logits应用softmax函数，将分数转换为概率分布。如果self.use_softmax为False，则直接使用logits作为概率。
            probs = logits.softmax(dim=-1)
        else:
            probs = logits
        entailment_probs = probs[:, self.entailment_index]  # 从概率分布中提取表示蕴含类别的概率。self.entailment_index是在初始化方法中设置的，表示蕴含类别的索引。
        return entailment_probs  # 返回蕴含概率
# 用于计算对比学习中的 InfoNCE 损失。InfoNCE 损失是一种在自监督学习中常用的损失函数，用于学习一个好的特征表示，使得正样本之间的相似度比它们与负样本的相似度高。
    def infoNCE(self, pos, neg, class_weights=None):  # 它接受三个参数：pos 是正样本的得分，neg 是负样本的得分矩阵，class_weights 是一个可选参数
        """
        pos (N, )
        neg (N, num_neg)
        max pos's loglikelihood over neg
        """
        if self.use_ranking_sum_as_InfoNCE:  # 检查一个类属性 use_ranking_sum_as_InfoNCE 是否为 True。如果是，使用排名和作为 InfoNCE 损失的计算方法。
            indicator = torch.ones_like(pos).long().to(pos.device)  # 创建一个与 pos 相同形状的全一张量，并转换为长整型（long），确保其位于同一设备上。
            return sum(
                F.margin_ranking_loss(pos, n, indicator, margin=self.margin)  # 使用 torch.unbind 按第二维（负样本维）解绑 neg 张量，对每个负样本计算 F.margin_ranking_loss，并求和返回。
                for n in neg.unbind(1)  # F.margin_ranking_loss 是一个用于计算排名损失的函数，用于使得正样本的得分比负样本的得分更高。
            )
        # (N, num_negs + 1)
        entailments = torch.column_stack([pos, neg])   # 如果不使用排名和作为损失计算方法，则将 pos 和 neg 按列堆叠成一个新的张量 entailments，其中正样本位于每一行的第一个位置。
        # first one is postive, need to max log prob
        indicator = torch.zeros(pos.size(0)).long().to(pos.device)  # 创建一个全零的长整型张量 indicator，其长度等于 pos 的第一维的大小，并确保其位于同一设备上
        entailments = entailments.div(self.temperature)  # 将 entailments 张量的每个元素除以一个温度参数 self.temperature，这是一种常用的在 softmax 计算中调整分布尖锐度的方法。
        # (N, )
        loss = F.cross_entropy(entailments, indicator, reduction="none")  # 计算交叉熵损失，其中 entailments 作为预测得分，indicator 作为目标标签。reduction="none" 表示不对损失进行聚合，返回每个样本的损失。
        if class_weights is None:  # 如果未提供 class_weights，则创建一个与 loss 形状相同且元素全为1的张量。
            class_weights = torch.ones_like(loss)
        loss = (loss * class_weights).mean()  # 将损失与类权重相乘并求均值，得到最终的加权损失。
        return loss  # 返回计算得到的 InfoNCE 损失。
    # 计算一个称为“答案一致性”(Answer Consistency, AC)的损失。其中模型需要判断一个前提和多个假设之间的关系。损失函数的目标是确保模型能够正确地识别哪些假设与前提相关（可回答的），哪些假设与前提不相关（不可回答的）。
    def AC_loss(self, pos_entailment, neg_entailments, none_location):   # 定义AC_loss方法，接受三个参数：pos_entailment：正例的蕴含概率。neg_entailments：负例的蕴含概率矩阵，每一行代表一个前提的所有负例。none_location：指示none类别在负例中的位置的整数数组。如果为-1，则表示负例中不存在none类别。
        """
        :param none_location: int (N, ) each in [0, num_negs], where in negs is none (i.e. no relation), either 0 (gt is none) or 1-num_negs (gt is not none)
            if -1 then none not in negs, but not happen if two_stage
        """

        """
        answerable (gt is not none)
            ranking loss of (gt, none)
        """
        loss = 0.0  # 初始化损失值为0，并确定哪些示例是可回答的。
        answerable = (none_location != 0)
        # answerable_none_location need to minus 1 due to 0 reserved for none
        answerable_pos, answerable_negs, answerable_none_location = pos_entailment[answerable], neg_entailments[answerable], none_location[answerable] - 1  # 从输入中提取可回答示例的正例概率、负例概率矩阵和none类别的位置。
        N_answerable = answerable_pos.size(0)  # 如果存在可回答的示例，计算正例和none类别的排名损失，并累加到总损失中。
        if N_answerable != 0:
            # answerable_pos (N_answerable, ); answerable_negs (N_answerable, num_negs); answerable_none_location (N_answerable, )
            none_negs = answerable_negs[torch.arange(N_answerable), answerable_none_location] # (N_answerable, )
            indicator = torch.ones_like(answerable_pos).long().to(pos_entailment.device)
            loss += F.margin_ranking_loss(answerable_pos, none_negs, indicator, margin=self.margin)
        """
        unanswerable (gt is none)
            sum of ranking loss (none, other)
        """
        unanswerable_pos, unanswerable_negs = pos_entailment[~answerable], neg_entailments[~answerable]  # 从输入中提取不可回答示例的正例概率和负例概率矩阵。
        N_unanswerable = unanswerable_pos.size(0)  # 如果存在不可回答的示例，对所有负例计算排名损失，并将它们累加到总损失中。
        if N_unanswerable != 0:
            indicator = torch.ones_like(unanswerable_pos).long().to(pos_entailment.device)
            loss += sum(
                F.margin_ranking_loss(unanswerable_pos, unanswerable_neg, indicator, margin=self.margin)
                for unanswerable_neg in unanswerable_negs.unbind(1)
            )
        assert N_answerable + N_unanswerable > 0   # 断言确保至少有一个示例被处理。最后，返回计算出的损失值。
        return loss
# 用于计算文本蕴含任务的损失。它根据不同的配置选择不同的损失函数组合。
    def compute_loss(self, pos_entailment, neg_entailments, none_location, class_weights=None):   # 定义compute_loss方法，接受四个参数：pos_entailment：正例的蕴含概率。neg_entailments：负例的蕴含概率矩阵。none_location：指示none类别在负例中的位置的整数数组。class_weights：可选的类权重，用于在计算损失时给不同的示例赋予不同的权重。
        """
        :param pos_entailment: positive entailment scores, (N, )
        :param neg_entailments: entailment scores for neg samples, (N, num_negs)
        :param class_weights: (N, ) higher if such instance need higher attention, must be all positive
        :param none_location: int (N, ) each in [0, num_negs], where in negs is none (i.e. no relation), either 0 (gt is none) or 1-num_negs (gt is not none)
            if -1 then none not in negs, but not happen if two_stage
        """
        # pos_entailment, neg_entailments, none_location = map(lambda t: t.detach().cpu(), [pos_entailment, neg_entailments, none_location])
        N, num_negs = neg_entailments.shape  # 从负例的蕴含概率矩阵中获取批量大小的数量N和负例的数量num_negs。
        if self.use_loss == "NCE+AC":  # 检查是否使用“NCE+AC”损失组合。
            # InfoNCE  如果使用“NCE+AC”，则计算InfoNCE损失和AC损失，并将它们按相应的强度加权后相加。
            loss = self.NCE_strength * self.infoNCE(pos_entailment, neg_entailments, class_weights=class_weights)
            # AC
            loss += self.AC_strength * self.AC_loss(pos_entailment, neg_entailments, none_location)
        elif self.use_loss == "NCE+AC Two Stage":  # 检查是否使用“NCE+AC Two Stage”损失组合。
            loss = self.AC_loss(pos_entailment, neg_entailments, none_location)  # 如果使用“NCE+AC Two Stage”，则首先计算AC损失，然后进行两阶段的InfoNCE损失计算。
            answerable = (none_location != 0)  # 创建一个布尔索引，标记出 none_location 中不为0的位置，即有答案的样本
            # answerable_none_location need to minus 1 due to 0 reserved for none
            answerable_pos, answerable_negs, answerable_none_location = pos_entailment[answerable], neg_entailments[answerable], none_location[answerable] - 1   # 从正样本和负样本得分中，只选择有答案的样本。
            N_answerable = answerable_pos.size(0)
            if N_answerable != 0:
                # (N_answerable, num_negs) mask, True if belong to not-none; False if belong to none in answerable_none_location
                not_none_mask = torch.ones_like(answerable_negs).bool().scatter(1, answerable_none_location.view(-1, 1), 0)  # 创建一个掩码，将 none_location 指示的位置设为 False，其他为 True。
                # (N_answerable, num_negs - 1) excluding none
                not_none_negs = answerable_negs[not_none_mask].reshape(-1, num_negs - 1)  # 从 answerable_negs 中使用掩码过滤出有答案的负样本。
                loss += self.infoNCE(answerable_pos, not_none_negs, class_weights[answerable])
        elif self.use_loss == "NCE":   # 如果 use_loss 设为 "NCE"，则只使用 NCE 损失计算。
            assert num_negs > 1
            loss = self.infoNCE(pos_entailment, neg_entailments, class_weights)
        else: # "Ranking_loss"  如果其他模式都不匹配，默认使用排名损失。
            assert num_negs == 1   # 使用排名损失时，断言负例数量为1，然后计算边际排名损失。
            # (N, )
            neg_entailment = neg_entailments.squeeze(1)
            indicator = torch.ones(N).long().to(pos_entailment.device)
            loss = F.margin_ranking_loss(pos_entailment, neg_entailment, indicator, margin=self.margin)
        return loss
# 用于获取类实例中参数所在的设备（CPU或GPU)
    @property
    def device(self):
        return next(self.parameters()).device  # next函数来获取通过self.parameters()返回的迭代器中的第一个元素