import os
import re
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from dep_parser import DepInstanceParser

# 作用是将输入单词中的特定字符串替换为其他字符串。处理文本中可能出现的左右圆括号的占位符（-LRB- 和 -RRB-），将它们转换回常规的圆括号字符
def change_word(word):
    # 单词左或者右是-RRB-、-LRB-将会被替换为左圆括号 (或者)，然后返回修改后的单词
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word

def ensure_spacing(text):
    # # 匹配@CHEMICAL$、@GENE$、@CHEM-GENE$，并确保它们前后有空格
    # patterns = [r'@CHEMICAL\$', r'@GENE\$', r'@CHEM-GENE\$']
    #
    # for pattern in patterns:
    #     # 检查并添加前面的空格，除非它已经是行首
    #     text = re.sub(r'(?<!^)(?<!\s)(' + pattern + ')', r' \1', text)
    #     # 检查并添加后面的空格
    #     text = re.sub(r'(' + pattern + ')(?!\s)', r'\1 ', text)

        # 匹配 @CHEMICAL$、@GENE$、@CHEM-GENE$，并确保它们前后有空格
    patterns = [r'@CHEMICAL\$', r'@GENE\$', r'@CHEM-GENE\$']

    for pattern in patterns:
        # 确保标记前后的空格
        text = re.sub(r'(?<!^)(?<!\s)(' + pattern + ')', r'\1', text)
        text = re.sub(r'(' + pattern + ')(?!\s)', r'\1', text)

    # 移除所有标点符号和空格
    text = re.sub(r'[^\w@\$]', '', text)


    return text

def add_space_around_tag(sentence, tag):
    # 查找标签在句子中的位置
    index = sentence.find(tag)

    # 检查并添加前面的空格
    if index != 0 and sentence[index - 1] != ' ':
        sentence = sentence[:index] + ' ' + sentence[index:]
        index += 1  # 调整标签的位置以反映添加的空格

    # 检查并添加后面的空格
    tag_end = index + len(tag)
    if tag_end < len(sentence) and sentence[tag_end] != ' ':
        sentence = sentence[:tag_end] + ' ' + sentence[tag_end:]

    return sentence

def split_punctuation(word):
    # 首先识别和替换带有特殊符号的模式
    text = re.sub(r'@(CHEMICAL|GENE|CHEM-GENE)\$', r'\1', word)

    # 定义一个正则表达式，用于分割单词和标点，同时保留CHEM-GENE、CHEMICAL、GENE作为一个整体
    split_pattern = r'(CHEM-GENE|CHEMICAL|GENE)|(\W)'

    # 使用正则表达式分割文本
    split_words = re.split(split_pattern, text)
    # 去除空字符串
    split_words = [word for word in split_words if word and not word.isspace()]

    return split_words

class REDataset(Dataset):
    def __init__(self, features, max_seq_length):
        self.data = features
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        input_ids = torch.tensor(self.data[index]["input_ids"], dtype=torch.long)
        input_mask = torch.tensor(self.data[index]["input_mask"], dtype=torch.long)
        valid_ids = torch.tensor(self.data[index]["valid_ids"], dtype=torch.long)
        segment_ids = torch.tensor(self.data[index]["segment_ids"], dtype=torch.long)
        e1_mask_ids = torch.tensor(self.data[index]["e1_mask"], dtype=torch.long)
        e2_mask_ids = torch.tensor(self.data[index]["e2_mask"], dtype=torch.long)
        label_ids = torch.tensor(self.data[index]["label_id"], dtype=torch.long)
        entity_start_mark_position = torch.tensor(self.data[index]["entity_start_mark_position"], dtype=torch.long)
        b_use_valid_filter = torch.tensor(self.data[index]["b_use_valid_filter"], dtype=torch.long)

        def get_dep_matrix(ori_dep_type_matrix):
            dep_type_matrix = np.zeros((self.max_seq_length, self.max_seq_length), dtype=int)
            max_words_num = len(ori_dep_type_matrix)
            for i in range(max_words_num):
                dep_type_matrix[i][:max_words_num] = ori_dep_type_matrix[i]
            return torch.tensor(dep_type_matrix, dtype=torch.long)
        def get_dep_key_list(dep_key_list):
            t_dep_key_list = np.zeros((self.max_seq_length), dtype=int)
            for i in range(len(dep_key_list)):
                t_dep_key_list[i] = dep_key_list[i]
            return torch.tensor(t_dep_key_list, dtype=torch.long)

        dep_order_dep_type_matrix = get_dep_matrix(self.data[index]["dep_order_dep_type_matrix"])
        dep_path_dep_type_matrix = get_dep_matrix(self.data[index]["dep_path_dep_type_matrix"])
        dep_key_list = get_dep_key_list(self.data[index]["dep_key_list"])

        return input_ids,input_mask,valid_ids,segment_ids,label_ids,e1_mask_ids,e2_mask_ids, \
               dep_key_list, dep_order_dep_type_matrix,dep_path_dep_type_matrix

    def __len__(self):
        return len(self.data)

class RE_Processor():
    def __init__(self, direct=True,dep_order="first_order", keys_dict={}, vals_dict={}, labels_dict={}):
        self.direct = direct   # 可以直接调用的方式
        self.dep_order = dep_order
        self.keys_dict = keys_dict
        self.vals_dict = vals_dict
        self.labels_dict = labels_dict

    def get_train_examples(self, data_dir,formatted_sentence):  # 目的是获取训练数据的示例(训练传入数据)
        return self._create_examples(self.get_knowledge_feature(data_dir,formatted_sentence, flag="train"), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="dev"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self.get_knowledge_feature(data_dir, flag="test"), "test")

    def get_knowledge_feature(self, data_dir,formatted_sentence, flag="train"):    # 目的是获取指定数据目录中的知识特征，根据给定的标志来指定是获取训练、测试还是验证数据的特征
        return self.read_features(data_dir,formatted_sentence, flag=flag)     # data_dir是存储数据文件的目录路径，而flag是一个标志，用于指示需要获取哪种类型的数据特征

    # 目的是从一个指定的目录中读取包含标签数据的 JSON 文件，并返回这些标签。
    def get_labels(self, data_dir):
        label_path = os.path.join(data_dir, "label.json")
        # 使用 os.path.join 函数和给定的目录 data_dir 构造出标签文件 label.json 的完整路径。
        label_path = os.path.join(data_dir, "label.json")
        # 使用 with 语句以只读模式（'r'）打开标签文件。
        with open(label_path, 'r') as f:
            # 使用 json.load 函数从打开的文件对象 f 中加载 JSON 数据。
            labels = json.load(f)
        # 方法返回加载的标签数据
        return labels  # 创建了标签字典

    # 从一个指定的目录中读取依赖类型（dependency types）信息，并根据这些信息及其他条件构建一个依赖标签（dependency labels）的列表。
    # 它接受一个参数 data_dir，即包含依赖类型信息文件的目录。
    def get_dep_labels(self, data_dir):
        # 初始化一个列表 dep_labels，并预先添加了一个元素 "self_loop"。
        dep_labels = ["self_loop"]
        # 使用 os.path.join 函数和 data_dir 参数构造出依赖类型信息文件 dep_type.json 的完整路径。
        dep_type_path = os.path.join(data_dir, "dep_type.json")
        # 用 with 语句安全地打开文件，并将文件对象赋给变量 f。使用 'r' 模式表示以只读方式打开文件。
        with open(dep_type_path, 'r') as f:
            # 函数从打开的文件中加载 JSON 数据，并将解析后的数据（通常是一个列表或字典）赋给变量 dep_types。
            dep_types = json.load(f)
            # 遍历 dep_types 中的每个元素
            for label in dep_types:   # 将每一个依赖关系都生成_in和_out
                # 检查实例变量 self.direct 的值。如果为 True，则执行下面的代码块，意味着需要处理有向依赖。
                if self.direct:
                    dep_labels.append("{}_in".format(label))   # 对于每个依赖类型标签，添加两个变体：以 _in 结尾，表示入向依赖
                    dep_labels.append("{}_out".format(label))   # _out 结尾，表示出向依赖
                else:
                    # 则直接添加原始标签。直接将依赖类型的标签添加到 dep_labels 列表中，不区分入向或出向。
                    dep_labels.append(label)
        # 返回构建好的依赖标签列表
        return dep_labels  # 返回包括_in和_out的依赖关系

    def get_key_list(self):
        return self.keys_dict.keys()

    def _create_examples(self, features, set_type):   # 目的是将特征列表转换成一个包含多个示例的列表，每个示例都附带一个唯一标识符（guid）
        examples = []  # 初始化一个空列表examples，用于存储转换后的数据示例。
        for i, feature in enumerate(features):  # 使用enumerate函数遍历features列表。
            guid = "%s-%s" % (set_type, i)   # 为每个数据示例生成一个唯一标识符guid
            feature["guid"] = guid  # 将生成的guid添加到当前的feature字典中。
            examples.append(feature)  # 将包含guid的feature添加到examples列表中。
        return examples  # 最终返回包含所有数据示例的列表examples。

# 目的是从一系列文本文件中统计词频，然后基于这些词频创建一个词到索引的映射字典（keys_dict）
    def prepare_keys_dict(self, data_dir):
        # 创建了一个 defaultdict 对象，用于统计每个词的出现次数
        keys_frequency_dict = defaultdict(int)
        # 循环遍历三个数据集：训练集、测试集、开发集。
        for flag in ["train", "test", "dev"]:
            # 构造每个数据集文件的路径
            datafile = os.path.join(data_dir, '{}.txt'.format(flag))
            # 检查文件是否存在，如果不存在则跳过当前循环迭代。
            if os.path.exists(datafile) is False:
                continue
            # 调用 load_textfile 方法，加载文本文件并返回其中的数据。
            all_data = self.load_textfile(datafile)
            for data in all_data:  # 根据words中的词统计词频
                for word in data['words']:
                    keys_frequency_dict[change_word(word)] += 1  # 统计每个词出现的词频
        # 初始化 keys_dict，包含一个未知词汇 "[UNK]" 的条目，其索引为 0。
        keys_dict = {"[UNK]": 0}
        # 循环中，按照词频降序排序 keys_frequency_dict 中的词，然后遍历排序后的词频对。
        for key, freq in sorted(keys_frequency_dict.items(), key=lambda x: x[1], reverse=True):
            # 每个词添加到 keys_dict 中，其值为当前 keys_dict 的长度，从而保证每个词都有一个唯一的索引。
            keys_dict[key] = len(keys_dict)  # 按照词频从高到低将词保存在keys_dict中，并附上索引号。{'[UNK]': 0, 'the': 1, 'of': 2......}
        # 将创建好的 keys_dict 保存到实例变量 keys_dict 中
        self.keys_dict = keys_dict
        # 将词频字典 keys_frequency_dict 保存到实例变量 keys_frequency_dict 中
        self.keys_frequency_dict = keys_frequency_dict  # keys_frequency_dict内容是每个词对应出现的次数  'this'=638,'study'=610......
        # 打印 keys_dict="[UNK]":0
        # print(keys_dict)    # 词频表

    # 目的是根据依赖类型的列表来准备一个值字典（vals_dict），这个字典映射了每个依赖类型到一个唯一的索引。
    # 定义了一个名为 prepare_vals_dict 的方法，它接受一个参数 data_dir。
    def prepare_vals_dict(self, data_dir):
        # 获取依赖类型标签列表  (构建好的依赖类型标签一个是in 一个是Cross)
        dep_type_list = self.get_dep_labels(data_dir)
        # 初始化一个字典 vals_dict，它开始时包含一个键
        vals_dict = {"none": 0}
        # 遍历 dep_type_list 中的每个依赖类型标签。
        for dep_type in dep_type_list:
            # 对于列表中的每个 dep_type，在 vals_dict 中为它创建一个新的条目。{'none': 0, 'self_loop': 1, 'mwe_in': 2, 'mwe_out': 3......}
            vals_dict[dep_type] = len(vals_dict)
        # 将准备好的 vals_dict 字典赋值给实例变量vals_dict
        self.vals_dict = vals_dict
        # 打印出 vals_dict 字典的内容
        # print(vals_dict)  # 依赖关系对应的索引号

    # 目的是从给定的数据目录中提取标签列表，并基于这些标签创建一个标签到索引的映射字典
    def prepare_labels_dict(self, data_dir):
        label_list = self.get_labels(data_dir)          # 得到标签列表
        # 初始化一个空字典 labels_dict，用于存储标签到索引的映射
        labels_dict = {}
        # 遍历 label_list 中的每个标签
        for label in label_list:
            # 对于每个标签，将其添加到 labels_dict 字典中，并将其值按照0123...对应
            labels_dict[label] = len(labels_dict)
        # 将填充好的映射字典 labels_dict 赋值给实例变量 labels_dict
        self.labels_dict = labels_dict   # 创建了标签字典
        # 打印 labels_dict 字典
        # print(labels_dict)

#=-----------拼装所有数据-------
    def read_features(self, data_dir,formatted_sentence, flag):
        all_text_data = self.load_textfile(os.path.join(data_dir,  '{}.txt'.format(flag)))  # 获得所有数据，包括5个部分
        all_dep_info = self.load_depfile(os.path.join(data_dir,  '{}.txt.dep'.format(flag)))
        all_feature_data = [{} for _ in range(len(formatted_sentence))]
        # 需要改成循环
        for i in range(len(formatted_sentence)):
            formatted_sentence[i] = ensure_spacing(formatted_sentence[i])
        for text_data, dep_info in zip(all_text_data, all_dep_info):  # 对于每一对文本数据和依存信息，代码执行以下步骤来处理实体位置和调整标签名称
            log = True
            # print(len(formatted_sentence))
            for i in range(len(formatted_sentence)):
                #if text_data["ori_sentence"] == formatted_sentence[i]:
                if ensure_spacing(text_data["ori_sentence"]) == formatted_sentence[i]:
                    index = i
                    log = False
                # else:
                #     print(text_data["ori_sentence"])
                #     print(formatted_sentence[i])
            if log:
                continue
            label = text_data["label"]
            if label == "other":
                label = "Other"
            ori_sentence = text_data["ori_sentence"]  # 通过text_data["ori_sentence"].split(" ")将原始句子（ori_sentence）分割成单词列表。
            pattern = re.compile(r'\@CHEMICAL\$|\@CHEM-GENE\$|\@GENE\$|\w+|[^\w\s]')
            ori_sentence=pattern.findall(ori_sentence)
            tokens = text_data["words"]
            # 实体位置初始化
            e1_start = None
            e1_end = None
            e2_start = None
            e2_end = None

            # 查找实体位置并记录
            for i, token in enumerate(ori_sentence):
                if "@CHEMICAL$" in token:
                    e1_start = i
                    e1_end = i + 1
                elif "@GENE$" in token:
                    e2_start = i
                    e2_end = i + 1
                elif "@CHEM-GENE$" in token:
                    e1_start = i
                    e1_end = i + 1


            if e1_start is not None and e2_start is not None:
                if e1_start < e2_start:
                    start_range = list(range(e1_start, e1_end))
                    end_range = list(range(e2_start, e2_end))
                else:
                    start_range = list(range(e2_start, e2_end))
                    end_range = list(range(e1_start, e1_end))
                relation_label = "relation"
            elif e1_start is not None and e2_start is None:
                relation_label = "no_relation"
                start_range = list(range(e1_start, e1_end))
                end_range = []
            elif e1_start is None and e2_start is not None:
                relation_label = "no_relation"
                start_range = list(range(e2_start, e2_end))
                end_range = []

            dep_instance_parser = DepInstanceParser(basicDependencies=dep_info,tokens=tokens)  # 实例化依存关系解析器:首先，创建一个DepInstanceParser实例，传入基本依存关系信息（dep_info）和令牌列表（tokens）。
            if self.dep_order == "first_order":  # 根据依存关系的层次生成矩阵:
                dep_order_dep_adj_matrix, dep_order_dep_type_matrix = dep_instance_parser.get_first_order(direct=self.direct)
            elif self.dep_order == "second_order":
                dep_order_dep_adj_matrix, dep_order_dep_type_matrix = dep_instance_parser.get_second_order(direct=self.direct)
            elif self.dep_order == "third_order":
                dep_order_dep_adj_matrix, dep_order_dep_type_matrix = dep_instance_parser.get_third_order(direct=self.direct)

            if relation_label == "relation":
                dep_path_dep_adj_matrix, dep_path_dep_type_matrix = dep_instance_parser.get_dep_path(start_range,end_range,direct=self.direct)
            else:
                dep_path_dep_adj_matrix = np.zeros((128,128))
                dep_path_dep_type_matrix = np.ones((128,128))
            all_feature_data[index] = {
                "words": dep_instance_parser.words,
                "ori_sentence": ori_sentence,
                "dep_order_dep_adj_matrix": dep_order_dep_adj_matrix,
                "dep_order_dep_type_matrix": dep_order_dep_type_matrix,
                "dep_path_dep_adj_matrix": dep_path_dep_adj_matrix,
                "dep_path_dep_type_matrix": dep_path_dep_type_matrix,
                "label": label}
            # all_feature_data.append({
            #     "words": dep_instance_parser.words,
            #     "ori_sentence": ori_sentence,
            #     "dep_order_dep_adj_matrix": dep_order_dep_adj_matrix,
            #     "dep_order_dep_type_matrix": dep_order_dep_type_matrix,
            #     "dep_path_dep_adj_matrix": dep_path_dep_adj_matrix,
            #     "dep_path_dep_type_matrix": dep_path_dep_type_matrix,
            #     "label": label
            # })

        return all_feature_data

    def load_depfile(self, filename):
        data = []
        with open(filename, 'r') as f:
            dep_info = []
            for line in f:
                line = line.strip()  # 去除每行的首尾空白字符
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({   # 将这个字典添加到列表dep_info中
                        "governor": int(items[0]),    # 创建一个字典，包含当前依存关系的信息，其中"governor"是支配词的索引
                        "dependent": int(items[1]),    # "dependent"是依存词的索引
                        "dep": items[2],          # "dep"是依存关系的类型
                    })
                else:
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data

    # 用于从给定的文本文件中加载数据，并以特定的格式组织这些数据
    # 定义了 load_textfile 方法，它接受一个参数 filename，即要加载数据的文本文件的名称。
    def load_textfile(self, filename):
        # 创建了一个空的列表，用于存储从文件中加载的数据。
        data = []
        # 使用 with 语句以只读模式（'r'）打开文件 filename，并将文件对象赋给变量 f。
        with open(filename, 'r') as f:
            # 遍历文件中的每一行
            for line in f:
                # 使用 json.loads() 解析每一行的 JSON 数据
                record = json.loads(line)
                sentence = record["sentence"]
                # -----————————————————————————————添加的检查@GENE$(cpr数据集)------————————————————————————————————
                #分别处理"@GENE$"和"@CHEMICAL$"
                if "@GENE$" in sentence and "@CHEMICAL$" in sentence:
                    sentence = add_space_around_tag(sentence, "@GENE$")
                    sentence = add_space_around_tag(sentence, "@CHEMICAL$")
                elif "@CHEM-GENE$" in sentence:
                    sentence = add_space_around_tag(sentence, "@CHEM-GENE$")

                label = record["label"]
                e1 = None
                e2 = None
                # 查找并提取特定的实体标记
                if "@GENE$" in sentence and "@CHEMICAL$" in sentence:
                    e1 = "@CHEMICAL$"
                    e2 = "@GENE$"
                elif "@CHEM-GENE$" in sentence:
                    e1 = "@CHEM-GENE$"
                    e2 = "None"
                # 从句子中移除实体标记，并分割单词
                clean_sentence = sentence.replace("@GENE$", "GENE").replace("@CHEMICAL$", "CHEMICAL").replace(
                    "@CHEM-GENE$", "CHEM-GENE")
                # words = clean_sentence.split()
                words = split_punctuation(clean_sentence)
                # 检查最后一个元素是否包含句号，并将其分割
                # if words[-1].endswith('.'):
                #     last_word = words.pop()  # 移除最后一个元素
                #     words.append(last_word[:-1])  # 添加最后一个单词（不包含句号）
                #     words.append('.')  # 添加句号作为单独的元素
                # # 构建字典并添加到数据列表中
                data.append(
                    {
                        "id": record["id"],
                        "e1": e1,
                        "e2": e2,
                        "label": label,
                        "ori_sentence": sentence,
                        "words": words,
                        "sentence": sentence
                    }
                )
            # ——————————————————————结束————————————————————————

        # 完成文件的读取和数据的整理后，返回含有所有处理好的数据字典的列表 data。
        return data  # data数据格式是包含了e1、e2、label、ori_sentence是包括了<e1>和<e2>标识符的句子、words是不包括标识符的单词'this', 'study'...这样的list

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length):    # 用于将原始数据示例转换为模型训练所需的特征
        """Loads a data file into a list of `InputBatch`s."""

        label_map = self.labels_dict   # 初始化标签映射
        dep_label_map = self.vals_dict

        features = []   # 初始化特征列表和标记
        # b_use_valid_filter = False
        for (ex_index, example) in enumerate(examples):
            # dep_key_list = []   # 句子单词
            # tokens = ["[CLS]"]
            # valid = [0]
            # e1_mask = []   # 构造掩码和有效位序列
            # e2_mask = []
            # e1_mask_val = 0
            # e2_mask_val = 0
            # entity_start_mark_position = [0, 0]
            # for i, word in enumerate(example["ori_sentence"]):  # 处理和分词
            #     if len(tokens) >= max_seq_length - 1:
            #         break
            #     if word in ["<e1>", "</e1>", "<e2>", "</e2>"]:
            #         tokens.append(word)
            #         valid.append(0)
            #         if word in ["<e1>"]:
            #             e1_mask_val = 1
            #             entity_start_mark_position[0] = len(tokens) - 1
            #         elif word in ["</e1>"]:
            #             e1_mask_val = 0
            #         if word in ["<e2>"]:
            #             e2_mask_val = 1
            #             entity_start_mark_position[1] = len(tokens) - 1
            #         elif word in ["</e2>"]:
            #             e2_mask_val = 0
            #         continue
            #
            #     token = tokenizer.tokenize(word)
            #     if len(tokens) + len(token) > max_seq_length - 1:
            #         break
            #     dep_key_list.append(word)
            #     tokens.extend(token)
            #     e1_mask.append(e1_mask_val)
            #     e2_mask.append(e2_mask_val)
            # _____________________________________________修改____________________________________________
            tokens = ["[CLS]"]
            valid = [0]
            e1_mask = []  # 构造掩码和有效位序列
            e2_mask = []
            # e1_mask_val = 0
            # e2_mask_val = 0
            # entity_start_mark_position = [0, 0]
            # has_e1 = False
            # has_e2 = False

            for i, word in enumerate(example["ori_sentence"]):  # 处理和分词
                if len(tokens) >= max_seq_length - 1:
                    break
                if word in ['@CHEMICAL$', '@GENE$', '@CHEM-GENE$']:
                    tokens.append(word)
                    valid.append(1)
                    continue

                #  处理非实体情况
                token = tokenizer.tokenize(word)
                if len(tokens) + len(token) > max_seq_length - 1:
                    break
                # dep_key_list.append(word)
                tokens.extend(token)
                for i in range(len(token)):
                    if i == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
                # e1_mask.append(e1_mask_val)
                # if has_e2:  # 只有当存在第二种实体时，才添加e2_mask
                #     e2_mask.append(e2_mask_val)
            tokens.append("[SEP]")
            valid.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)   # 转换令牌为模型输入ID
            input_mask = [1] * len(input_ids)   # 变成全1的矩阵

            # Zero-pad up to the sequence length.  在准备模型输入数据时如何对序列进行零填充（zero-padding），以确保所有序列都有统一的长度max_seq_length。
            padding = [0] * (max_seq_length - len(input_ids))    # 计算填充长度
            input_ids += padding  # 执行填充操作
            input_mask += padding
            valid += padding
            e1_mask += [0] * (max_seq_length - len(e1_mask))
            e2_mask += [0] * (max_seq_length - len(e2_mask))

            assert len(input_ids) == max_seq_length  # 断言检查  使用assert语句确保所有序列的最终长度都严格等于max_seq_length。
            assert len(input_mask) == max_seq_length
            assert len(valid) == max_seq_length

            max_words_num = sum(valid)
            def get_adj_with_value_matrix(dep_adj_matrix, dep_type_matrix):  # 根据输入的依赖关系矩阵和类型矩阵，创建两个新的矩阵：final_dep_adj_matrix和final_dep_type_matrix。这两个新矩阵包含了过滤和处理后的依赖关系信息。
                final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=int)    # 创建一个大小为max_words_num x max_words_num的零矩阵，用于存储过滤后的依赖关系邻接信息
                final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=int)     # 创建一个同样大小的零矩阵，用于存储过滤后的依赖关系类型信息。
                for pi in range(max_words_num):  # 开始一个双层循环，遍历矩阵的行和列。
                    for pj in range(max_words_num):   # 如果在原始依赖关系邻接矩阵中的某个位置是0，表示没有依赖关系，循环继续到下一个位置。
                        if dep_adj_matrix[pi][pj] == 0:
                            continue
                        if pi >= max_seq_length or pj >= max_seq_length:
                            continue
                        final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                        final_dep_type_matrix[pi][pj] = dep_label_map[dep_type_matrix[pi][pj]]
                return final_dep_adj_matrix, final_dep_type_matrix

            dep_order_dep_adj_matrix, dep_order_dep_type_matrix = get_adj_with_value_matrix(example["dep_order_dep_adj_matrix"], example["dep_order_dep_type_matrix"])
            dep_path_dep_adj_matrix, dep_path_dep_type_matrix = get_adj_with_value_matrix(example["dep_path_dep_adj_matrix"], example["dep_path_dep_type_matrix"])

            # dep_key_list = [self.keys_dict[key] for key in dep_key_list]   # 根据单词的词频得到的dep_key_list


            # label_id = label_map[example["label"]]       # 标签映射

            # if ex_index < 5:
            #     logging.info("*** Example ***")
            #     logging.info("guid: %s" % (example["guid"]))
            #     logging.info("sentence: %s" % (example["ori_sentence"]))
            #     logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            #     logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logging.info("valid: %s" % " ".join([str(x) for x in valid]))
            #     logging.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            #     logging.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            #     logging.info("dep_key_list: %s" % " ".join([str(x) for x in dep_key_list]))
            #     logging.info("dep_order_dep_type_matrix: %s" % " ".join([str(x) for x in dep_order_dep_type_matrix]))
            #     logging.info("dep_path_dep_type_matrix: %s" % " ".join([str(x) for x in dep_path_dep_type_matrix]))
            #     logging.info("label: %s (id = %d)" % (example["label"], label_id))
            #
            features.append({
                # "input_ids": input_ids,
                # "input_mask": input_mask,
                # "label_id": label_id,
                # "valid_ids": valid,
                # "e1_mask": e1_mask,
                # "e2_mask": e2_mask,
                "dep_order_dep_type_matrix": dep_order_dep_type_matrix,
                "dep_path_dep_type_matrix": dep_path_dep_type_matrix,
                # "b_use_valid_filter": b_use_valid_filter,
                # "dep_key_list": dep_key_list,
                # "entity_start_mark_position":entity_start_mark_position
            })
        return features







    def build_dataset(self, examples, tokenizer, max_seq_length):   # # 为给定的输入示例构建一个适用于训练、验证或测试的数据集
        return self.convert_examples_to_features(examples, tokenizer, max_seq_length)  # 这一步调用convert_examples_to_features方法，将原始的文本示例（examples）转换为模型训练所需的特征。
  # 使用处理好的特征和指定的最大序列长度max_seq_length构建并返回一个REDataset实例。