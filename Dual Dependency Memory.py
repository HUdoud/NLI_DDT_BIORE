import argparse
import numpy as np
import torch
from torch import nn
from data_utils import (RE_Processor)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn


def yilai(tokenizer,formatted_sentence):
        # 把所有话拿出来，并建立关系
        dep_order = 'second_order'
        data_dir = '/root/autodl-tmp/NLI/data/chemprot_hf_new'
        processor = RE_Processor(dep_order=dep_order)  # 传入的是参数dep_order
        processor.prepare_keys_dict(data_dir=data_dir)  # 传入参数data_dir
        # 准备依赖树值字
        processor.prepare_vals_dict(data_dir)  # 例子：vals_dict={none': 0, 'self_loop': 1, 'mwe_in': 2, 'mwe_out': 3, 'det_in': 4, 'det_out': 5....}  这里会被打印出来
        # 准备标签字典
        processor.prepare_labels_dict(data_dir)  # 例子：labels_dict={1': 0, '0': 1}  这里会被打印出来
        all_feature_data = processor.get_train_examples(data_dir,formatted_sentence)
        # print(all_feature_data)
        dep_matrix = processor.build_dataset(all_feature_data, tokenizer,max_seq_length=128)
        result_horizontal = []

        for i in range(len(dep_matrix)):
                a = dep_matrix[i]['dep_order_dep_type_matrix']
                b = dep_matrix[i]['dep_path_dep_type_matrix']
                # 水平拼接
                c = np.concatenate((a, b), axis=1)
                target_size = 256
                rows_to_add = target_size - c.shape[0]
                cols_to_add = target_size - c.shape[1]
                # 分别计算上下和左右需要填充的行数和列数
                # 使用 //2 保证尽可能均匀分布
                pad_top = rows_to_add // 2
                pad_bottom = rows_to_add - pad_top
                pad_left = cols_to_add // 2
                pad_right = cols_to_add - pad_left
                # 应用填充
                try:
                        padded_matrix = np.pad(c, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                               constant_values=(0,))
                except ValueError as e:
                        print(f"错误: {e}")
                        print(f"pad_top: {pad_top}, pad_bottom: {pad_bottom}, pad_left: {pad_left}, pad_right: {pad_right}")
                        # 这里可以添加更多的调试信息，如 c 的尺寸等
                        print(f"c 的形状: {c.shape}")

                result_horizontal.append(padded_matrix)

        dep_type_matrix = torch.Tensor(result_horizontal)

        fc = nn.Linear(256*256, 3)
        dep_type_matrix = dep_type_matrix.view(dep_type_matrix.size(0), -1)
        # 通过全连接层
        dep_type_matrix = fc(dep_type_matrix)
        return dep_type_matrix

        # print(x)

