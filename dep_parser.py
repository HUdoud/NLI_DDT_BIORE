import copy


class DepInstanceParser():
    def __init__(self, basicDependencies, tokens=[]):  # 它是一个包含依存关系信息的结构（可能是一个列表或其他数据结构），以及一个默认为空的tokens列表
        self.basicDependencies = basicDependencies  # 将传入的基本依存关系信息保存为实例属性。
        self.tokens = tokens
        self.words = []
        self.dep_governed_info = []
        self.dep_parsing()  # 在构造函数的最后调用了一个名为dep_parsing的方法，这个方法很可能是用来处理和解析basicDependencies中的依存关系信息，将解析结果填充到self.words和self.dep_governed_info等属性中。


    def dep_parsing(self):
        if len(self.tokens) > 0:
            words = []
            for token in self.tokens:
                words.append(self.change_word(token))  # 对每个令牌调用change_word方法进行处理（比如进行标准化、修正等），然后将处理后的结果添加到words列表中。
            dep_governed_info = [   # 生成依存被支配信息 使用列表推导式和enumerate函数，为words列表中的每个单词创建一个包含单词信息的字典，并将这些字典收集到列表dep_governed_info中
                {"word": word}
                for i,word in enumerate(words)
            ]     # 创造了一个dict{‘word’:'单词'}
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicDependencies)
        for dep in self.basicDependencies:
            dependent_index = dep['dependent'] - 1   # 由于索引在dep字典中是从1开始的（通常是基于句子中单词的顺序），而Python列表的索引是从0开始的，所以需要减去1来进行调整。
            governed_index = dep['governor'] - 1
            dep_governed_info[dependent_index] = {
                "governor": governed_index,
                "dep": dep['dep']
            }
        self.dep_governed_info = dep_governed_info   # 列表中对应于依存词索引dependent_index的元素
        # print(dep_governed_info)

    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def get_init_dep_matrix(self):   # 目的是初始化依存关系的邻接矩阵和类型矩阵
        dep_adj_matrix = [[0] * len(self.words) for _ in range(len(self.words))]   # 创建一个大小为len(self.words)乘len(self.words)的邻接矩阵dep_adj_matrix，初始值全部为0。
        dep_type_matrix = [["none"] * len(self.words) for _ in range(len(self.words))] # 创建一个同样大小的类型矩阵dep_type_matrix，初始值全部为"none"。
        for i in range(len(self.words)):
            dep_adj_matrix[i][i] = 1   # 将邻接矩阵中对角线上的元素设置为1，表示每个单词与自己存在依存关系（自循环）
            dep_type_matrix[i][i] = "self_loop"   # 将类型矩阵中对角线上的元素设置为"self_loop"，表示这是一个自循环的依存关系。
        return dep_adj_matrix, dep_type_matrix   # 返回初始化后的邻接矩阵和类型矩阵

    def get_first_order(self, direct=False):   # 用于生成第一阶依存关系的邻接矩阵和类型矩阵
        dep_adj_matrix, dep_type_matrix = self.get_init_dep_matrix()   # 初始化邻接矩阵和类型矩阵

        for i, dep_info in enumerate(self.dep_governed_info):
            governor = dep_info["governor"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[i][governor] = 1
            dep_adj_matrix[governor][i] = 1
            dep_type_matrix[i][governor] = dep_type if direct is False else "{}_in".format(dep_type)   # 当direct参数为False时,表示不区分依存关系的方向，邻接矩阵和类型矩阵的相关元素直接使用dep_type。
            dep_type_matrix[governor][i] = dep_type if direct is False else "{}_out".format(dep_type)   # 当direct参数为True时，需要区分依存关系的方向。此时，类型矩阵中单词i到它的支配词governor的依存关系标记为"{}_in".format(dep_type)，反向的依存关系标记为"{}_out".format(dep_type)。

        return dep_adj_matrix, dep_type_matrix   # 句子矩阵 和 依赖树 矩阵

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):    # 用于生成下一阶（比如从一阶依存到二阶依存）的依存关系的邻接矩阵和类型矩阵
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)   # 复制原始矩阵
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):   # 扩展依存关系 过三层嵌套循环遍历所有可能的依存关系组合
            for first_order_index in range(len(dep_adj_matrix[target_index])):   # 外层循环遍历所有目标单词target_index。
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):    # 中层循环遍历与目标单词有直接依存关系的单词first_order_index。
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue                                                           # 内层循环遍历与first_order_index有直接依存关系的单词second_order_index，即目标单词的间接依存关系。
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1                      # 如果找到一个有效的间接依存关系（即target_index和second_order_index之间的关系），并且这个关系之前没有被记录过（new_dep_adj_matrix[target_index][second_order_index]为0），则在新的邻接矩阵中标记这个关系为存在（设置为1），并从原始类型矩阵中复制相应的依存类型到新的类型矩阵。
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):        # 用于搜索和确定从一个起始单词到一个结束单词的依存路径
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_range, end_range, direct=False):         # 用于找出并标记句子中两组特定单词之间的依存路径，这对于分析句子的句法结构和理解句子中单词之间的复杂关系非常有用。
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()   # 初始化邻接矩阵和类型矩阵

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)    # 获取第一阶依存关系矩阵
        for start_index in start_range:       # 遍历起始和结束范围内的单词索引 从起始索引到结束索引的依存路径，返回路径中的索引列表。
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):   # 标记依存路径
                    dep_path_adj_matrix[start_index][right_index] = 1
                    dep_path_type_matrix[start_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[end_index][left_index] = 1
                    dep_path_type_matrix[end_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix   # 返回依存路径矩阵
