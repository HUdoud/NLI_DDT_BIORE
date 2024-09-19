import os
import re
import json
import argparse
from tqdm import tqdm
from corenlp import StanfordCoreNLP
from spacynlp import SpaCyDepTreeParser

FULL_MODEL = './stanford-corenlp-full-2018-10-05'  # 指定了 Stanford CoreNLP 工具包的路径。
punctuation = ['.', ',', ':', '?', '!', '(', ')', '"', '[', ']', ';', '\'']  # 是一个包含各种标点符号的列表。
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']   # 是一个包含常见短语类型（名词短语、介词短语、动词短语等）的列表。

# 将字符串左右替换，例如‘hello’-->'-LRB-hello-RRB-'
def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char


def split_punc(s):
    word_list = ''.join([" "+x+" " if x in punctuation else x for x in s]).split()  # 对于字符串中的每个字符 x，如果它是标点符号，则在其前后添加空格，否则保持不变。
    return [w for w in word_list if len(w) > 0]   # 以空格为分隔符，得到一个单词列表 word_list,过滤掉列表中长度为零的元素，得到最终的单词列表。

def tokenize(sentence):
    words = []
    for seg in re.split('(<e1>|</e1>|<e2>|</e2>)', sentence):  # re正则表达式
        # 使用 re.split('(<e1>|</e1>|<e2>|</e2>)', sentence) 对句子进行分割。其中正则表达式 (<e1>|</e1>|<e2>|</e2>) 表示将 "<e1>", "</e1>", "<e2>", "</e2>" 这四个字符串看作分隔符进行分割。分割后得到一个由句子和实体标记组成的列表。
        if seg in ["<e1>", "</e1>", "<e2>", "</e2>"]:
            words.append(seg)
        else:
            words.extend(split_punc(seg))
            # 如果 seg 是实体标记 ("<e1>", "</e1>", "<e2>", "</e2>")，则将其添加到 words 列表中。
            # 否则，调用之前定义的 split_punc(seg) 函数，将 seg 进行进一步的分割，并将分割后的单词列表添加到 words 列表中。
    return words

def read_txt(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            sentence = data['sentence']
            label = data['label']
            entry_id = data['id']

            # 检查是否存在 @CHEM-GENE$ 实体标记
            chem_gene_match = re.search('@CHEM\-GENE\$', sentence)
            if chem_gene_match:
                e1_start = chem_gene_match.start()
                e1_end = chem_gene_match.end()
                e1 = sentence[e1_start:e1_end]
                e1 = re.sub(r'<[^>]*>', '', e1)
                e1 = re.sub(r'\s+', ' ', e1)
                sentence = sentence.replace('@CHEM-GENE$', f'<e1>CHEM-GENE</e1>')
                ori_sentence = tokenize(sentence)
                filtered_sentence = [word for word in ori_sentence if
                                     word not in {'<e1>', '</e1>'}]
                datas.append({
                    'id': entry_id,
                    'sentence': sentence,
                    'label': label,
                    'e1': e1,
                    'e2': None,  # 因为只有一个实体，所以 e2 设置为 None
                    'tokenized_sentence': ori_sentence,
                    'tokenized_sentence_dell': filtered_sentence
                })
            else:
                # 假设 @CHEMICAL$和 @GENE$ 是您的实体标记
                e1_start = sentence.find('@CHEMICAL$')
                e1_end = e1_start + len('@CHEMICAL$') if e1_start != -1 else -1
                e2_start = sentence.find('@GENE$')
                e2_end = e2_start + len('@GENE$') if e2_start != -1 else -1

                # 如果找到了实体标记
                if e1_start != -1 and e2_start != -1:
                    e1 = sentence[e1_start:e1_end]
                    e1 = re.sub(r'<[^>]*>', '', e1)
                    e1 = re.sub(r'\s+', ' ', e1)
                    e2 = sentence[e2_start:e2_end]
                    e2 = re.sub(r'<[^>]*>', '', e2)
                    e2 = re.sub(r'\s+', ' ', e2)
                    sentence = sentence.replace('@CHEMICAL$', f'<e1>CHEMICAL</e1>')
                    sentence = sentence.replace('@GENE$', f'<e2>GENE</e2>')

                    # 分词和实体标记
                    ori_sentence = tokenize(sentence)
                    filtered_sentence = [word for word in ori_sentence if
                                         word not in {'<e1>', '</e1>', '<e2>', '</e2>'}]

                    # 添加到 datas 列表中
                    datas.append({
                        'id': entry_id,
                        'e1': e1,
                        'e2': e2,
                        'label': label,
                        'sentence': sentence,
                        'tokenized_sentence': ori_sentence,
                        'tokenized_sentence_dell': filtered_sentence
                    })
                else:
                    print(f"Entity markers not found in sentence: {sentence}")

        return datas

def request_features_from_stanford(data_dir, flag):
    data_path = os.path.join(data_dir, flag + '.tsv')  #  方法将文件路径组合得到完整路径 data_path。
    print("request_features_from_stanford {}".format(data_path))
    if not os.path.exists(data_path):
        print("{} not exist".format(data_path))  # 检查文件是否存在，如果不存在则输出错误信息并返回。
        return
    all_sentences = read_txt(data_path)
    sentences_str = []
    for entry in all_sentences:
        id = entry['id']
        e1 = entry['e1']
        e2 = entry['e2']
        label = entry['label']
        raw_sentence = entry['sentence']
        ori_sentence = entry['tokenized_sentence']
        sentence = entry['tokenized_sentence_dell']
        sentence = [change(s) for s in sentence]
        sentences_str.append([e1, e2, label, raw_sentence, ori_sentence, sentence])
    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='en') as nlp:
        for e1, e2, label, raw_sentence, ori_sentence, sentence in tqdm(sentences_str):
            props = {'timeout': '5000000','annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true' ,
                     'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
            results=nlp.annotate(' '.join(sentence), properties=props)
            results["e1"] = e1
            results["e2"] = e2
            results["label"] = label
            results["raw_sentence"] = raw_sentence
            results["ori_sentence"] = ori_sentence
            results["word"] = sentence
            all_data.append(results)
    assert len(all_data) == len(sentences_str)
    # 使用 with 语句分别打开保存分析结果的文本文件和依存分析结果的文本文件，并分别将依存句法分析结果写入文件。
    with open(os.path.join(data_dir, flag + '.txt'), 'w', encoding='utf8') as fout_text, \
            open(os.path.join(data_dir, flag + '.txt.dep'), 'w', encoding='utf8') as fout_dep:
        for data in all_data:
            #text
            fout_text.write("{}\t{}\t{}\t{}\n".format(data["e1"], data["e2"], data["label"], " ".join(data["ori_sentence"])))
            #dep
            for dep_info in data["sentences"][0]["basicDependencies"]:
                fout_dep.write("{}\t{}\t{}\n".format(dep_info["governor"], dep_info["dependent"], dep_info["dep"]))
            fout_dep.write("\n")
            # 打印依赖关系数量和原始句子单词数量
            assert len(data["sentences"][0]["basicDependencies"]) + 2 == len(data["ori_sentence"]) or len(data["sentences"][0]["basicDependencies"]) + 4 == len(data["ori_sentence"])
            # try:
            #     assert len(data["sentences"][0]["basicDependencies"]) + 4 == len(data["ori_sentence"])
            # except AssertionError:
            #     print("Assertion failed for sentence:", " ".join(data["ori_sentence"]))
            #     print("Dependencies count:", len(data['sentences'][0]['basicDependencies']))
            #     print("Original sentence word count:", len(data['ori_sentence']))


# 函数的作用是从文件中读取数据，并使用 SpaCy 对句子进行依存句法分析，并将结果保存到文件中。
def request_features_from_spacy(data_dir, flag):
    data_path = os.path.join(data_dir, flag + '.tsv')
    print("request_features_from_stanford {}".format(data_path))
    if not os.path.exists(data_path):
        print("{} not exist".format(data_path))
        return
    all_sentences = read_txt(data_path)
    sentences_str = []
    for e1, e2, label, raw_sentence, ori_sentence, sentence in all_sentences:
        sentence = [change(s) for s in sentence]
        sentences_str.append([e1, e2, label, raw_sentence, ori_sentence, sentence])
    all_data = []
    parser = SpaCyDepTreeParser()
    for e1, e2, label, raw_sentence, ori_sentence, sentence in tqdm(sentences_str):
        results = parser.parsing(' '.join(sentence))
        results["e1"] = e1
        results["e2"] = e2
        results["label"] = label
        results["raw_sentence"] = raw_sentence
        results["ori_sentence"] = ori_sentence
        results["word"] = sentence
        all_data.append(results)
    assert len(all_data) == len(sentences_str)
    with open(os.path.join(data_dir, flag + '.txt'), 'w', encoding='utf8') as fout_text, \
            open(os.path.join(data_dir, flag + '.txt.dep'), 'w', encoding='utf8') as fout_dep:
        for data in all_data:
            #text
            fout_text.write("{}\t{}\t{}\t{}\n".format(data["e1"], data["e2"], data["label"], " ".join(data["ori_sentence"])))
            #dep
            for dep_info in data["sentences"][0]["basicDependencies"]:
                fout_dep.write("{}\t{}\t{}\n".format(dep_info["governor"], dep_info["dependent"], dep_info["dep"]))
            fout_dep.write("\n")
            # 打印依赖关系数量和原始句子单词数量
            assert len(data["sentences"][0]["basicDependencies"]) + 2 == len(data["ori_sentence"]) or len(data["sentences"][0]["basicDependencies"]) + 4 == len(data["ori_sentence"])


def get_labels_dict(data_dir):
    labels_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.txt')
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                e1,e2,label,sentence = line.strip().split("\t")
                labels_set.add(label)
    save_path = os.path.join(data_dir, "label.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(labels_set), f, ensure_ascii=False)

def get_dep_type_dict(data_dir):
    dep_type_set = set()
    for flag in ["train", "dev", "test"]:
        data_path = os.path.join(data_dir, flag + '.txt.dep')
        if not os.path.exists(data_path):
            continue
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                governor,dependent,dep_type = line.strip().split("\t")
                dep_type_set.add(dep_type)
    save_path = os.path.join(data_dir, "dep_type.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(list(dep_type_set), f, ensure_ascii=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    for flag in ["dev", "test", "train"]:
        request_features_from_stanford(args.data_path, flag)
    get_labels_dict(args.data_path)
    get_dep_type_dict(args.data_path)
