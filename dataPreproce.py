import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def loadCSV():
    """

    :return:
    """
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values  # 8551 numpy.ndarray
    labels = df.label.values  # 8551 numpy.ndarray
    return sentences, labels


def findMax_len(sentences):
    """
    返回输入语句列表中的最大长度
    :param sentences:
    :return:
    """
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)  # 47
    return max_len


def input_ids_attention_masks(sentences):
    """

    :param sentences:
    :return:
    """
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...  遍历循环句子集list
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`         句子编码为IDs的长度补齐
        #   (6) Create attention masks for [PAD] tokens.             构造 [MASK]
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])  # append[[101, 2256,...., 102, 0, 0,...,0]]  1*64

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])  # append[[1, 1, 1, ...1, 0, 0..., 0 ]]   1*64

    return input_ids, attention_masks


def train_val_Dataset(input_ids, attention_masks, labels):
    # Combine the training inputs into a TensorDataset.  将用于训练的所有输入 融合进TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    print(type(dataset))
    print(dataset)

    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    return train_dataset, val_dataset


def train_val_DataLoader(train_dataset, val_dataset):
    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = 32
    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # 验证集不需要随机化，这里顺序读取就好
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader


def dataPrepare():
    # 从 CSV 文件读出 sentences 和 labels
    sentences, labels = loadCSV()
    # 求得 数据集的 maxLen 但是现在没啥用 定死在了 64
    maxLen = findMax_len(sentences)
    #
    input_ids, attention_masks = input_ids_attention_masks(sentences)
    # 转换成 Tensor
    input_ids = torch.cat(input_ids, dim=0)  # torch.Size([8551, 64])
    attention_masks = torch.cat(attention_masks, dim=0)  # torch.Size([8551, 64])
    labels = torch.tensor(labels)  # torch.Size([8551])
    # 制作 Dataset
    train_dataset, val_dataset = train_val_Dataset(input_ids, attention_masks, labels)
    # 制作 DataLoader
    train_dataloader, validation_dataloader = train_val_DataLoader(train_dataset, val_dataset)

    return train_dataloader, validation_dataloader


if __name__ == '__main__':
    print('Test begin')

    trainLoader, valLoader = dataPrepare()

    print('Test Finish')

