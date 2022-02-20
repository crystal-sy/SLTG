import tensorflow as tf
#产生训练数据，
#对Generator来说，只在预训练中使用dataloader来得到训练数据，
#对Discriminator来说，在预训练和对抗过程中都要使用dataloader来得到训练数据。

def generator_dataloader(data_file, batch_size):
    token_stream = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]

            if len(parse_line) == 20:
                token_stream.append(parse_line)
    #把给定的元组、列表和张量等数据进行特征切片后随机排序
    return tf.data.Dataset.from_tensor_slices(token_stream).shuffle(len(token_stream)).batch(batch_size)


def discriminator_dataloader(positive_file, negative_file, batch_size):
    examples = []
    labels = []
    with open(positive_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == 20:
                examples.append(parse_line)
                labels.append([0, 1])

    with open(negative_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            if len(parse_line) == 20:
                examples.append(parse_line)
                labels.append([1, 0])
    return tf.data.Dataset.from_tensor_slices((examples, labels)).shuffle(len(examples)).batch(batch_size)
