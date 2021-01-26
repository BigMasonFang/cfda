#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchtext import data
import jieba
import numpy as np
import sys
# import torch.autograd as autograd
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report
import time
from typing import Callable, Iterator, Tuple
# from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance


#%%
# environment variable set 
BATCH_SIZE = 512
EMBEDDING_SIZE = 300
SENTENCE_LENGTH = 300
LEARNING_RATE = 1e-4
FILTER_NUM = 300
LABLE_NUM = 63 # 63 when filter > 11, 67 when filter > 6
INPUT_FILE_PATH = 'cleaned_data/'
EPOCHS = 1
TEXT = 0
LABEL = 0
STOP_WORDS = 'stop_words.txt'
LOG_FILE = 'logs/result_log.log'


#%%
def build_logger(log_file_name):
    # create logger 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create and set handler
    ch = logging.FileHandler(filename='.log', mode='w')
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # create and set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    return logger

def tokenizer(corpus: str) -> list:
    result = [w for w in jieba.cut(corpus)]
    return result


def data_iter_generate(train: str,
                       validate: str, 
                       test: str, 
                       tokenizer: Callable = None,
                       ) -> Tuple[Iterator]:

    """
    docstring
    """
    
    global TEXT, LABEL, SENTENCE_LENGTH, INPUT_FILE_PATH, STOP_WORDS

    # read stop list txt
    stop_words_list = [' ']
    with open(STOP_WORDS) as f:
        for l in f.readlines():
            stop_words_list.append(l.strip())

    # build field object
    if tokenizer is not None:
        TEXT = data.Field(sequential=True,
                        tokenize=tokenizer,
                        fix_length=SENTENCE_LENGTH,
                        stop_words=stop_words_list)
    else:
        TEXT = data.Field(sequential=True,
                        tokenize=lambda x:x,
                        fix_length=SENTENCE_LENGTH,
                        stop_words=stop_words_list)

    LABEL = data.Field(sequential=False, use_vocab=True)

    # splits data
    train, valid, test = data.TabularDataset.splits(
        path=INPUT_FILE_PATH, 
        train=train,
        validation=validate,
        test=test,
        format='CSV',
        # skip_header=True,
        csv_reader_params={'delimiter':'\t'},
        fields=[('label', LABEL), ('text', TEXT)])

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # data check 
    print(len(TEXT.vocab))
    print(train.examples[17].text)
    print(train.examples[17].label)
    # print(train.examples)
    print(LABEL.vocab.freqs)
    print(len(LABEL.vocab.freqs))
    # print(type(LABEL.vocab.freqs))
    # print(len(LABEL.vocab.freqs.elements()))
    # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    # build iterator
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, valid, test),
    batch_size=BATCH_SIZE,
    sort=False)

    return train_iter, val_iter, test_iter


class textCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size,
                 filter_num=50,
                 kernel_list=(1,2,3), dropout=0.5):
        super(textCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_num, (kernel, embedding_dim))
            for kernel in kernel_list
        ])

        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, x):
        x = self.embedding(x) # 32, 150, 200
        x = x.unsqueeze(1) # 32, 1, 150, 200

        conved = [F.hardswish(conv(x)).squeeze(3) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        out = torch.cat(pooled, dim=1) # 32, 300, 1, 1
        # out = out.view(x.size(0), -1) # 32, 300
        out = self.dropout(out)
        logit = self.fc(out) # 32, 68

        return logit


#%%
def train(train_iter, test_iter, model, epochs, logger):
    
    model.cuda()

    optimizer1 = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
    optimizer2 = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.9)
    optimizer3 = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.01)
    optimizer4 = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer4,
        patience=2,
        verbose=True,
        mode='min',
    )

    steps = 0
    # best_acc = 0
    # last_step = 0

    for epoch in range(1, epochs+1):
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            
            feature, target = feature.cuda(), target.cuda()

            optimizer4.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer4.step()

            steps += 1
            
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects/batch.batch_size
            sys.stdout.write(
                '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                    steps,
                    loss.item(),
                    accuracy.item(),
                    corrects.item(),
                    batch.batch_size
                ))

        logger.info('In Epoch[{}]'.format(epoch))
        _, avg_loss = eval(test_iter, model, logger)
        scheduler.step(avg_loss)


        # sys.stdout.write(
        #         '\rEpoch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch, 
        #                                                                     loss.item(), 
        #                                                                     accuracy.item(),
        #                                                                     corrects.item(),
        #                                                                     batch.batch_size))


def eval(data_iter, model, logger):
    model.eval()
    model.cuda()

    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        
        feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))

    logger.info('The result of model on validation set is: ')
    logger.info('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy, avg_loss


def predict(data_iter, model, label_field):
    model.eval()
    model.cuda()

    opt_list = list()
    true_label_list = list()

    for batch in data_iter:
        feature, label = batch.text.cuda(), batch.label.cuda()

        opt = model(feature.t_())

        # print(feature.size())

        _, predicted = torch.max(opt, 1)

        # print(opt.size())

        opt_list.extend([label_field.vocab.itos[i+1] for i in predicted.tolist()])
        # print(len(opt_list))
        true_label_list.extend([label_field.vocab.itos[i] for i in label.tolist()])

    return opt_list, true_label_list


def save(model, save_dir, epochs) -> str:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_path = '{}/steps_model_{}_{}'.format(save_dir, epochs, time.ctime())
    torch.save(model.state_dict(), save_path)

    return save_path


def confusion(true, predicted, label_field):
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(true, predicted, labels=label_field.vocab.itos),
        columns=label_field.vocab.itos)
    return confusion_matrix_df


def show_values(pc, fmt="%.2f", **kwargs):

    pc.update_scalarmappable()

    ax = pc.axes

    # print(ax)

    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)

        if np.all(color[:3] > 0.5):
            color = [0.0, 0.0, 0.0]

        else:
            color = [1.0, 1.0, 1.0]

        ax.text(x, y, fmt % value, ha='center', va='center', color=color, **kwargs)
    
    print(pc.get_array())
    print(ax.text)


def heatmap(AUC, title, xlabel,
            ylabel, xtick, ytick,
            figure_width, figure_height,
            correct_orientation=False, cmap="RdBu"):

    fig, ax = plt.subplots()

    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(xtick, minor=False)
    ax.set_yticklabels(ytick, minor=False)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim((0, AUC.shape[1]))

    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick10n = False
        t.tick20n = False
    for t in ax.yaxis.get_major_ticks():
        t.tick10n = False
        t.tick20n = False
    
    plt.colorbar(c)

    show_values(c)

    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_cls_report(cls_report, title='Classification Report', cmap='RdBu'):

    plotMat = list()
    support = list()
    class_names = list()

    for label, values_dict in cls_report.items():

        print(label)

        print(values_dict)

        if type(values_dict) is dict:
        
            v = [values_dict['precision'],values_dict['recall'],values_dict['f1-score']] 

            support.append(values_dict['support'])

            class_names.append(label)

            plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xtick = ['Precision', 'Recall', 'F1-score']
    ytick = ['{0}{1}'.format(class_names[idx], sup) for idx, sup in enumerate(support)]

    figure_width = 25
    figure_height = len(class_names) + 7

    correct_orientation = False

    heatmap(np.array(plotMat), title, xlabel,
            ylabel, xtick, ytick,
            figure_width, figure_height, correct_orientation,
            cmap=cmap)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusuion Matrix',
                          cmap=None,
                          normalize=True):

    fig, ax = plt.subplots()

    acc = np.trace(cm) / np.sum(cm).astype('float')

    misclass = 1 - acc

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    figure_height = len(target_names) + 7
    figure_width = len(target_names) + 10

    tick_marks = np.arange(len(target_names))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        threshold = cm.max() / 1.5
    else:
        threshold = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, '{:0.4f}'.format(cm[i, j]),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black')
        else:

            plt.text(j, i, '{:,}'.format(cm[i, j]),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black')

    plt.figure(figsize=(cm2inch(figure_width, figure_height)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\nAccuracy={:0.4f}; Misclass={:0.4f}'.format(acc, misclass))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    fig = plt.gcf()

    pass


#%%
if __name__ == '__main__':

    os.environ["CUDA_VISIABLE_DEVICES"] = "1"
    # os.environ["CUDA_LANUCH_BLOCKING"] = "1"

    # build logger
    logger = build_logger(LOG_FILE)

    test_set = ['train_aug_set_1.txt', 
                'train_aug_set_2.txt', 
                'train_aug_set_3.txt', 
                'train_aug_set_4.txt', 
                'train_aug_set_5.txt' 
    ]
    
    for train_set in test_set:
        # build data
        print(f"\n\n===============Start Build Data {train_set}=================\n\n")
        train_iter, test_iter, val_iter = data_iter_generate(
            train=train_set,
            validate='val_data.txt',
            test='test_data.txt',
        )

        # Train
        cnn = textCNN(len(TEXT.vocab), EMBEDDING_SIZE, LABLE_NUM, filter_num=FILTER_NUM)
        print(cnn)

        cnn = cnn.cuda()

        print(f"\n\n================Start Traing {set}===============\n\n")

        train(train_iter, test_iter, cnn, EPOCHS, logger)

        _, _ = eval(val_iter, cnn, logger)

        # Save
        save_path = save(cnn, 'model_save', EPOCHS)

        # Load
        cnn_infer = textCNN(len(TEXT.vocab), EMBEDDING_SIZE, LABLE_NUM, filter_num=FILTER_NUM).cuda()

        cnn_infer.load_state_dict(torch.load(train_set))
        # cnn_infer.load_state_dict(torch.load('model_save/steps_model_12_Sun Jan 17 11:11:59 2021'))

        # Inference
        predicted, true = predict(val_iter, cnn_infer, LABEL)

        cm = confusion_matrix(true, predicted, labels=LABEL.vocab.itos[1:])

        # confusion_matrix_df = confusion(true, predicted, LABEL)

        # print(confusion(true, predicted, LABEL))
        
        print(classification_report(true, predicted))

        # Plot
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False

        os.chdir('aug_pics')
        os.mkdir(train_set.partition('.')[0])
        os.chdir('..')

        plot_cls_report(classification_report(true, predicted, output_dict=True))
        plt.savefig(f"aug_pics/{train_set.partition('.')[0]}/cls_report.jpg", dpi=500, quality=95, bbox_inches='tight')

        plot_confusion_matrix(cm, LABEL.vocab.itos[1:])
        plt.savefig(f"aug_pics/{train_set.partition('.')[0]}/confusion.jpg", dpi=500, bbox_inches='tight')

        print('\n\n================End Training {set}==================\n\n')



    # visual_1 = ClassificationReport()

    



    
