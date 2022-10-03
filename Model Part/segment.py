import torch
import pickle
from torch.nn import init
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
import torch.optim as optim
from torch.functional import F

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
bert_config = './data/bert'


def argmax(vec):

    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):

    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 kernel_num, filter_size, dropout):
        super().__init__()

        self.char_size = vocab_size
        self.char_ebd_dim = embedding_size
        self.kernel_num = kernel_num
        self.filter_size = filter_size
        self.dropout = dropout

        self.char_ebd = BertModel.from_pretrained(bert_config)
        self.char_cnn = nn.Conv2d(in_channels=1,
                                  out_channels=self.kernel_num,
                                  kernel_size=(self.filter_size, self.char_ebd_dim))
        self._init_weight()

    def _init_weight(self, scope=1.):
        init.xavier_uniform_(self.char_ebd.weight)

    def forward(self, input):
        encode = F.relu(self.char_cnn(input))
        encode = F.max_pool2d(encode,
                              kernel_size=(encode.size(2), 1))
        encode = F.dropout(encode.squeeze(), p=self.dropout)
        return encode


class Model(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=3, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag2id[START_TAG], :] = -10000
        self.transitions.data[:, tag2id[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(6, 1, self.hidden_dim // 2),
                torch.randn(6, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        #:param feats: LSTM+hidden2tag的输出
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        init_alphas[0][self.tag2id[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):

                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)

                trans_score = self.transitions[next_tag].view(1, -1)

                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2id[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        #  当前句子的tag路径score
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2id[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag2id[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag2id[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2id[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def test(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def handle_input(str, w2d):
    text = []
    Enword = ""
    index = 0
    line = str.strip()
    for i in range(len(line)):
        if is_alphabet(line[i]):
            index = i
            Enword += line[i]
        elif is_chinese(line[i]):
            text.append(w2d[line[i]])
    return text, Enword, index - len(Enword)+1


def calculate(x,y,id2word,id2tag,res=[],index=0,Enw=""):
    entity=[]
    res_entity =[]
    for j in range(len(x)):
        if j == index:
            res_entity.append(Enw)
        if id2tag[y[j]]=='B':
            entity=[id2word[x[j]]]
        elif id2tag[y[j]]=='M' and len(entity)!=0:
            entity.append(id2word[x[j]])
        elif id2tag[y[j]]=='E' and len(entity)!=0:
            entity.append(id2word[x[j]])
            res.append(entity)
            res_entity.append(entity)
            entity=[]
        elif id2tag[y[j]]=='S':
            entity=[id2word[x[j]]]
            res.append(entity)
            res_entity.append(entity)
            entity=[]
        else:
            entity=[]
            #res_entity = []
    return res, res_entity


with open('../data/datasave1.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
model = torch.load("./model/model" + str(7) + ".pkl")
while(1):
    string = input("请输入句子：")
    sentence, Enw, ind = handle_input(string, word2id)
    model.eval()
    entityres = []
    entityall = []
    sentence = torch.tensor(sentence, dtype=torch.long)
    score, predict = model.test(sentence)
    entityres, pred = calculate(sentence, predict, id2word, id2tag, entityres, ind, Enw)
    print('pred-seg :', ' '.join([''.join(t) for t in pred]))