# coding: utf-8

'''
    BiLSTM + Soft-Attention + deepout layer
'''

import random
from builtins import range
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.nn.beam_search import BeamSearch
from options import args

class BiEncoder(nn.Module):
    def __init__(self, feature_size, projected_size, hidden_size, max_frames):
        '''
        :param feature_size: size of frame feature extracted by CNN, 2048(2D) or 6144(2D + 3D)
        :param projected_size: size of embedded frame feature
        :param hidden_size: number of LSTM hidden units
        :param max_frames: encoder length
        '''
        super(BiEncoder, self).__init__()
        self.feature_size = feature_size
        self.projected_size = projected_size
        self.hidden_size = hidden_size
        self.max_frames = max_frames

        # feature embedding
        self.feature_embed = nn.Linear(feature_size, projected_size)
        nn.init.xavier_normal_(self.feature_embed.weight)
        self.frame_drop = nn.Dropout(p=args.drop_out)

        # lstm encoder 1
        self.lstm1_cell = nn.LSTMCell(projected_size, hidden_size)
        self.lstm1_drop = nn.Dropout(p=args.drop_out)

        # lstm encoder 2
        self.lstm2_cell = nn.LSTMCell(projected_size, hidden_size)
        self.lstm2_drop = nn.Dropout(p=args.drop_out)

        # BiLSTM
        #self.BiLSTM = nn.LSTM(projected_size, hidden_size, )

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = Variable(d.data.new(batch_size, self.hidden_size).zero_())
        lstm_state_c = Variable(d.data.new(batch_size, self.hidden_size).zero_())
        return lstm_state_h, lstm_state_c

    def forward(self, video_feats):
        batch_size = len(video_feats)

        # initialize bidirectional lstm state
        lstm1_h, lstm1_c = self._init_lstm_state(video_feats)
        lstm2_h, lstm2_c = self._init_lstm_state(video_feats)

        # use appearance feature only if feature_size=2048
        video_feats = video_feats[:, :, :self.feature_size].contiguous()

        # video feature embedding
        v = video_feats.view(-1, self.feature_size)
        v = self.feature_embed(v)
        v = self.frame_drop(v)
        v = v.view(batch_size, -1, self.projected_size)

        # lstmx_out stores encoded features
        lstm1_out = []
        lstm2_out = []
        for i in range(self.max_frames):
            lstm1_h, lstm1_c = self.lstm1_cell(v[:, i, :], (lstm1_h, lstm1_c))
            lstm1_h = self.lstm1_drop(lstm1_h)
            lstm1_out.append(lstm1_h)

            lstm2_h, lstm2_c = self.lstm2_cell(v[:, self.max_frames - i - 1, :], (lstm2_h, lstm2_c))
            lstm2_h = self.lstm2_drop(lstm2_h)
            lstm2_out.append(lstm2_h)

        # batch_size*max_frame*hidden_size
        lstm1_out = torch.stack(lstm1_out).permute(1, 0, 2)
        lstm2_out.reverse()
        lstm2_out = torch.stack(lstm2_out).permute(1, 0, 2)

        return lstm1_out, lstm2_out


class MFB(nn.Module):
    def __init__(self, hidden_size):
        super(MFB, self).__init__()
        self.hidden_size = hidden_size
        self.MFB_FACTOR_NUM = 5
        self.fc1 = nn.Linear(hidden_size, hidden_size * self.MFB_FACTOR_NUM)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, hidden_size * self.MFB_FACTOR_NUM)
        nn.init.xavier_normal_(self.fc2.weight)
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, feats1, feats2):
        *input_size, _ = feats1.size() # (b, m, h)
        feats1 = feats1.contiguous().view(-1, self.hidden_size) # (b*m, h)
        feats2 = feats2.contiguous().view(-1, self.hidden_size) # (b*m, h)
        feats1 = self.fc1(feats1)  # (b*m, h*factor)
        feats2 = self.fc2(feats2)  # (b*m, h*factor)
        feats = torch.mul(feats1, feats2)  # (b*m, h*factor)
        feats = self.drop_out(feats)  # (b*m, h*factor)
        feats = feats.view(-1, self.hidden_size, self.MFB_FACTOR_NUM) # (b*m, h, factor)
        feats = torch.sum(feats, 2)  # sum pool, (b*m, h)
        feats = torch.sqrt(F.relu(feats)) - torch.sqrt(F.relu(-feats))  # signed sqrt, (b*m, h)
        feats = F.normalize(feats) # (b*m, h)
        feats = feats.view(*input_size, self.hidden_size) # (b, m, h)
        return feats


class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, video_feats, lstm_h):
        '''
        soft attention machanism
        :param video_feats: batch_size*max_frames*hidden_size
        :param lstm_h: batch_size*hidden_size
        :return: encoded_feats: batch_size*hidden_size
        '''
        batch_size, max_frames, _ = video_feats.size()
        lstm_h = lstm_h.unsqueeze(1).repeat(1, max_frames, 1)
        inputs = torch.cat([video_feats, lstm_h], dim=2).view(-1, 2 * self.hidden_size)
        out = self.linear2(torch.tanh(self.linear1(inputs)))
        e = out.view(batch_size, max_frames)
        alpha = F.softmax(e, dim=1)
        encoded_feats = torch.bmm(alpha.unsqueeze(1), video_feats).squeeze(1)
        return encoded_feats


class Decoder(nn.Module):
    def __init__(self, encoded_size, projected_size, hidden_size, word_size, max_words, vocab):
        '''
        :param encoded_size: size of feature from soft attention model
        :param projected_size: size of embedded frame feature
        :param hidden_size: number of LSTM hidden units
        :param word_size: word embedding size
        :param max_words: decoder length
        :param vocab: training set vocabulary
        '''
        super(Decoder, self).__init__()
        self.encoded_size = encoded_size
        self.projected_size = projected_size
        self.hidden_size = hidden_size
        self.word_size = word_size
        self.batch_size = 100
        self.max_words = max_words
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # soft attention module
        self.softatt = SoftAttention(hidden_size)

        # word embedding matrix
        self.word_embed = nn.Embedding(self.vocab_size, self.word_size)
        # with open(vocab_embed_path, 'rb') as f:
        #     embedding_weight = pickle.load(f)
        # self.word_embed = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
        self.word_drop = nn.Dropout(p=args.drop_out)

        # lstm decoder
        self.lstm = nn.LSTMCell(hidden_size + word_size, hidden_size)
        self.lstm_drop = nn.Dropout(p=args.drop_out)

        # MFB module
        self.mfb = MFB(hidden_size)

        # linear layer
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(2 * hidden_size + word_size, hidden_size)
        nn.init.xavier_normal_(self.fc2.weight)
        #self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        #nn.init.xavier_normal_(self.fc3.weight)
        self.word_restore = nn.Linear(hidden_size, self.vocab_size)
        nn.init.xavier_normal_(self.word_restore.weight)

        # beam search
        self.beam_search = BeamSearch(vocab('<end>'), max_words, args.beam_size, per_node_beam_size=args.beam_size)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = Variable(d.data.new(batch_size, self.hidden_size).zero_())
        lstm_state_c = Variable(d.data.new(batch_size, self.hidden_size).zero_())
        return lstm_state_h, lstm_state_c

    def forward(self, video_encoded1, video_encoded2, captions, teacher_forcing_ratio=1.0):
        self.batch_size = video_encoded1.size(0)
        # linear concatenate
        video_encoded = torch.tanh(self.fc1(torch.cat([video_encoded1, video_encoded2], dim=2)))  # b*m*h

        # MFB fusion
        # video_encoded = self.mfb(video_encoded1, video_encoded2)

        infer = True if captions is None else False

        outputs = []
        # add a '<start>' sign
        start_id = self.vocab('<start>')
        start_id = Variable(video_encoded1.data.new(self.batch_size).long().fill_(start_id))
        word = self.word_embed(start_id)
        word = self.word_drop(word)  # b*w

        # initialize lstm decoder state
        lstm_h, lstm_c = self._init_lstm_state(video_encoded1)
        start_state = {'lstm_h': lstm_h, 'lstm_c': lstm_c, 'video_encoded': video_encoded}

        # training stage
        if not infer:
            for i in range(self.max_words):
                # <pad> index=0, captions[:, i].data.sum()=0 means
                # all words except <pad> are already fed into lstm decoder
                if captions[:, i].data.sum() == 0:
                    break

                # lstm decoder with attention model
                word_logits, lstm_h, lstm_c = self.decode(video_encoded, lstm_h, lstm_c, word)
                outputs.append(word_logits)

                # teacher_forcing: a training trick
                use_teacher_forcing = (random.random() < teacher_forcing_ratio)
                if use_teacher_forcing:
                    word_id = captions[:, i]
                else:
                    word_id = word_logits.max(1)[1]
                word = self.word_embed(word_id)
                word = self.word_drop(word)

            # unsqueeze(1): n --> n*1
            outputs = torch.cat([output.unsqueeze(1) for output in outputs], dim=1).contiguous()  # b*m*v
        # evaluating stage
        else:
            # details of allennlp.nn.beam_search：
            # https://allenai.github.io/allennlp-docs/api/allennlp.nn.beam_search.html#module-allennlp.nn.beam_search
            predictions, log_prob = self.beam_search.search(start_id, start_state, self.beam_step)
            max_prob, max_index = torch.topk(log_prob, 1)  # b*1
            max_index = max_index.squeeze(1)  # b
            for i in range(self.batch_size):
                outputs.append(predictions[i, max_index[i], :])
            outputs = torch.stack(outputs)
            #while True:
            #    pass
        return outputs

    def beam_step(self, last_predictions, current_state):
        '''
        A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        '''
        group_size = last_predictions.size(0) #batch_size or batch_size*beam_size
        batch_size = self.batch_size
        log_probs = []
        new_state = {}
        num = int(group_size / batch_size)  # 1 or beam_size
        for k, state in current_state.items():
            _, *last_dims = state.size()
            current_state[k] = state.reshape(batch_size, num, *last_dims)
            new_state[k] = []
        for i in range(num):
            # read current state
            lstm_h = current_state['lstm_h'][:, i, :]
            lstm_c = current_state['lstm_c'][:, i, :]
            video_encoded = current_state['video_encoded'][:, i, :]

            # decode stage
            word_id = last_predictions.reshape(batch_size, -1)[:, i]
            word = self.word_embed(word_id)
            word_logits, lstm_h, lstm_c = self.decode(video_encoded, lstm_h, lstm_c, word)

            # store log probabilities
            log_prob = F.log_softmax(word_logits, dim=1)  # b*v
            log_probs.append(log_prob)

            #update new state
            new_state['lstm_h'].append(lstm_h)
            new_state['lstm_c'].append(lstm_c)
            new_state['video_encoded'].append(video_encoded)

        # transform log probabilities
        # from list to tensor(batch_size*beam_size, vocab_size)
        log_probs = torch.stack(log_probs, dim=0).permute(1, 0, 2).reshape(group_size, -1)  # group_size*vocab_size

        # transform new state
        # from list to tensor(batch_size*beam_size, *)
        for k, state in new_state.items():
            new_state[k] = torch.stack(state, dim=0) #(beam_size, batch_size, *)
            _, _, *last_dims = new_state[k].size()
            dim_size = len(new_state[k].size())
            dim_size = range(2, dim_size)
            new_state[k] = new_state[k].permute(1, 0, *dim_size) #(batch_size, beam_size, *)
            new_state[k] = new_state[k].reshape(group_size, *last_dims) #(batch_size*beam_size, *)
        return (log_probs, new_state)

    def decode(self, video_encoded, lstm_h, lstm_c, word):
        video_encoded_att = self.softatt(video_encoded, lstm_h)  # b*h
        decoder_input = torch.cat([video_encoded_att, word], dim=1)
        lstm_h, lstm_c = self.lstm(decoder_input, (lstm_h, lstm_c))
        lstm_h = self.lstm_drop(lstm_h)
        # deepout layer
        decoder_output = torch.tanh(self.fc2(torch.cat([lstm_h, video_encoded_att, word], dim=1)))  # b*h
        #decoder_output = torch.tanh(self.fc3(torch.cat([lstm_h, video_encoded_att], dim=1)))  # b*h
        word_logits = self.word_restore(decoder_output)  # b*v
        return word_logits, lstm_h, lstm_c

class BiLSTM_attention_deepout(nn.Module):
    def __init__(self, feature_size, projected_size, hidden_size, word_size, max_frames, max_words, vocab):
        super(BiLSTM_attention_deepout, self).__init__()
        self.encoder = BiEncoder(feature_size, projected_size, hidden_size, max_frames)
        self.decoder = Decoder(hidden_size, projected_size, hidden_size, word_size, max_words, vocab)

    def forward(self, videos, captions, teacher_forcing_ratio=1.0):
        video_encoded1, video_encoded2 = self.encoder(videos)
        output = self.decoder(video_encoded1, video_encoded2, captions, teacher_forcing_ratio)
        return output
