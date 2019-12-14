'''
@file s2vt_deform2.py: the implementation of S2VT model with some changes.
Encoder and decoder does not share parameters. Encoder's output serve as 
one part of decoder's inputs, the other part is word embedding.
relevant paper: Sequence to Sequence -- Video to Text, ICCV 2015.
'''

import sys
import torch
import torch.nn as nn
from utils import Vocabulary

class S2VT(nn.Module):

    def __init__(self, 
                 num_time_steps_encoder, 
                 num_time_steps_decoder,
                 dim_frame_feature, 
                 dim_embedded_frame_feature,
                 dim_hidden_state, 
                 dim_embedded_word,
                 vocabulary,
                 rate_dropout,
                 DEVICE):
        
        super(S2VT, self).__init__()
        self.num_time_steps_encoder = num_time_steps_encoder
        self.num_time_steps_decoder = num_time_steps_decoder
        self.dim_frame_feature = dim_frame_feature
        self.dim_embedded_frame_feature = dim_embedded_frame_feature
        self.dim_hidden_state = dim_hidden_state
        self.dim_embedded_word = dim_embedded_word
        self.vocabulary = vocabulary
        self.rate_dropout = rate_dropout
        self.DEVICE = DEVICE

        self.num_words_vocabulary = len(self.vocabulary)

        # define S2VT's learnable parameters

        ## define embed operation
        self.embed_frame_feature = nn.Linear(self.dim_frame_feature, self.dim_embedded_frame_feature)
        nn.init.xavier_normal_(self.embed_frame_feature.weight)
        self.embed_word = nn.Embedding(self.num_words_vocabulary, self.dim_embedded_word)
        nn.init.xavier_normal_(self.embed_word.weight)

        ## define LSTM unit (encoder and decoder share parameters)
        self.LSTM_unit = nn.LSTMCell(self.dim_embedded_frame_feature + self.dim_embedded_word, self.dim_hidden_state)
        #self.LSTM_unit_dropout = nn.Dropout(p = self.rate_dropout)

        ## define linear layer to map hidden state to logits of words
        ## Linear(x) = w * x + b
        self.hiddenState2logits = nn.Linear(self.dim_hidden_state, self.num_words_vocabulary)
    
    def forward(self, feats_videos, labels, teacher_forcing_ratio = 1.0):
        
        #encode

        batch_size = len(feats_videos)

        ## define padding of embedding words
        padding_embedding_words = torch.zeros(batch_size, self.dim_embedded_word, device = self.DEVICE)

        ## initialize hidden states of LSTM
        hidden_state_h = torch.zeros(batch_size, self.dim_hidden_state, device = self.DEVICE)
        hidden_state_c = torch.zeros(batch_size, self.dim_hidden_state, device = self.DEVICE)

        ## embed frame feature
        feats_videos = feats_videos.view(-1, self.dim_frame_feature)
        feats_videos = self.embed_frame_feature(feats_videos)
        feats_videos = feats_videos.view(batch_size, self.num_time_steps_decoder, self.dim_embedded_frame_feature)

        for i in range(self.num_time_steps_encoder):
            input_LSTM_unit = torch.cat([feats_videos[:, i, :], padding_embedding_words], dim = 1)
            hidden_state_h, hidden_state_c = self.LSTM_unit(input_LSTM_unit, (hidden_state_h, hidden_state_c))

        # decode
        
        ## define padding of video features
        padding_video_features = torch.zeros(batch_size, self.dim_embedded_frame_feature, device = self.DEVICE)

        ## set initial word '<start>' and embed it
        ## nn.Embedding requires torch.long
        ids_start = torch.full((batch_size, ), self.vocabulary('<start>'), dtype = torch.long, device = self.DEVICE)
        embedding_input_words_decoder = self.embed_word(ids_start)

        ## is current stage "train" or "apply"?
        flag_apply = True if labels is None else False

        list_outputs_decoder = []
        #list_logProbs_words = []

        if flag_apply is False:
            # train stage
            for i in range(self.num_time_steps_decoder):
                input_LSTM_unit = torch.cat([padding_video_features, embedding_input_words_decoder], dim = 1)
                hidden_state_h, hidden_state_c = self.LSTM_unit(input_LSTM_unit, (hidden_state_h, hidden_state_c))
                logit_words = self.hiddenState2logits(hidden_state_h) # batch_size * num_words_vocabulary
                list_outputs_decoder.append(logit_words)
                embedding_input_words_decoder = self.embed_word(labels[:, i])
            # outputs_decoder.size(): batch_size * num_time_steps_decoder * num_words_vocabulary
            outputs_decoder = torch.cat([outputs_decoder_every_step.unsqueeze(1) for outputs_decoder_every_step in list_outputs_decoder], dim=1)

        else:
            # apply stage
            for i in range(self.num_time_steps_decoder):
                input_LSTM_unit = torch.cat([padding_video_features, embedding_input_words_decoder], dim = 1)
                hidden_state_h, hidden_state_c = self.LSTM_unit(input_LSTM_unit, (hidden_state_h, hidden_state_c))
                logit_words = self.hiddenState2logits(hidden_state_h) # batch_size * num_words_vocabulary
                #logProbs_words = nn.functional.log_softmax(logit_words, dim = 1)
                #list_logProbs_words.append(logProbs_words)
                # max()[0] are values, max()[1] are indices
                id_words = logit_words.max(1)[1] # batch_size
                list_outputs_decoder.append(id_words)
                embedding_input_words_decoder = self.embed_word(id_words)
            # outputs_decoder.size(): batch_size * num_time_steps_decoder
            outputs_decoder = torch.cat([outputs_decoder_every_step.unsqueeze(1) for outputs_decoder_every_step in list_outputs_decoder], dim=1)

        return outputs_decoder
