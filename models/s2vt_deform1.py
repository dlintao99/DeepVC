'''
@file s2vt.py: the implementation of S2VT model with some changes.
Encoder and decoder does not share parameters. Encoder's output serve as the initial hidden state of decoder's RNN.
relevant paper: Sequence to Sequence -- Video to Text, ICCV 2015.
'''

import sys
import torch
import torch.nn as nn
from utils import Vocabulary

class Encoder(nn.Module):

    def __init__(self, 
                 num_time_steps_encoder, 
                 dim_frame_feature,
                 dim_embedded_frame_feature,
                 dim_hidden_state_encoder,
                 rate_dropout):

        super(Encoder, self).__init__()
        self.num_time_steps_encoder = num_time_steps_encoder
        self.dim_frame_feature = dim_frame_feature
        self.dim_embedded_frame_feature = dim_embedded_frame_feature
        self.dim_hidden_state_encoder = dim_hidden_state_encoder
        self.rate_dropout = rate_dropout

    def forward(self, feats_videos):
        
        batch_size = len(feats_videos)

        # initialize hidden states of LSTM
        hidden_state_h = torch.zeros(batch_size, self.dim_hidden_state_encoder, requires_grad = True)
        hidden_state_c = torch.zeros(batch_size, self.dim_hidden_state_encoder, requires_grad = True)

        # define embed operation
        embed_frame_feature = nn.Linear(self.dim_frame_feature, self.dim_embedded_frame_feature)
        nn.init.xavier_normal_(embed_frame_feature.weight)

        # embed frame feature
        feats_videos = feats_videos.view(-1, self.dim_frame_feature)
        feats_videos = embed_frame_feature(feats_videos)
        feats_videos = feats_videos.view(batch_size, -1, self.dim_embedded_frame_feature)

        # define encoder's LSTM unit
        LSTM_unit_encoder = nn.LSTMCell(self.dim_embedded_frame_feature, self.dim_hidden_state_encoder)
        #LSTM_unit_dropout = nn.Dropout(p = self.rate_dropout)

        # initialize the list of encoder's outputs
        list_outputs_encoder = []

        for i in range(self.num_time_steps_encoder):
            hidden_state_h, hidden_state_c = LSTM_unit_encoder(feats_videos[:, i, :], (hidden_state_h, hidden_state_c))
            list_outputs_encoder.append(hidden_state_h)
        
        # generate encoder's output with size (batch_size, num_time_steps_encoder, dim_frame_feature)
        outputs_encoder = torch.stack(list_outputs_encoder, dim = 1)

        return outputs_encoder

class Decoder(nn.Module):

    def __init__(self, 
                 dim_embedded_feature_encoder, 
                 num_time_steps_decoder,
                 dim_hidden_state_decoder,
                 dim_embedded_word,
                 rate_dropout,
                 vocabulary):
        
        super(Decoder, self).__init__()
        self.dim_embedded_feature_encoder = dim_embedded_feature_encoder
        self.num_time_steps_decoder = num_time_steps_decoder
        self.dim_hidden_state_decoder = dim_hidden_state_decoder
        self.dim_embbeded_word = dim_embbeded_word
        self.rate_dropout = rate_dropout
        self.vocabulary = vocabulary

    def forward(self, outputs_encoder, labels, teacher_forcing_ratio = 1.0):
        
        batch_size = len(outputs_encoder)

        # embed output feature of encoder
        dim_output_encoder = outputs_encoder.size()[-1]
        embed_features_encoder = nn.Linear(dim_output_encoder, self.dim_embedded_feature_encoder)

        # is current stage "train" or "apply"?
        flag_apply = True if labels is None else False

        # set initial word '<start>' and embed it
        ids_start = torch.full((batch_size, 1), self.vocabulary('<start>'))
        num_words_vocabulary = len(self.vocabulary)
        embed_word = nn.Embedding(num_words_vocabulary, self.dim_embedded_word)
        embedding_input_words_decoder = embed_word(ids_start)

        # initialize hidden states of LSTM with encoder's outputs
        hidden_state_h = torch.zeros(batch_size, self.dim_hidden_state_encoder, requires_grad = True)
        hidden_state_c = torch.zeros(batch_size, self.dim_hidden_state_encoder, requires_grad = True)

        # define decoder's LSTM unit
        LSTM_unit_decoder = nn.LSTMCell(self.dim_embedded_feature_encoder + self.dim_embbeded_word, self.dim_hidden_state_decoder)
        #LSTM_unit_dropout = nn.Dropout(p = self.rate_dropout)

        list_outputs_decoder = []

        if flag_apply is None:
            # train stage
            for i in range(self.num_time_steps_decoder):
                input_LSTM_unit_decoder = torch.cat([embed_features_encoder[:, i, :], embedding_input_words_decoder], dim = 1)
                hidden_state_h, hidden_state_c = LSTM_unit_decoder(input_LSTM_unit_decoder, (hidden_state_h, hidden_state_c))

        else:
            # apply stage
            pass

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

class S2VT(nn.Module):
    def __init__(self, 
                 num_time_steps_encoder, 
                 num_time_steps_decoder,
                 dim_frame_feature, 
                 dim_embedded_frame_feature,
                 dim_hidden_state_encoder, 
                 dim_hidden_state_decoder,
                 dim_embedded_feature_encoder,
                 dim_embedded_word,
                 max_words,
                 vocabulary,
                 rate_dropout):
        
        super(S2VT, self).__init__()
        self.num_time_steps_encoder = num_time_steps_encoder
        self.num_time_steps_decoder = num_time_steps_decoder
        self.dim_frame_feature = dim_frame_feature
        self.dim_embedded_frame_feature = dim_embedded_frame_feature
        self.dim_hidden_state_encoder = dim_hidden_state_encoder
        self.dim_hidden_state_decoder = dim_hidden_state_decoder
        self.dim_embedded_feature_encoder = dim_embedded_feature_encoder
        self.dim_embbeded_word = dim_embbeded_word
        self.max_words = max_words
        self.vocabulary = vocabulary
        self.rate_dropout = rate_dropout

        self.encoder = Encoder(num_time_steps_encoder, 
                               dim_frame_feature, 
                               dim_embedded_frame_feature, 
                               dim_hidden_state_encoder, 
                               rate_dropout)
        self.decoder = Decoder(dim_embedded_feature_encoder, 
                               num_time_steps_decoder, 
                               dim_hidden_state_decoder, 
                               dim_embedded_word, 
                               rate_dropout,
                               vocabulary)

    def forward(self, feats_videos, labels, teacher_forcing_ratio = 1.0):
        outputs_encoder = self.encoder(feats_videos)
        output = self.decoder(outputs_encoder, labels, teacher_forcing_ratio)
        return output
