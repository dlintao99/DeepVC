from .s2vt import S2VT
from .bilstm_attention_deepout import BiLSTM_attention_deepout
from .bilstm_attention_seqDeepout import BiLSTM_attention_seqDeepout

def build_model(args, vocab, DEVICE):
    if (args.model == 'S2VT'):
        model = S2VT(args.max_frames, 
                            args.max_words,
                            args.feature_size, 
                            args.projected_size,
                            args.hidden_size, 
                            args.word_size,
                            vocab,
                            args.drop_out,
                            DEVICE)
    elif (args.model == 'BiLSTM_attention_deepout'):
        model = BiLSTM_attention_deepout(args.feature_size, 
                                        args.projected_size, 
                                        args.hidden_size, 
                                        args.word_size, 
                                        args.max_frames, 
                                        args.max_words, 
                                        vocab)
    elif (args.model == 'BiLSTM_attention_seqDeepout'):
        model = BiLSTM_attention_seqDeepout(args.feature_size, 
                                        args.projected_size, 
                                        args.hidden_size, 
                                        args.word_size, 
                                        args.max_frames, 
                                        args.max_words, 
                                        vocab)
    return model