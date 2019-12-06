from args import test_prediction_txt_path, test_reference_txt_path
# file = test_reference_txt_path
file = test_prediction_txt_path
with open(file, 'r') as f:
    lines = f.readlines()
    sents = []
    count = 0
    for line in lines:
        sent = line.split('\t')[1].strip().split(' ')
        count += len(sent)
        sents.append(sent)
print('{} words in {} test predictions, {:.3} words per sentence on average'.
      format(count, len(sents), count / len(sents)))
