from collections import Counter



def get_mwes(file):
    sents = []
    mwe_list = []

    with open(file, encoding='utf8') as f:
        sent = []
        mwes = {}
        for line in f:
            if line.strip().startswith('#'):
                continue
            elif line.strip():
                split = line.strip().split('\t')
                tok = {'token': split[1], 'lemma': split[2], 'token_id': split[0], 'pos': split[3], 'dep': split[7],
                       'mwe': split[10]}
                sent.append(tok)
                if not tok['mwe'] == '*':
                    # operate on mwe tokens
                    ms = tok['mwe'].split(';')
                    ms = [m.split(':')[0] for m in ms]
                    for m in ms:
                        if not m in mwes:
                            mwes[m]=[]
                        mwes[m].append(tok)
            if not line.strip() and sent:
                sents.append(sent)
                mwe_list.append(mwes.values())
                mwes = {}
                sent = []
    return sents, mwe_list


gold_file = 'results/test.cupt'
pred_file = 'results/predicted_EN_STREUSLE_model_Att_Based_glove_system.cupt'

_, gold_mwes = get_mwes(gold_file)
_, pred_mwes = get_mwes(pred_file)

exact_correct = 0
fuzzy_correct = 0
gold_total = 0
pred_total = 0

for gold_mwes_sent, pred_mwes_sent in zip(gold_mwes, pred_mwes):
    gold_total += len(gold_mwes_sent)
    pred_total += len(pred_mwes_sent)
    # for gold_mwe in gold_mwes_sent:
        # print('gold','-'.join(tok['token'] for tok in gold_mwe))
    # for pred_mwe in pred_mwes_sent:
        # print('pred','-'.join(tok['token'] for tok in pred_mwe))
    for gold_mwe in gold_mwes_sent:
        for pred_mwe in pred_mwes_sent:
            exact_match = ([tok['token_id'] for tok in gold_mwe]==[tok['token_id'] for tok in pred_mwe])
            if exact_match:
                exact_correct +=1
            a = set(tok['token_id'] for tok in gold_mwe)
            b = set(tok['token_id'] for tok in pred_mwe)
            fuzzy_correct += len(a & b) / len(a | b)
            fuzzy_match = len(a & b) / len(a | b)
            if not exact_match and fuzzy_match>0:
                print('error', 'gold','-'.join(tok['token'] for tok in gold_mwe))
                print('error', 'pred','-'.join(tok['token'] for tok in pred_mwe))

print('gold mwes:',gold_total)
print('pred mwes:',pred_total)
print('exact prec:',exact_correct/pred_total)
print('exact rec:',exact_correct/gold_total)
exact_rec = exact_correct/gold_total
exact_prec = exact_correct/pred_total
print('exact F1:',2*exact_rec*exact_prec/(exact_rec+exact_prec))
print('fuzzy prec:',fuzzy_correct/pred_total)
print('fuzzy rec:',fuzzy_correct/gold_total)
fuzzy_rec = fuzzy_correct/gold_total
fuzzy_prec = fuzzy_correct/pred_total
print('fuzzy F1:',2*fuzzy_rec*fuzzy_prec/(fuzzy_rec+fuzzy_prec))