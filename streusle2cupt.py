from collections import Counter


streusle_file = 'data/streusle.conllulex'
cupt_file = 'data/streusle.cupt'

mwe_idx = 0
with open(streusle_file, encoding='utf8') as f:
    with open(cupt_file, 'w', encoding='utf8') as f2:
        for line in f:
            if not line.strip().startswith('#') and line.strip():
                split = line.strip().split('\t')
                tok = {'token': split[1], 'lemma': split[2], 'token_id': split[0], 'pos': split[3], 'dep': split[7],
                       'mwe1': split[11], 'mwe2': split[18]}
                # fix mwe label
                if tok['mwe2'].startswith('B'):
                    mwe_idx+=1
                mwe = '*'
                mwe = str(mwe_idx) if tok['mwe2'].startswith('I') else str(mwe_idx)+':'+tok['mwe1'].replace('V.','') if tok['mwe2'].startswith('B') else '*'

                line = '\t'.join([tok['token_id'], tok['token'], tok['lemma'],
                                  tok['pos'], '_', '_','_',tok['dep'],'_','_',mwe])+'\n'
            if not line.strip():
                mwe_idx = 0

            f2.write(line)


