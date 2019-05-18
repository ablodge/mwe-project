from collections import Counter

sents = []
labels = Counter()
labels_by_type = Counter()
mwe_types = {}
mwe_type_counts = Counter()
mwe_heads = Counter()
mwe_pos = Counter()
mwe_dep = Counter()
head_dep_per_token = Counter()
verb_dep_per_token = Counter()
non_head_dep_per_token = Counter()
common_example= {}


with open('data/streusle.cupt', encoding='utf8') as f:
	with open('stats.txt', 'w+', encoding='utf8') as fout:
		sent = []
		for line in f:
			if line.strip().startswith('#'):
				continue
			elif line.strip():
				split = line.strip().split('\t')
				tok = {'token':split[1], 'lemma':split[2], 'token_id':split[0], 'pos':split[3], 'dep':split[7], 'mwe':split[10]}
				sent.append(tok)
				if not tok['mwe']=='*':
					# operate on mwe tokens
					fout.write(str(tok)+'\n')
					mwes = tok['mwe'].split(';')
					mwes = [m.split(':')[1] for m in mwes if ':' in m]
					for m in mwes:
						labels[m] += 1
				if tok['pos'] == 'VERB':
					verb_dep_per_token[tok['dep']] += 1
			if not line.strip() and sent:
				sents.append(sent)
				fout.write('\n')
				mwes = {}

				for tok in sent:
					if tok['mwe']=='*':
						continue
					for m in tok['mwe'].split(';'):
						idx = m.split(':')[0]
						if idx not in mwes:
							mwes[idx] = {'lemma':[], 'pos':[], 'dep':[], 'mwe':'', 'head':'', 'dep_head':[]}
						mwes[idx]['lemma'].append(tok['lemma'])
						mwes[idx]['pos'].append(tok['pos'])
						mwes[idx]['dep'].append(tok['dep'])
						mwes[idx]['dep_head'].append(tok['dep'])
						if ':' in m:
							mwes[idx]['mwe'] = m.split(':')[1]
							mwes[idx]['head'] = tok['lemma']
							head_dep_per_token[tok['dep']] += 1
							mwes[idx]['dep_head'][-1] = '*HEAD*'
				for idx, mwe in mwes.items():
					# operate on mwe types
					mwe_types['-'.join(mwe['lemma'])] = mwe
					mwe_type_counts['-'.join(mwe['lemma'])] +=1
					non_head_dep_per_token[mwe['mwe']+' '+' '.join(mwe['dep_head'])] += 1
				sent = []

for lemma, mwe in mwe_types.items():
	mwe_heads[mwe['head']] += 1
	mwe_pos[' '.join(mwe['pos'])] += 1
	mwe_dep[' '.join(mwe['dep'])] += 1
	labels_by_type[mwe['mwe']] +=1
	if mwe['mwe'] not in common_example:
		common_example[mwe['mwe']] = []
	common_example[mwe['mwe']].append('-'.join(mwe['lemma']))
print(labels)
print(labels_by_type)
print(mwe_type_counts)
print(common_example)
print(mwe_heads)
print(mwe_pos)
print(mwe_dep)
print(head_dep_per_token)
print(verb_dep_per_token)
print(non_head_dep_per_token)