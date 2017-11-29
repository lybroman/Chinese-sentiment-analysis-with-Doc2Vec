__author__ = "lybroman@hotmail.com"

import gensim
import smart_open
from gensim.models import Doc2Vec
from sklearn.externals import joblib
from collections import defaultdict

res0 =defaultdict(int)
res1 =defaultdict(int)
num_1 = 2910
num_5 = 5882

model = Doc2Vec.load('./my_db.d2v')

# have fun with the amazing most_similar method
# the definition of similar please refer to word embedding
print(model.most_similar(positive=['交通']))
print(model.most_similar(positive=['风景']))

rf = joblib.load('classifier.model')
sources_train = {'1_train.txt':'ONE', '5_train.txt':'FIVE'}

def read_corpus(source_set):
    ct = 0
    for source_file, prefix in source_set.items():
        with smart_open.smart_open(source_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                # split with space to isolate each word
                # the words list are tagged with a label as its identity
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.to_unicode(line).split(), [prefix + '_%s' % i])

train_corpus = list(read_corpus(sources_train))

# since 1_test is empty..., use 2_test
with open('./2_test.txt', 'r', encoding='utf-8') as f, open('./2_sim.txt', 'w', encoding='utf-8') as ff:
    while True:
        line = f.readline()
        if not line: break
        line_ls = line.replace('\n','').split(' ')
        line_vec = model.infer_vector(line_ls, alpha=0.95, steps=50)

        # find the most similar docs/sentence
        sims = model.docvecs.most_similar([line_vec], topn=1)
        ff.write(line)
        index = int(sims[0][0].split('_')[1]) + (num_1 if sims[0][0].split('_')[0] == 'FIVE' else 0)
        ff.write(' '.join(train_corpus[index].words) + '\n\n')

        res0[int(rf.predict([line_vec]))] += 1

print('Accuracy for negative comments: %s' % (res0[0] / (res0[0] +res0[1]), ))


with open('./5_test.txt', 'r', encoding='utf-8') as f, open('./5_sim.txt', 'w', encoding='utf-8') as ff:
    while True:
        line  = f.readline()
        if not line: break
        line_ls = line.split(' ')
        # pls fine-tune your hyper-parameters: alpha & steps
        line_vec = model.infer_vector(line_ls, alpha=0.05, steps=50)

        # find the most similar docs/sentence
        sims = model.docvecs.most_similar([line_vec], topn=1)
        ff.write(line)
        index = int(sims[0][0].split('_')[1]) + (num_1 if sims[0][0].split('_')[0] == 'FIVE' else 0)
        ff.write(' '.join(train_corpus[index].words) + '\n\n')

        res1[int(rf.predict([line_vec]))] += 1

print('Accuracy for positive comments: %s' % (res1[1] / (res1[0] +res1[1]), ))


