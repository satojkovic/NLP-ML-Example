#-*- encoding: utf-8 -*-

"""
Reference: https://github.com/Scripted/NLP-Tutorial
"""

#! /usr/bin/env python


from math import sqrt
import gensim
import gensim.parsing
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import os
import re


def vec2dense(vec, num_terms):

    '''Convert from sparse gensim format to dense list of numbers'''
    return list(gensim.matutils.corpus2dense([vec], num_terms=num_terms).T[0])

if __name__ == '__main__':

    # コーパスのロード, 改行の除去, 小文字に統一
    docs = {}
    corpus_dir = 'corpus'
    for filename in os.listdir(corpus_dir):
        path = os.path.join(corpus_dir, filename)
        doc = open(path).read().strip().lower()
        docs[filename] = doc

    names = docs.keys()

    # ストップワードの除去, ステミング
    print "\n---Corpus with Stopwords Removed---"

    preprocessed_docs = {}
    for name in names:
        preprocessed = gensim.parsing.preprocess_string(docs[name])
        preprocessed_docs[name] = preprocessed
        print name, ":", preprocessed

    # 辞書を作成
    # 低頻度と高頻度のワードは除く
    dct = gensim.corpora.Dictionary(preprocessed_docs.values())
    unfiltered = dct.token2id.keys()
    dct.filter_extremes(no_below=3, no_above=0.6)
    filtered = dct.token2id.keys()
    filtered_out = set(unfiltered) - set(filtered)

    print "\nThe following super common/rare words are filtered out..."
    print list(filtered_out), '\n'

    print "Vocabulary after filtering..."
    print dct.token2id.keys(), "(%d words)" % len(dct.token2id.keys()), '\n'

    print "Save Dictionary..."
    dct_txt = "id2word.txt"
    dct.save_as_text(dct_txt)
    print "  saved to %s\n" % dct_txt

    # Bag of Words Vectorsの作成
    # ストップワード除去とステミング処理した文書に対して, 辞書のワードをカウント
    print "---Bag of Words Corpus---"

    bow_docs = {}
    bow_docs_all_zeros = {}
    for name in names:

        sparse = dct.doc2bow(preprocessed_docs[name])
        bow_docs[name] = sparse
        dense = vec2dense(sparse, num_terms=len(dct))
        print name, ":", dense
        bow_docs_all_zeros[name] = all(d == 0 for d in dense)

    print "\nall zeros...\n", [name for name in bow_docs_all_zeros
                               if bow_docs_all_zeros[name]]

    # LSIにより次元削減
    print "\n---LSI Model---"

    lsi_docs = {}
    num_topics = 2
    lsi_model = gensim.models.LsiModel(bow_docs.values(),
                                       id2word=dct.load_from_text('id2word.txt'),
                                       num_topics=num_topics)

    for name in names:

        vec = bow_docs[name]
        sparse = lsi_model[vec]
        dense = vec2dense(sparse, num_topics)
        lsi_docs[name] = sparse
        print name, ":", dense

    print "\nTopics"
    print lsi_model.print_topics()

    # 次元削減後のベクトルを正規化(ベクトルの方向が重要)
    print "\n---Unit Vectorization---"

    unit_vecs = {}
    for name in names:

        vec = vec2dense(lsi_docs[name], num_topics)
        norm = sqrt(sum(num ** 2 for num in vec))
        unit_vec = [num / norm for num in vec]
        unit_vecs[name] = unit_vec
        print name, ":", unit_vec

    # 2クラス分類
    print "\n---Classification---\n"

    all_data = [unit_vecs[name] for name in names if re.match("ipad", name)]
    all_data.extend([unit_vecs[name] for name in names
                     if re.match("sochi", name)])

    all_labels = [1 for name in names if re.match("ipad", name)]
    all_labels.extend([2 for name in names if re.match("sochi", name)])

    train_data, test_data, train_label, test_label = train_test_split(all_data,
                                                                      all_labels)

    # SVMの学習
    classifier = SVC()
    classifier.fit(train_data, train_label)

    # 予測
    predict_label = classifier.predict(test_data)

    target_names = ["class 'ipad'", "class 'sochi'"]
    print classification_report(test_label, predict_label,
                                target_names=target_names)
