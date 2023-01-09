import math
import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer
from gensim.models.ldamodel import LdaModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from gensim import corpora
from sklearn import svm
import string
import scipy


def MakeEval():
    system_results = pd.read_csv("system_results.csv")
    qrels = pd.read_csv("qrels.csv")
    ir_eval = pd.DataFrame()
    for sys_num in range(1,7):
        # read number of system_results for each query
        qr_dic = dict(system_results[system_results["system_number"] == sys_num].query_number.value_counts())
        # read system_results of each system
        sys_results = system_results[system_results["system_number"] == sys_num]
        ireval = pd.DataFrame()
        ireval["query_number"] = np.arange(1,11)
        ireval["system_number"] = sys_num
        P_10 = []
        R_50 = []
        r_precision = []
        AP = []
        nDCG_10 = []
        nDCG_20 = []
        index = 0
        # loop each query
        for qr_id in range(1,11):
            qrel = list(qrels[qrels["query_id"] == qr_id]["doc_id"])
            #P@10
            result1 = list(sys_results[index: index + 10]["doc_number"])
            P_10.append(len(set(result1) & set(qrel)) / 10)
            #R@50
            result2 = list(sys_results[index: index + 50]["doc_number"])
            R_50.append(len(set(result2) & set(qrel)) / len(qrel))
            #r_precision
            result3 = list(sys_results[index: index + len(qrel)]["doc_number"])
            r_precision.append(len(set(result3) & set(qrel)) / len(qrel))
            #AP
            result4 = sys_results[index: index + qr_dic[qr_id]] 
            ap_list = []
            for doc in qrel:
                if doc in list(result4["doc_number"]):
                    ap_list.append(int(result4[result4["doc_number"] == doc]["rank_of_doc"]))
            count = 1
            total = 0
            for idx in sorted(ap_list):
                total = total+ (count / idx)
                count += 1
            AP.append(total/len(qrel))
            #nDCG@10
            result5 = sys_results[index: index + 10]
            #DCG@10
            DCG10 = 0
            for doc in qrel:
                if doc in list(result5["doc_number"]):
                    G = int(qrels.loc[(qrels["query_id"] == qr_id) & (qrels["doc_id"] == doc)]["relevance"])
                    if int(result5[result5["doc_number"] == doc]["rank_of_doc"]) == 1:
                        DCG10 += G
                    else:
                        DCG10 += G / math.log2(int(result5[result5["doc_number"] == doc]["rank_of_doc"]))
            rel_list = sorted(list(qrels[qrels["query_id"] == qr_id]["relevance"]),reverse=True)
            # calculate iDCG@10
            iDCG_10 = rel_list[0]
            if len(rel_list) <= 10:
                for i in range(1,len(rel_list)):
                    iDCG_10 += rel_list[i] / math.log2(i+1)
            else:
                for i in range(1,10):
                    iDCG_10 += rel_list[i] / math.log2(i+1)
            nDCG_10.append(DCG10 / iDCG_10)
            # calculate nDCG@20
            result6 = sys_results[index: index + 20]
            # calculate DCG@20
            DCG20 = 0
            for doc in qrel:
                if doc in list(result6["doc_number"]):
                    G = int(qrels.loc[(qrels["query_id"] == qr_id) & (qrels["doc_id"] == doc)]["relevance"])
                    if int(result6[result6["doc_number"] == doc]["rank_of_doc"]) == 1:
                        DCG20 += G
                    else:
                        DCG20 += G / math.log2(int(result6[result6["doc_number"] == doc]["rank_of_doc"]))
            # calculate iDCG@10
            rel_list = sorted(list(qrels[qrels["query_id"] == qr_id]["relevance"]),reverse=True)
            iDCG_20 = rel_list[0]
            if len(rel_list) <= 20:
                for i in range(1,len(rel_list)):
                    iDCG_20 += rel_list[i] / math.log2(i+1)
            else:
                for i in range(1,20):
                    iDCG_20 += rel_list[i] / math.log2(i+1)
            nDCG_20.append(DCG20/iDCG_20)
            index += qr_dic[qr_id]
        ireval["P@10"] = np.array(P_10)
        ireval["R@50"] = np.array(R_50)
        ireval["r-precision"] = np.array(r_precision)
        ireval["AP"] = np.array(AP)
        ireval["nDCG@10"] = np.array(nDCG_10)
        ireval["nDCG@20"] = np.array(nDCG_20)
        ireval.index = ireval.index + 1
        ireval.loc["mean"] = ireval.mean()
        ir_eval = pd.concat([ir_eval,ireval], axis=0)
    f1 = "ir_eval.csv"
    with open(f1, "a+") as file:
        file.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20" + "\n")
    for index, row in ir_eval.iterrows():
        with open(f1, "a+") as file:
            file.write(str(int(row["system_number"])) + "," + str(index) + "," + "{:.3f}".format(row["P@10"]) + "," "{:.3f}".format(row["R@50"]) + "," + "{:.3f}".format(row["r-precision"]) +  "," "{:3f}".format(row["AP"]) + "," + "{:.3f}".format(row["nDCG@10"]) + "," + "{:.3f}".format(row["nDCG@20"]) + "\n")
def tokenization(text):
    """seperate sentence into lower case word tokens """
    regular = r'\w+'
    return re.findall(regular, text.lower())

def removestopwords(text):
    """load the stopwods file and delete the word if it exists in the file"""
    with open('englishST.txt') as f:
        stpwds = f.read().split('\n')
    return [w for w in text if not w in stpwds]

def stemming(text):
    """normolize words"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

def preprocessing(data):
    lines=data.split('\n')
    corQuran=[]
    corNT=[]
    corOT=[]
    dicQuran={}
    dicNT={}
    dicOT={}
    for line in lines:
        corpus_name,text=line.split("\t")
        terms=stemming(removestopwords(tokenization(text)))
        if corpus_name=="OT":
            corOT.append(terms)
            for term in terms:
                if term not in dicOT:
                    dicOT[term]=1
                else:
                    dicOT[term]+=1
        elif corpus_name=="NT":
            corNT.append(terms)
            for term in terms:
                if term not in dicNT:
                    dicNT[term]=1
                else:
                    dicNT[term]+=1
        elif corpus_name=="Quran":
            corQuran.append(terms)
            for term in terms:
                if term not in dicQuran:
                    dicQuran[term]=1
                else:
                    dicQuran[term]+=1
    return corQuran,corNT,corOT,dicQuran,dicNT,dicOT

def Getscore(total_dic, corpus_dic, corQuran, corNT, corOT,dicQuran,dicNT,dicOT):
    dicmi = {}
    dicchi = {}
    if corpus_dic == dicQuran:
        other_dic = {**dicOT,**dicNT}
        total1 = len(corQuran)
        total2 = len(corNT) + len(corOT)
    elif corpus_dic == dicOT:
        other_dic = {**dicQuran,**dicNT}
        total1 = len(corOT)
        total2 = len(corNT) + len(corQuran)
    else:
        other_dic = {**dicQuran,**dicOT}
        total1 = len(corNT)
        total2 = len(corQuran) + len(corOT)
    # total number of terms
    N = total1 + total2
    for term in total_dic.keys():
        if term in corpus_dic.keys():
            N11 = corpus_dic[term]
            N01 = total1 - N11
            if term in other_dic.keys():
                N10 = other_dic[term]
                N00 = total2 - N10
                resultmi = N11 / N * math.log2(float(N * N11) / float((N10 + N11) * (N01 + N11))) + N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11))) + N10 / N * math.log2(float(N * N10) / float((N10 + N11) * (N00 + N10))) + N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))
            else:
                N10 = 0
                N00 = total2
                resultmi = N11 / N * math.log2(float(N * N11) / float((N10 + N11) * (N01 + N11))) + N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11)))+ N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))
        else:
            N11 = 0
            N01 = total1
            N10 = other_dic[term]
            N00 = total2 - N10
            resultmi = N01/N*math.log2(float(N*N01) / float((N00+N01)*(N01+N11))) + N10/N*math.log2(float(N*N10) / float((N10+N11)*(N00+N10)))+ N00/N*math.log2(float(N*N00) / float((N00+N01)*(N00+N10)))

        dicmi[term] = resultmi
        resultchi = ((N11 + N10 + N01 + N00) * math.pow(N11 * N00 - N10 * N01, 2)) / ((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))
        dicchi[term] = resultchi
    return dicmi, dicchi

def TextAna(corQuran,corNT,corOT,dicQuran,dicNT,dicOT):
    data=open('train_and_dev.tsv').read()
    corQuran,corNT,corOT,dicQuran,dicNT,dicOT=preprocessing(data)
    dicAll={**dicQuran,**dicNT,**dicOT}
    miQuran,chiQuran=Getscore(dicAll,dicQuran,corQuran,corNT,corOT,dicQuran,dicNT,dicOT)
    miOT,chiOT=Getscore(dicAll,dicOT,corQuran,corNT,corOT,dicQuran,dicNT,dicOT)
    miNT,chiNT=Getscore(dicAll,dicNT,corQuran,corNT,corOT,dicQuran,dicNT,dicOT)
    print(sorted(miQuran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(chiQuran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(miOT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(chiOT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(miNT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(chiNT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])

def runLDA(corQuran, corNT, corOT):
    total_corpus = corQuran + corNT + corOT
    dic = corpora.Dictionary(total_corpus)
    corpus = [dic.doc2bow(text) for text in total_corpus]
    lda = LdaModel(corpus, num_topics=20, id2word=dic, random_state=1)
    dic1 = corpora.Dictionary(corQuran)
    corpus1 = [dic1.doc2bow(text) for text in corQuran]
    topics_Quran = lda.get_document_topics(corpus1)
    topic_dicQuran = {}
    for doc in topics_Quran:
        for topic in doc:
            if topic[0] not in topic_dicQuran.keys():
                topic_dicQuran[topic[0]] = topic[1]
            else:
                topic_dicQuran[topic[0]] += topic[1]
    dic2 = corpora.Dictionary(corOT)
    corpus2 = [dic2.doc2bow(text) for text in corOT]
    topics_OT = lda.get_document_topics(corpus2)
    topic_dicOT = {}
    for doc in topics_OT:
        for topic in doc:
            if topic[0] not in topic_dicOT.keys():
                topic_dicOT[topic[0]] = topic[1]
            else:
                topic_dicOT[topic[0]] += topic[1]
    dic3 = corpora.Dictionary(corNT)
    corpus3 = [dic3.doc2bow(text) for text in corNT]
    topics_NT = lda.get_document_topics(corpus3)
    topic_dicNT = {}
    for doc in topics_NT:
        for topic in doc:
            if topic[0] not in topic_dicNT.keys():
                topic_dicNT[topic[0]] = topic[1]
            else:
                topic_dicNT[topic[0]] += topic[1]
    for k, v in topic_dicQuran.items():
        topic_dicQuran[k] = v / len(corQuran)
    for k, v in topic_dicOT.items():
        topic_dicOT[k] = v / len(corOT)
    for k, v in topic_dicNT.items():
        topic_dicNT[k] = v / len(corNT)
    return lda, topic_dicQuran, topic_dicNT, topic_dicOT

def printoutLDA(lda, topic_dicQuran, topic_dicNT, topic_dicOT):
    topic_ranked_NT = sorted(topic_dicNT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
    topic_ranked_OT = sorted(topic_dicOT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
    topic_ranked_Quran = sorted(topic_dicQuran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
    for topic in topic_ranked_NT:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
        print("\n")
    for topic in topic_ranked_OT:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
        print("\n")
    for topic in topic_ranked_Quran:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
        print("\n")

def preprocessingTrain(traindata):
    chars = re.compile(f'[{string.punctuation}]')
    documents = []
    categories = []
    vocab = set([])
    train1=' '.join(traindata)
    train2=re.sub(r'http\S+', "", train1)
    lines=train2.split("\n")
    for line in lines:
        line= line.strip()
        id, category, text = line.split("\t")
        words = chars.sub("", text).lower().split()
        for word in words:
            vocab.add(word)
        documents.append(words)
        categories.append(category)
    return documents, categories, vocab

def preprocessingTrainNostp(traindata):
    chars = re.compile(f'[{string.punctuation}]')
    documents = []
    categories = []
    vocab = set([])
    train1=' '.join(traindata)
    train2=re.sub(r'http\S+', "", train1)
    lines=train2.split("\n")
    for line in lines:
        line= line.strip()
        id, category, text = line.split("\t")
        words = chars.sub("", text).lower().split()
        words=removestopwords(words)
        for word in words:
            vocab.add(word)
        documents.append(words)
        categories.append(category)
    return documents, categories, vocab

def bowmatrix(preprocessed_data, word2id):
    matrix_size = (len(preprocessed_data), len(word2id) + 1)
    # out of vocabulary index
    index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)
    for doc_id, doc in enumerate(preprocessed_data):
        for word in doc:
            X[doc_id, word2id.get(word, index)] += 1
    return X

def accuracy_check(true,pred,cat,cat2id):
    TP = np.sum(np.logical_and(np.equal(true, cat2id[cat]), np.equal(pred, cat2id[cat])))
    FP = np.sum(np.logical_and(np.not_equal(true, cat2id[cat]), np.equal(pred, cat2id[cat])))
    TN = np.sum(np.logical_and(np.not_equal(true, cat2id[cat]), np.not_equal(pred, cat2id[cat])))
    FN = np.sum(np.logical_and(np.equal(true, cat2id[cat]), np.not_equal(pred, cat2id[cat])))
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = 2 * p * r / (p + r)
    return p, r, f1

def get_score(y_true, y_pred, cat2id):
    p_ne, r_ne, f_ne = accuracy_check(y_true, y_pred, "neutral", cat2id)
    p_po, r_po, f_po = accuracy_check(y_true, y_pred,"positive", cat2id)
    p_neg, r_neg, f_neg = accuracy_check(y_true, y_pred,"negative", cat2id)
    p_macro = precision_score(y_true, y_pred, average="macro")
    r_macro = recall_score(y_true, y_pred, average="macro")
    f_macro = f1_score(y_true, y_pred, average="macro")
    print(f_macro)
    accuracy_list = [str(p_po), str(r_po), str(f_po), str(p_neg), str(r_neg),str(f_neg), str(p_ne), str(r_ne), str(f_ne),str(p_macro), str(r_macro), str(f_macro)]
    return accuracy_list

if __name__ == "__main__":
    data=open('train_and_dev.tsv').read()
    corQuran,corNT,corOT,dicQuran,dicNT,dicOT=preprocessing(data)
    # operate evaluation
    MakeEval()
    #text analysis
    TextAna(corQuran,corNT,corOT,dicQuran,dicNT,dicOT)
    # run LDA model on three corpora
    lda, topic_dicQuran, topic_dicNT, topic_dicOT = runLDA(corQuran,corNT,corOT)
    # rank the topics for each corpus
    printoutLDA(lda, topic_dicQuran, topic_dicNT, topic_dicOT)

    #baseline
    traindata=open("train.txt").readlines()[1:]
    preprocessed_data, categories, vocab=preprocessingTrain(traindata)
    preprocessed_training_data, preprocessed_dev_data,training_categories, dev_categories = train_test_split(preprocessed_data, categories,test_size=0.2,shuffle=True)
    word2id = {}
    for wordid, word in enumerate(vocab):
        word2id[word] = wordid
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    trainx = bowmatrix(preprocessed_training_data, word2id)
    trainy = [cat2id[cat] for cat in training_categories]
    model = svm.SVC(C=1000, random_state=0)
    model.fit(trainx, trainy)
    devx = bowmatrix(preprocessed_dev_data, word2id)
    devy = [cat2id[cat] for cat in dev_categories]
    trainy_pred=model.predict(trainx)
    devy_pred = model.predict(devx)
    index=0
    for row_index, (input, prediction, label) in enumerate(zip (devx, devy_pred, devy)):
        if index == 3:
            break
        if prediction != label:
            #{'positive': 0, 'negative': 1, 'neutral': 2}
            print('Row', preprocessed_dev_data[row_index], 'has been classified as', prediction, 'and should be', label)
            index+=1
    testing_data = open("test.txt").readlines()[1:]
    preprocessed_test_data, test_categories, test_vocab = preprocessingTrain(testing_data)
    testx = bowmatrix(preprocessed_test_data, word2id)
    testy = [cat2id[cat] for cat in test_categories]
    y_test_predictions = model.predict(testx)
    print("baseline:")
    base_train = get_score(trainy, trainy_pred, cat2id)
    base_dev = get_score(devy, devy_pred, cat2id)
    base_test = get_score(testy, y_test_predictions, cat2id)

    ################################################################################################
    #no stopwords
    traindata=open("train.txt").readlines()[1:]
    preprocessed_data, categories, vocab=preprocessingTrainNostp(traindata)
    preprocessed_training_data, preprocessed_dev_data,training_categories, dev_categories = train_test_split(preprocessed_data, categories,test_size=0.2,shuffle=True)
    word2id = {}
    for wordid, word in enumerate(vocab):
        word2id[word] = wordid
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    trainx = bowmatrix(preprocessed_training_data, word2id)
    trainy = [cat2id[cat] for cat in training_categories]
    model = svm.SVC(C=1000, random_state=0)
    model.fit(trainx, trainy)
    devx = bowmatrix(preprocessed_dev_data, word2id)
    devy = [cat2id[cat] for cat in dev_categories]
    trainy_pred=model.predict(trainx)
    devy_pred = model.predict(devx)
    index=0
    for row_index, (input, prediction, label) in enumerate(zip (devx, devy_pred, devy)):
        if index == 3:
            break
        if prediction != label:
            #{'positive': 0, 'negative': 1, 'neutral': 2}
            print('Row', preprocessed_dev_data[row_index], 'has been classified as', prediction, 'and should be', label)
            index+=1
    testing_data = open("test.txt").readlines()[1:]
    preprocessed_test_data, test_categories, test_vocab = preprocessingTrain(testing_data)
    testx = bowmatrix(preprocessed_test_data, word2id)
    testy = [cat2id[cat] for cat in test_categories]
    y_test_predictions = model.predict(testx)
    print("Nostopwords:")
    nostp_train = get_score(trainy, trainy_pred, cat2id)
    nostp_dev = get_score(devy, devy_pred, cat2id)
    nostp_test = get_score(testy, y_test_predictions, cat2id)

    ################################################################################################
    #C=10
    traindata=open("train.txt").readlines()[1:]
    preprocessed_data, categories, vocab=preprocessingTrain(traindata)
    preprocessed_training_data, preprocessed_dev_data,training_categories, dev_categories = train_test_split(preprocessed_data, categories,test_size=0.2,shuffle=True)
    word2id = {}
    for wordid, word in enumerate(vocab):
        word2id[word] = wordid
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    trainx = bowmatrix(preprocessed_training_data, word2id)
    trainy = [cat2id[cat] for cat in training_categories]
    model = svm.SVC(C=10, random_state=0)
    model.fit(trainx, trainy)
    devx = bowmatrix(preprocessed_dev_data, word2id)
    devy = [cat2id[cat] for cat in dev_categories]
    trainy_pred=model.predict(trainx)
    devy_pred = model.predict(devx)
    index=0
    for row_index, (input, prediction, label) in enumerate(zip (devx, devy_pred, devy)):
        if index == 3:
            break
        if prediction != label:
            #{'positive': 0, 'negative': 1, 'neutral': 2}
            print('Row', preprocessed_dev_data[row_index], 'has been classified as', prediction, 'and should be', label)
            index+=1
    testing_data = open("test.txt").readlines()[1:]
    preprocessed_test_data, test_categories, test_vocab = preprocessingTrain(testing_data)
    testx = bowmatrix(preprocessed_test_data, word2id)
    testy = [cat2id[cat] for cat in test_categories]
    y_test_predictions = model.predict(testx)
    print("C10:")
    c10_train = get_score(trainy, trainy_pred, cat2id)
    c10_dev = get_score(devy, devy_pred, cat2id)
    c10_test = get_score(testy, y_test_predictions, cat2id)
    ################################################################################################
    #C=6000
    traindata=open("train.txt").readlines()[1:]
    preprocessed_data, categories, vocab=preprocessingTrain(traindata)
    preprocessed_training_data, preprocessed_dev_data,training_categories, dev_categories = train_test_split(preprocessed_data, categories,test_size=0.2,shuffle=True)
    word2id = {}
    for wordid, word in enumerate(vocab):
        word2id[word] = wordid
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    trainx = bowmatrix(preprocessed_training_data, word2id)
    trainy = [cat2id[cat] for cat in training_categories]
    model = svm.SVC(C=6000, random_state=0)
    model.fit(trainx, trainy)
    devx = bowmatrix(preprocessed_dev_data, word2id)
    devy = [cat2id[cat] for cat in dev_categories]
    trainy_pred=model.predict(trainx)
    devy_pred = model.predict(devx)
    index=0
    for row_index, (input, prediction, label) in enumerate(zip (devx, devy_pred, devy)):
        if index == 3:
            break
        if prediction != label:
            #{'positive': 0, 'negative': 1, 'neutral': 2}
            print('Row', preprocessed_dev_data[row_index], 'has been classified as', prediction, 'and should be', label)
            index+=1
    testing_data = open("test.txt").readlines()[1:]
    preprocessed_test_data, test_categories, test_vocab = preprocessingTrain(testing_data)
    testx = bowmatrix(preprocessed_test_data, word2id)
    testy = [cat2id[cat] for cat in test_categories]
    y_test_predictions = model.predict(testx)
    print("c6000:")
    c6000_train = get_score(trainy, trainy_pred, cat2id)
    c6000_dev = get_score(devy, devy_pred, cat2id)
    c6000_test = get_score(testy, y_test_predictions, cat2id)
    ################################################################################################
    #LogisticRegression
    traindata=open("train.txt").readlines()[1:]
    preprocessed_data, categories, vocab=preprocessingTrain(traindata)
    preprocessed_training_data, preprocessed_dev_data,training_categories, dev_categories = train_test_split(preprocessed_data, categories,test_size=0.2,shuffle=True)
    word2id = {}
    for wordid, word in enumerate(vocab):
        word2id[word] = wordid
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    trainx = bowmatrix(preprocessed_training_data, word2id)
    trainy = [cat2id[cat] for cat in training_categories]
    model = LogisticRegression(random_state=0,max_iter=3000)
    model.fit(trainx, trainy)
    devx = bowmatrix(preprocessed_dev_data, word2id)
    devy = [cat2id[cat] for cat in dev_categories]
    trainy_pred=model.predict(trainx)
    devy_pred = model.predict(devx)
    index=0
    for row_index, (input, prediction, label) in enumerate(zip (devx, devy_pred, devy)):
        if index == 3:
            break
        if prediction != label:
            #{'positive': 0, 'negative': 1, 'neutral': 2}
            print('Row', preprocessed_dev_data[row_index], 'has been classified as', prediction, 'and should be', label)
            index+=1
    testing_data = open("test.txt").readlines()[1:]
    preprocessed_test_data, test_categories, test_vocab = preprocessingTrain(testing_data)
    testx = bowmatrix(preprocessed_test_data, word2id)
    testy = [cat2id[cat] for cat in test_categories]
    y_test_predictions = model.predict(testx)
    print("logistic:")
    logistic_train = get_score(trainy, trainy_pred, cat2id)
    logistic_dev = get_score(devy, devy_pred, cat2id)
    logistic_test = get_score(testy, y_test_predictions, cat2id)

    with open("classification.csv", "a+") as file:
        file.write("system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro" + "\n")
        file.write("baseline,train," + ",".join(base_train) + "\n")
        file.write("baseline,dev," + ",".join(base_dev) + "\n")
        file.write("baseline,test," + ",".join(base_test) + "\n")
        file.write("improved,train," + ",".join(logistic_train) + "\n")
        file.write("improved,dev," + ",".join(logistic_dev) + "\n")
        file.write("improved,test," + ",".join(logistic_test) + "\n")

