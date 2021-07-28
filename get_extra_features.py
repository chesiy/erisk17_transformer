from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,PCA
from sklearn.linear_model import LogisticRegression
import xml.dom.minidom
import re
import nltk
import os
import numpy as np
import liwc
from collections import Counter

antidepressants= ['abilify','aripiprazole', 'adapin', 'doxepin','anafranil', 'clomipramine','Aplenzin', 'bupropion','asendin', 'amoxapine',
'aventyl','nortriptyline','brexipipzole','rexulti','celexa','citalopram','cymbalta','duloxetine','desyrel','trazodone','effexor','venlafaxine','emsam' ,'selegiline',
'esketamine','spravato','etrafon','elavil','amitriptyline','endep','fetzima','levomilnacipran','khedezla','desvenlafaxine','latuda','lurasidone',
'lamictal', 'lamotrigine','lexapro', 'escitalopram','limbitrol','chlordiazepoxide','marplan','isocarboxazid',
'nardil','phenelzine','norpramin','desipramine','oleptro','trazodone','pamelor','nortriptyline','parnate','tranylcypromine','paxil','paroxetine',
'pexeva','paroxetine','prozac','fluoxetine','pristiq','desvenlafaxine','remeron','mirtazapine','sarafem','fluoxetine','seroquel','quetiapine','serzone','nefazodone','sinequan','doxepin',
'surmontil','trimipramine','symbyax','tofranil', 'imipramine','triavil','trintelllix','vortioxetine','viibryd','vilazodone','vivactil','protriptyline','wellbutrin','bupropion',
'zoloft','sertraline','zyprexa','olanzapine']


LIWC_parse, category_names = liwc.load_token_parser('LIWC2015_English.dic')

def get_LIWC(tokened_text):
    liwc_counts = Counter(category for token in tokened_text for category in LIWC_parse(token))
    total_words = sum(liwc_counts.values())         # the total number of lexicon words found
    return liwc_counts,total_words

def process_raw(res):
    res.lower()
    res = re.sub('\n', ' ', res)
    res = res.strip()
    res = res.split()
    res = ' '.join(res)
    # print('ff',res)
    res = nltk.word_tokenize(res)
    words = []
    # print(res)
    for word in res:
        if word.isalpha():
            words.append(word)
    # print('f',words)
    return words


def get_input_data(file, path):
    post_num = 0
    dom = xml.dom.minidom.parse(path + "/" + file)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')
    res = ""
    User_text = []
    for i in range(len(title)):
        res = res + title[i].firstChild.data + text[i].firstChild.data+'\n'
        tmp = process_raw(title[i].firstChild.data + text[i].firstChild.data)
        if len(tmp)>0:
            User_text.append(tmp)
        post_num += 1
    res.lower()
    res = re.sub('\n', ' ', res)
    res = res.strip()
    res = res.split()
    emoji_cnt = 0
    antidep_cnt = 0
    for word in res:
        if word==':)' or word==':(' or word=='):' or word=='(:':
            emoji_cnt += 1
        if word in antidepressants:
            antidep_cnt += 1
    res = ' '.join(res)
    return_str = ""
    words = nltk.word_tokenize(res)
    for word in words:
        if word.isalpha():
            return_str= return_str + word + ' '

    # print(return_str)
    print(file + " complete!")

    return return_str,User_text,post_num,emoji_cnt,antidep_cnt

FEATURE_NUM = 9

def cal_features(posts,postnum):
    x = np.zeros(FEATURE_NUM, dtype=float)

    for post in posts:
        liwc_counts, total_num = get_LIWC(post)
        x[0] += liwc_counts['function (Function Words)']
        x[1] += liwc_counts['pronoun (Pronouns)']
        x[2] += liwc_counts['i (I)']
        x[3] += liwc_counts['ppron (Personal Pronouns)']
        x[4] += liwc_counts['verb (Verbs)']
        x[5] += liwc_counts['cogproc (Cognitive Processes)']
        x[6] += liwc_counts['focuspresent (Present Focus)']

        # x[7] += liwc_counts['certain (Certainty)']
        # x[8] += liwc_counts['affect (Affect)']
        # x[9] += liwc_counts['sexual (Sexual)']
        # x[10] += liwc_counts['ingest (Ingest)']
        # x[11] += liwc_counts['health (Health)']

    for i in range(FEATURE_NUM-2):
        x[i] /= postnum

    return x


def processdata(files,path):
    n=0
    res = []
    LIWC_features = []
    for file in files:
        text,posts,postnum,emoji_cnt,antidep_cnt = get_input_data(file, path)
        res.append(text)
        LIWC_fea = cal_features(posts,postnum)
        LIWC_fea[FEATURE_NUM-2] = emoji_cnt/postnum
        LIWC_fea[FEATURE_NUM-1] = antidep_cnt
        LIWC_features.append(LIWC_fea)
        n = n+1
        # if n==4:
        #     break
    return res,n,LIWC_features


def cal_F1(y,y_hat):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y)):
        if y[i] == 1:
            if y_hat[i] == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_hat[i] == 1:
                fp = fp + 1
            else:
                tn = tn + 1

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    F1 = 2 * p * r / (p + r)
    acc = (tn + tp) / (tn + tp + fp + fn)
    print('F1:' + str(F1) + " p:" + str(p) + " r:" + str(r) + ' acc:' + str(acc))

nega_path="./dataset/negative_examples_anonymous"
posi_path="./dataset/positive_examples_anonymous"

nega_test_path = "./dataset/negative_examples_test"
posi_test_path = "./dataset/positive_examples_test"


def get_features():
    nega_files = os.listdir(nega_path)
    posi_files = os.listdir(posi_path)
    nega_test_files = os.listdir(nega_test_path)
    posi_test_files = os.listdir(posi_test_path)

    posires, posinum, feature1 = processdata(posi_files, posi_path)
    negares, neganum, feature2 = processdata(nega_files, nega_path)
    positest, positestnum, feature3 = processdata(posi_test_files, posi_test_path)
    negatest, negatestnum, feature4 = processdata(nega_test_files, nega_test_path)

    features = np.array(feature1 + feature2 + feature3 + feature4)

    tv = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english', min_df=2)
    lda = LatentDirichletAllocation(n_components=25, learning_offset=10., random_state=0, max_iter=100)

    totalX = posires + negares + positest + negatest

    # TF-IDF的结果
    totalX = tv.fit_transform(totalX)
    # LDA的结果
    docres = lda.fit_transform(totalX)
    print('doc', docres, docres.shape)
    print('lda', lda.components_.shape)

    totalX = totalX.toarray()

    # print(totalX.shape,docres.shape,type(totalX),type(docres),features.shape)
    totalX = np.concatenate((docres, features), axis=1)
    # totalX=np.concatenate((totalX,features),axis=1)

    trainX = totalX[:(posinum + neganum)]
    testX = totalX[(posinum + neganum):]
    print(testX.shape)
    print(trainX.shape)

    return trainX,testX


if __name__ == '__main__':
    trainX,testX=get_features()

    # clf = LogisticRegression(random_state=0,C=16,penalty='l1',class_weight={0:0.2,1:0.8},dual=False,solver='liblinear')
    # clf.fit(trainX,trainY)
    # # with open('res.txt','w') as f:
    # #     f.write(str(clf.coef_.tolist()))
    # #     f.write(str(tv.vocabulary_))
    #
    # print("coef:",len(clf.coef_.tolist()[0]), clf.coef_)
    # # print("vocab:",len(tv.vocabulary_))
    # # vocab = sorted(tv.vocabulary_.items(), key=lambda d:d[1])
    # # print('ff',vocab)
    # # coefs = clf.coef_[0]
    # # sort_coef = np.argsort(coefs)
    # #
    # # nega_voc = []
    # # posi_voc = []
    # # for i in range(30):
    # #     nega_voc.append((vocab[sort_coef[i]][0],coefs[sort_coef[i]]))
    # # for i in range(1,31):
    # #     posi_voc.append((vocab[sort_coef[-i]][0],coefs[sort_coef[-i]]))
    # # print(nega_voc)
    # # print(posi_voc)
    # y_hat = clf.predict(trainX)
    # test_y_hat = clf.predict(testX)
    #
    # print('train:')
    # cal_F1(trainY,y_hat)
    # print('test:')
    # cal_F1(testY,test_y_hat)