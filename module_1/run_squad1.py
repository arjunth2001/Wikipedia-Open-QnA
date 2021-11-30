#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
with open("../data/dev-v2.0.json") as f:
    data = json.load(f)
data = data['data']
jsonl=[]


# In[2]:


import wikipedia
from retriever import Retriever
from gensim.matutils import cossim
from gensim import utils
BASE_PATH = "/scratch/arjunth2001/"
ret = Retriever(BASE_PATH)


# In[3]:


from tqdm.auto import tqdm
corpus =[]
for doc in tqdm(data):
    title = doc['title']
    paragraphs = doc['paragraphs']
    for para in paragraphs:
        qas = para['qas']
        for qa in qas:
            question = qa['question']
            context = para['context']
            answers = [a['text'] for a in qa['answers']]
            if (question,answers,context,title) not in corpus:
                corpus.append((question,answers,context,title))


# In[4]:


len(corpus)


# In[5]:


import spacy
nlp = spacy.load("en_core_web_sm")
def get_length(text):
    text = text.replace('\n', ' ')
    doc = nlp(text, disable=['parser','tagger','ner'])
    return len(doc)


# In[6]:


Q_TYPE = {'what was': 0, 'what is': 1, 'what': 2, 'in what': 3, 'in which': 4, 'in': 5,
          'when': 6, 'where': 7, 'who': 8, 'why': 9, 'which': 10, 'is': 11, 'other': 12}


def get_question_type_features(question):
    qwords = question.split(' ')
    if qwords[0].lower() in Q_TYPE:
        return Q_TYPE[qwords[0].lower()]
    if ' '.join(list(map(lambda x: x.lower(), qwords[0:2]))) in Q_TYPE:
        return Q_TYPE[' '.join(list(map(lambda x: x.lower(), qwords[0:2])))]
    return Q_TYPE['other']


# In[7]:


import sqlite3
connection = sqlite3.connect("/scratch/arjunth2001/wiki.db", check_same_thread=False)


def get_doc_text(doc_id):
    cursor = connection.cursor()
    cursor.execute(
        "SELECT text FROM documents WHERE id = ?",
        (doc_id,)
    )
    result = cursor.fetchone()
    cursor.close()
    if result is None:
        return None
    return result[0]
def get_doc_text2(title):
    cursor = connection.cursor()
    cursor.execute(
        "SELECT id , text FROM documents WHERE title = ?",
        (title,)
    )
    result = cursor.fetchone()
    cursor.close()
    if result is None:
        return None
    return result[0], result[1]


# In[8]:


get_doc_text("23621594")


# In[9]:


get_doc_text2("Euchlaena")


# In[10]:


len(corpus)


# In[ ]:


file_path="/scratch/arjunth2001/dataset1"
jsonl = []
for i,(q,ans,c,t) in tqdm(enumerate(corpus),total=len(corpus)):
    rets = ret.get_similar(q)
    q_vec = ret.getTfidfForText(q)
    #_id , doc_text = get_doc_text2(t)
    #my_dict = {
        #'doc_id': _id,
        #'title':t,
        #'doc_sim': cossim(q_vec, ret.getTfidfForText(doc_text)),
        #'par_sim': cossim(q_vec, ret.getTfidfForText(c)),
        #'par_length': get_length(c),
        #'doc_length': get_length(doc_text),
        #'question_type': get_question_type_features(q),
        #'truths': list(set(ans)),
        #'q': q,
        #'para': c,
    #}
    #rele =[my_dict]
    rele = []
    for (_id,title),doc_sim in rets:
        content = get_doc_text(str(_id))
        if content is None:
            continue
        paras = content.split('\n')
        for para in paras:
            my_dict ={
                'doc_id':_id,
                'title':title,
                'doc_sim': doc_sim,
                'par_sim':cossim(q_vec,ret.getTfidfForText(para)),
                'par_length':get_length(para),
                'doc_length':get_length(content),
                'question_type':get_question_type_features(q),
                'truths':list(set(ans)),
                'q':q,
                'para':para
            }
            rele.append(my_dict)
    rele = sorted(rele, key=lambda item: -item["par_sim"])
    jsonl.append(rele)
    if len(jsonl)==1000:
        with open(file_path+f"/{i}.json","w") as f:
            json.dump(jsonl,f)
        jsonl=[]
if len(jsonl)!=0:
    with open(file_path+f"/final.json","w") as f:
        json.dump(jsonl,f)
    jsonl=[]


# In[ ]:




