
# coding: utf-8

# In[2]:


import numpy as np

import time

import re
from collections import Counter,defaultdict

import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']='1';


train_path='./DBPedia.full/full_train.txt';
test_path='./DBPedia.full/full_test.txt';
devel_path='./DBPedia.full/full_devel.txt';

# # Train Data
# 

# In[3]:

# vocab_count=Counter();
def func():
    return 0;

def data_read(path,train_flag=0):  
    
    vocabulary_count =defaultdict(func);
#     doc_count = defaultdict(func);
    
    file = open(path);
    file=file.readlines();

    train_data=[];
    labels=[];

    for line in file:
        
        
        total_labels=line.strip().split('\t')[0];
        
        line=re.sub(r'[^\w\s]','',line); ## removing punctuation and other small chars
        
        train_data.append( (line.strip().split('\t')[1:])[0].split()[2:] );
        
        
        
        
        if(train_flag==1):
           
            a=( (line.strip().split('\t')[1:])[0].split()[2:] );
            
            for word in a:
                vocabulary_count[word]+=1;

#             for word in list(set(a)):
#                 doc_count[word]+=1;

        labels.append(total_labels.split(','));
    
    for i in range(len(train_data)): ## lower chars
        train_data[i]=[word.lower() for word in train_data[i]];
        train_data[i][-1]=train_data[i][-1][:-2]; ## removing en in data
        labels[i][-1]=labels[i][-1][:-1]; ## removing spaces 
        
        
    return train_data,labels,vocabulary_count;


def label_to_key(labels,dict_):
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels[i][j]=dict_[labels[i][j]]
    return labels;



# In[3]:


x_train,y_train,vocab_count=data_read(train_path,1);

x_test,y_test,_=data_read(test_path,0);
# x_val,y_val,_,_=data_read(devel_path,0);


# In[4]:



word_to_index={};
count=1;

for key in vocab_count:
#     print(vocab_count[key])
    if vocab_count[key]<10000 and vocab_count[key]>100:
        word_to_index[key]=count;
        count+=1;
        


# In[5]:


for i in range(len(x_train)):
    x_train[i]=[word_to_index[word] for word in x_train[i] if vocab_count[word]<10000 and vocab_count[word]>100 ];

# for i in range(len(x_val)):
#     x_val[i]=[word_to_index[word] for word in x_val[i] if vocab_count[word]<10000 and vocab_count[word]>100 ];

for i in range(len(x_test)):
    x_test[i]=[word_to_index[word] for word in x_test[i] if vocab_count[word]<10000 and vocab_count[word]>100 ];


# In[6]:


dict_=list(set([label for line in y_train for label in line]));
label_to_index=dict( (x,y) for x,y in zip(dict_,range(0,len(dict_) ) ));
print(label_to_index)

y_train,y_test=label_to_key(y_train,label_to_index),label_to_key(y_test,label_to_index)
y_test


# In[7]:


y_train=[tuple(i) for i in y_train];
y_test=[tuple(i) for i in y_test];
# y_val=[tuple(i) for i in y_val];


# In[8]:


## multi labels 
from sklearn.preprocessing import MultiLabelBinarizer
enc=MultiLabelBinarizer();
y_train=enc.fit_transform(y_train)


# In[9]:


y_test=enc.transform(y_test);
# y_val=enc.transform(y_val);


# In[10]:


#### dividing weight in labels equally 

def row_normalise(arr):
    arr=np.asarray(arr,np.float64);
    for i in range(len(arr)):
        arr[i,:]=arr[i]/np.sum(arr[i]);
    return arr;

y_train=row_normalise(y_train);
y_test=row_normalise(y_test);
# y_val=row_normalise(y_val);


# In[11]:


data_train=np.zeros( (  len(x_train),len(word_to_index) ))
data_test=np.zeros( (  len(x_test),len(word_to_index) ))

for i in range(data_train.shape[0]):
    for key in x_train[i]:
        data_train[i][key-1]=1;

        
for i in range(data_test.shape[0]):
    for key in x_test[i]:
        data_test[i][key-1]=1;


# # Saving Data as h5py
# 

# In[12]:


### saving data as a sparse matrix 
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz


save_npz('data_train',coo_matrix(data_train) );
save_npz('y_train',coo_matrix(y_train));

save_npz('data_test',coo_matrix(data_test) );
save_npz('y_test',coo_matrix(y_test));


# In[20]:






