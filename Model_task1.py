#imports 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder 
#metricas
from sklearn.metrics import ndcg_score

#modelos
import lightgbm as lgb
import faiss
from annoy import AnnoyIndex

from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize



#from sentence_transformers import SentenceTransformer

import math



def train_section():
    
    #open dataset's
    train_df = pd.read_csv('dataset/train-v0.3.csv')
    product = pd.read_csv('dataset/product_catalogue-v0.3.csv')

    #meclando datasets
    full_train = pd.merge(train_df, 
                            product[["product_id", "product_locale", "product_title"]],
                            left_on=["product_id", "query_locale"], 
                            right_on=["product_id", "product_locale"]
    )
    '''
    #add colum id
    full_train['id']=full_train.index
    #remove NAN
    full_train = full_train.dropna()

    #Removendo StopWords / lowcase / caracteres especiais
    mask = (full_train['product_locale'] == 'us')
    full_train = full_train[mask]
    full_train = full_train.iloc[:500,:] #pegando as 500 primeiras amostras

    #print(full_train)

    #remove Stop Words
    stop_words = stopwords.words('english')
    text = full_train['product_title']
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
 
    text = text.str.lower() #remove lowcase
    text = text.str.replace('[!,#,@,&,%,0-9]','') #revome caracteres especiais 

    print(text)
    '''
    text = preProcessing(full_train)
    print(text)
    
    #criando vetores 
    sentences = text.tolist()
    model = SentenceTransformer('bert-base-nli-mean-tokens')  # initialize sentence transformer model
    # create sentence embeddings
    sentence_embeddings = model.encode(sentences)
    faiss.normalize_L2(sentence_embeddings) ## Normalising the Embeddings
    sentence_embeddings.shape

    #indexflat2
    size = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(size)
    index.is_trained
    index.add(sentence_embeddings)

    
    ncentroids=50 ## This is a hyperparameter, and indicates number of clusters to be split into
    m=16 ## This is also a hyper parameter
    quantiser = faiss.IndexFlatL2(size)
    index = faiss.IndexIVFPQ (quantiser, size,ncentroids, m , 8)
    index.train(sentence_embeddings) ## This step, will do the clustering and create the clusters
    print(index.is_trained)
    faiss.write_index(index, "trained.index")

    ### We have to add the embeddings to the Trained Index.
    ids=full_train['id'].tolist()

    ids=np.array(ids)
    index.add_with_ids(sentence_embeddings,ids)

    query="arsoft"
    search_result=searchFAISSIndex(full_train,"id",query,index,nprobe=10,model=model,topk=20)
    search_result=search_result[['id','product_id','product_title','esci_label','cosine_sim','L2_score']]
    print(search_result)
 



def calculateInnerProduct(L2_score):
    return (2-math.pow(L2_score,2))/2



def preProcessing(data = pd.DataFrame()):
    data['id'] = data.index
    #print(data)

        #remove NAN
    data = data.dropna()

    #Removendo StopWords / lowcase / caracteres especiais
    mask = (data['product_locale'] == 'us')
    data = data[mask]
    data = data.iloc[:500,:] #pegando as 500 primeiras amostras

    #print(full_train)

    #remove Stop Words
    stop_words = stopwords.words('english')
    text = data['product_title']
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
 
    text = text.str.lower() #remove lowcase
    text = text.str.replace('[!,#,@,&,%,0-9]','') #revome caracteres especiais 

    
    return text



def searchFAISSIndex(data,id_col_name,query,index,nprobe,model,topk=20):
    ## Convert the query into embeddings
    query_embedding = model.encode([query])[0]
    dim=query_embedding.shape[0]
    query_embedding=query_embedding.reshape(1,dim)
    #faiss.normalize_L2(query_embedding)
  
    
    index.nprobe=nprobe
    
    D,I=index.search(query_embedding,topk) 
    ids=[i for i in I][0]
    L2_score=[d for d in D][0]
    inner_product=[calculateInnerProduct(l2) for l2 in L2_score]
    search_result=pd.DataFrame()
    search_result[id_col_name]=ids
    search_result['cosine_sim']=inner_product
    search_result['L2_score']=L2_score
    print(search_result)
    dat=data[data[id_col_name].isin(ids)]
    dat=pd.merge(dat,search_result,on=id_col_name)
    dat=dat.sort_values('cosine_sim',ascending=False)
    return dat


#open dataset's
train_df = pd.read_csv('dataset/train-v0.3.csv')
product = pd.read_csv('dataset/product_catalogue-v0.3.csv')

    #meclando datasets
full_train = pd.merge(train_df, 
                        product[["product_id", "product_locale", "product_title"]],
                        left_on=["product_id", "query_locale"], 
                        right_on=["product_id", "product_locale"]
)


    #open dataset's
train_df = pd.read_csv('dataset/train-v0.3.csv')
product = pd.read_csv('dataset/product_catalogue-v0.3.csv')

    #meclando datasets
full_train = pd.merge(train_df, 
                        product[["product_id", "product_locale", "product_title"]],
                        left_on=["product_id", "query_locale"], 
                        right_on=["product_id", "product_locale"]
)

#preProcessing(full_train)
train_section()
