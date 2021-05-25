import pandas as pd

import pdb

#Functions from NSA Are called from shell because they run in Python3
import subprocess
import os
import pickle
import csv


import umap
import json

from topic_extractor import topic_extractor
from classes.document_class import Document
from classes.encoder_class import Encoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#TO load UMAP SESS MOdel
import joblib

from datetime import datetime
import math


#For Calculating Latent Space Stuff
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#Session
class Session:
    """Session Class"""
    def __init__(self, session_id):
        self.id = session_id
        self.sess_folder = './sessData/' + session_id
        self.documents = False
        self.topics = False
        self.topic_params = False
        self.authorList = False
        self.words = False
        self.UMAP = False
        self.topic_UMAP = False
        self.text_min_lenght = 0
        self.topic_min_length = 0
        self.abstract_conclusion_min_length = 0
        self.vectors2D = False
        self.vectors2D_topics = False
        self.encoder = False
        self.text_max_length = 0
        self.lastStepDocuments = pd.DataFrame()
        self.constrative_model_trained = False
        self.contrastive_model = False
        self.embedding_model = False
        self.study_results = pd.DataFrame()
        #Initialize Values, now I always initialize session from scratch but store things in global so I load global
        if session_id == 'globalSess':
            try:
                # self.documents = pd.read_csv(self.sess_folder + '/documents.csv',encoding='utf-8',index_col='index')
                self.documents = pd.read_json(self.sess_folder + '/documents.json')
                # self.documents = self.read_csv()
            except ValueError:
                self.documents = pd.DataFrame()
        else:
            self.documents = pd.DataFrame()
        ##LOAD THE REST OF THE DATA MAYBE I DONT NEED THIS NEED TO THINK ABOUT IT I STORE EVERYTHING IN DOCS EXCEPT THE SESS TOPICS THAT I CALCULATE EVERY TIME
        try:
            self.topics = pd.read_csv(self.sess_folder + '/topics.csv',encoding='utf-8')
        except IOError:
            self.topics = pd.DataFrame()
        try:
            self.authorList = pd.read_csv(self.sess_folder + '/authors.csv',encoding='utf-8',index_col='index')
        except IOError:
            self.authorList = pd.DataFrame()
        try:
            self.words = pd.read_csv(self.sess_folder + '/words.csv',encoding='utf-8')
        except IOError:
            self.words = pd.DataFrame()

    def addDoc(self, doc):
        """ Function to add a new document to the session """
        doc_to_add = doc.create_document_msg()
        document = pd.DataFrame([doc_to_add],columns=doc_to_add.keys(),index=[doc_to_add['globalID']])
        self.documents = self.documents.append(document)

    def addDocLastStep(self, doc):
        """ Function to add a new document to the session """
        doc_to_add = doc.create_document_msg()
        document = pd.DataFrame([doc_to_add],columns=doc_to_add.keys(),index=[doc_to_add['globalID']])
        self.lastStepDocuments = self.lastStepDocuments.append(document)

    def storeSessData(self):
        '''Store Session'''
        try:
            self.documents.to_json(self.sess_folder + '/documents.json')
        except ValueError:
            print('')
            pdb.set_trace()
    def returnDoc(self,doc):
        """Returns a specific document from a session"""
        return self.documents.loc[doc]
    
    def returnDocsBy(self, type):
        """Returns the documents of session ordered by type passed (authors and years) """
        docs_by_array = []
        if type == 'author':
            if isinstance(self.authorList.index,list):
                for each_author in self.authorList.index:
                    element = {
                    'author':each_author,
                    'Paper_Ids': self.authorList.loc[each_author]['Paper_Ids']   
                    }
                    docs_by_array.append(element)
            else: #Only one paper
                element = {
                'author':self.authorList.index,
                'Paper_Ids': self.authorList.loc[self.authorList.index]['Paper_Ids']   
                }
                docs_by_array.append(element)
        return docs_by_array

    
    def docInSess(self,doc):
        """Aux function to show if a document is already in a session"""
        try:
            is_doc_in_sess = self.documents['globalID'].isin([doc]).any()
        except KeyError:
            is_doc_in_sess = False
        return is_doc_in_sess

    def addDocTopics(self, doc):
        for topic in doc.topics:
            self.topics.append(topic)
             
    def addAuthor(self, author,doc_id):
        """Function to add author to session authorList"""
        try:
            papers_in_collection = author['Papers_in_collection']
            self.authorList.loc[author.Author, 'Papers_in_collection'] = papers_in_collection + 1
            paper_id_array = []
            paper_id_array.append(self.authorList.loc[author.Author, 'Paper_Ids'])
            paper_id_array.append(doc_id)
            self.authorList.loc[author.Author, 'Paper_Ids'] = paper_id_array.append(doc_id)

        except KeyError:
            author['Papers_in_collection'] = 1
            author['Paper_Ids'] = [doc_id]
            self.authorList = self.authorList.append(author)
    
    def searchAuthor(self, author):
        """ Aux Function to search for an author in the session returns True if in Session and False if not"""
        # pdb.set_trace()
        try:
            self.authorList.loc[author]
            is_author = True
        except KeyError:
            is_author = False
        return is_author

    def returnAuthor(self, author):
        author_name = author['firstName'] + ' ' +  author['lastName']
        return self.authorList.loc[author_name]
        
    
    def get_topics_by(self,data,organized_by):
        """Calculates the topics of the session organized by authors or years"""
        if organized_by == 'author':
            df = pd.DataFrame()
            #Get papers for each author
            # pdb.set_trace()
            for each_author in data:
                for each_paper in each_author['Paper_Ids']:
                    # pdb.set_trace()
                    df = df.append(self.returnDoc(each_paper))
        return self.get_topics(df)


    def get_topics(self,doc_dictionary): #Good One
        """Returns Topics object and Words Object from documents df passed"""
        topics_data = topic_extractor(doc_dictionary,'session')
        self.topics = topics_data['topics']
        self.topic_params = topics_data['topic_params']
        self.words = topics_data['lvls_df']
        return {'topics':self.topics,'words':self.words}

    """Function to calculate 2D projections of Paper Vectors in session"""
    def train_fit_UMAP_2D(self,doc_dictionary):
        if self.UMAP == False:
            self.UMAP = umap.UMAP(n_neighbors=3, n_components=2, metric='euclidean')
            # self.topic_UMAP = umap.UMAP(n_neighbors=3, n_components=2, metric='euclidean')
        clean_bert_vectors = doc_dictionary['abstract_vector'].copy()
        #Filter False Values
        # clean_bert_vectors = doc_dictionary['abstract_vector'][doc_dictionary['abstract_vector']!= False]
        # clean_bert_vectors = doc_dictionary['title_vector'][doc_dictionary['title_vector']!= False]
        # clean_topic_vectors = doc_dictionary['topics_vector'][doc_dictionary['topics_vector']!=False]
        #Since text are different sizes we need to set them to the same size. We extend smaller vectors to the size of the longest by duplicating its content.
        if self.text_max_length == 0:
            text_lenght = clean_bert_vectors.apply(lambda x: len(x))
            self.text_max_length = text_lenght.max()
        for index, doc_vector in clean_bert_vectors.iteritems():
            if len(doc_vector) > self.text_max_length:
                print ("document" + index + "has a longer size than any in previous")
                doc_vector = doc_vector[:self.text_max_length]
            if len(doc_vector) != self.text_max_length:
                while True:
                    size_to_extend = self.text_max_length - len(doc_vector)
                    # pdb.set_trace()
                    doc_vector.extend(doc_vector[:size_to_extend])
                    if len(doc_vector) == self.text_max_length:
                        break
        # NEW AVERAGE VECTORS ##################
        for index, doc_vector in clean_bert_vectors.iteritems():
            vectors = np.split(np.array(doc_vector),len(doc_vector)/768)
            clean_bert_vectors[index] = np.average(vectors, axis=0)
        #topics have different sizes too
        # if self.topic_min_length == 0:
        #     topic_length = clean_topic_vectors.apply(lambda x: len(x))
        #     self.topic_min_length = topic_length.min()
        # clean_topic_vectors = clean_topic_vectors.apply(lambda x: x[:self.topic_min_length])
        # if self.encoder == False :
        # pdb.set_trace()
        if self.constrative_model_trained == False:
            print('Encoder has not been trained yet so Using UMAP for fitting')
            vectors_list = clean_bert_vectors.values.tolist()
            # topics_list  = clean_topic_vectors.values.tolist()
            self.UMAP = self.UMAP.fit(vectors_list)
            # self.topic_UMAP = self.topic_UMAP.fit(topics_list)
            vec_2d = self.UMAP.transform(vectors_list)
            vec_2d_UMAP = vec_2d
            # vec_topic_2d = self.topic_UMAP.transform(topics_list)
        else:
            # print('Using Enconder to Fit')
            print('Using Constrative Model to Fit')
            # pdb.set_trace()
            papers_to_fit_indexes = doc_dictionary[doc_dictionary.isna().any(axis=1)].index.values.tolist()
            papers_to_fit = clean_bert_vectors.loc[papers_to_fit_indexes]
            try:
                # new_projections = self.encoder.transform(np.array(papers_to_fit.values.tolist()))
                new_projections = self.embedding_model.predict(np.array(papers_to_fit.values.tolist()))[0]
            except ValueError:
                print('error')
                pdb.set_trace()
            vec_2d = np.append(self.vectors2D,new_projections,axis=0)
            # papers_to_fit_topics = clean_topic_vectors.loc[papers_to_fit_indexes]
            # new_projections = self.topic_UMAP.transform(papers_to_fit_topics.values.tolist())
            # vec_topic_2d = np.append(self.vectors2D_topics,new_projections,axis=0)
            # We also train UMAP for comparison
            vectors_list = clean_bert_vectors.values.tolist()
            # self.UMAP = self.UMAP.fit(vectors_list)
            vec_2d_UMAP = self.UMAP.transform(vectors_list)
        #Store values for subsequent runs
        self.vectors2D = vec_2d
        # self.vectors2D_topics = vec_topic_2d
        #add to pandas columns
        doc_dictionary['vec_2d'] = vec_2d.tolist()
        doc_dictionary['vec_topic_2d'] = "" #substitutes below no topics in study
        # doc_dictionary['vec_topic_2d']= vec_topic_2d.tolist()
        doc_dictionary['vec_2d_UMAP'] = vec_2d_UMAP.tolist()
        #change NA for false
        doc_dictionary['vec_2d'].fillna(False)
        doc_dictionary['vec_topic_2d'].fillna(False)
        return doc_dictionary

        # conclusion_vec_2d.tolist()
    
    def get_years(self):
        return self.documents.groupby('year')['year'].count()

    def assign_topics_to_documents(self):
        """Function to assign session topics to doc topics"""
        def calculate_cosine_similarity(vect1,vect2,size):
            vect1 = np.array(vect1).reshape(1,size)
            vect2 = np.array(vect2).reshape(1,size)
            return cosine_similarity(vect1,vect2)
        
        sess_topic_params = self.topic_params
        for each_document in self.documents.iterrows():
            doc_topic_params_df = pd.DataFrame(each_document[1]['topic_params'])
            similarity_vector = []
            for each_topic in doc_topic_params_df.iterrows():
                #We compare each topic in paper with session_topics if its above a threshold we assign the topic to paper
                similarity = sess_topic_params['vector300'].apply(calculate_cosine_similarity,vect2=each_topic[1]['vector300'],size=300)
                #apply threshold
                similarity = similarity.apply(lambda x: x[0][0] > 0.7) #returns true false vectors
                similarity_vector.append(similarity)
            collapsed_similarity = sum(similarity_vector)
            collapsed_similarity = collapsed_similarity.where(collapsed_similarity==0,1) #Returns vector of ceros and ones
            sess_topic_params[each_document[1]['globalID']] = collapsed_similarity * sess_topic_params['weight']

    def update_model(self,new_data,stepName):
        self.study_results = pd.DataFrame()
        self.already_encoded_papers = new_data
        #Initialize cluster columns
        self.documents['cluster'] = ""
        self.documents['clusterID'] = ""
        self.documents['predicted_label'] = ""
        self.documents['clusterColor'] = ""

        #Get the clusters form labels and transform to integer categories for processing
        new_data['clusterID'] = new_data['cluster'].astype('category').cat.codes
        # pdb.set_trace()
        self.study_results = new_data[['key','title','cluster','clusterID','distance_moved','x','y']].set_index('key').sort_values(by=['title'])

        #loop list of papers
        for index, row in new_data.iterrows():
            #Get from paper in session
            this_paper = self.documents.loc[row['key']]
            #Set new x,y coordinates from paper in session
            this_paper.vec_2d[0] = new_data['x'][index]
            this_paper.vec_2d[1] = new_data['y'][index]
            #Set the cluster assignations
            # pdb.set_trace()
            this_paper.cluster = new_data['cluster'][index]
            this_paper.clusterID = new_data['clusterID'][index]
            this_paper.clusterColor = new_data['color'][index]
            #This forces the storage on the original df of this_paper copy
            self.documents.loc[row['key']] = this_paper
            # pdb.set_trace()
            # #Calculate distance and store in study
            # if math.isnan(new_data['distance_moved'][index]):
            #     this_paper[distance_column_name] = 0
            # else:
            #     this_paper[distance_column_name] = new_data['distance_moved'][index]
            #This forces the storage on the original study results with info about distances 
            # self.study_results.loc[row['key']] = this_paper
            # pdb.set_trace()
            # self.study_results = pd.concat([self.study_results,this_paper],axis=1)
            
        # clean_bert_vectors = self.documents['title_vector']
        clean_bert_vectors = self.documents['abstract_vector'].copy()
        
        # NEW AVERAGE VECTORS ##################
        for index, doc_vector in clean_bert_vectors.iteritems():
            vectors = np.split(np.array(doc_vector),len(doc_vector)/768)
            clean_bert_vectors[index] = np.average(vectors, axis=0)

        ### SECTION New Contrastive Model Troy  ####
        #transform cluster into vector encodings and create models
        # pdb.set_trace()
        embeddings = self.documents['vec_2d'].copy()
        categorical_data = self.to_categorical(self.documents['clusterID'],40)
        contrastive_data = self.get_contrastive_data(clean_bert_vectors, embeddings, categorical_data,len(clean_bert_vectors) * len(clean_bert_vectors) ) #documents squared
        self.contrastive_model = self.build_contrastive_model(768,40)
        self.embedding_model = self.get_embedding_model(self.contrastive_model, 768)
        #Get data needed to fit
        x1, x2, y, e1, e2, yc1, yc2 = self.get_inputs_and_outputs(contrastive_data)
        #compile models
        loss_weights = { "Classifier": 0.25, "Embedding": 0.5,"Contrastive": 0.25}
        loss_dict = {"Classifier": "categorical_crossentropy","Classifier_1": "categorical_crossentropy","Embedding": "mse","Embedding_1": "mse","Contrastive": "mse"}
        self.contrastive_model.compile(optimizer=Adam(lr=0.01), loss=loss_dict, loss_weights=loss_weights, metrics=["accuracy"])
        #Train constrative model
        self.contrastive_model.fit(x=[x1, x2], y=[y, e1, e2, yc1, yc2], epochs=10, validation_split=0.3, batch_size=16)
        self.constrative_model_trained = True
        #Transform current documents
        # pdb.set_trace()
        vectors_list = np.array(clean_bert_vectors.values.tolist())
        prediction = self.embedding_model.predict(vectors_list)
        new_vec_2d = prediction[0]   #WE CAN SEE WHAT WE DO WITH THE [1] value in the array that has the class predictions.
        # pdb.set_trace();
        self.documents['vec_2d'] = new_vec_2d.tolist()
        self.documents['predicted_label'] = np.argmax(prediction[1],axis=1)
        self.vectors2D = new_vec_2d
        #Store predicted class
        self.study_results['predicted_label'] = self.documents.sort_values(by=['title'])['predicted_label']

        #ISTORE THE DATA IN CSV
        # pdb.set_trace()
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        study_filename = "./sessData/studies/" + date_time + ".csv"
        self.study_results.to_csv(study_filename, encoding='utf-8')
        # study_filename_classes = "./sessData/studies/" + date_time + "_classes_prediction.csv"
        # self.documents = self.documents.sort_values(by=['title'])
        # pdb.set_trace()
        # self.documents.to_csv(study_filename_classes, columns = ['title','predicted_label'], encoding='utf-8')

    def train_encoder(self,vectors,vec_2d):
        vectors_list = np.array(vectors.values.tolist())
        #Train Encoder
        vector_size = self.text_max_length
        self.encoder = Encoder(vector_size,2)
        self.encoder.fit(vectors_list,vec_2d)
        new_vec_2d = self.encoder.transform(vectors_list)
        # pdb.set_trace()
        #add to pandas columns
        # self.documents.loc[self.already_encoded_papers['key']]['vec_2d'] = new_vec_2d.tolist()
        self.documents['vec_2d'] = new_vec_2d.tolist()
        self.vectors2D = new_vec_2d


    ############ Latent stuff Functions stuff ###################################################
    def get_contrastive_data(self,datax,embeddings, datay, n):
        results = []
        for i in range(0, n):
            index1 = np.random.randint(0, datax.shape[0])
            index2 = np.random.randint(0, datax.shape[0])
            datay1 = datay[index1]
            datay2 = datay[index2]
            label = 0 if np.argmax(datay1) == np.argmax(datay2) else 1
            results.append((
                datax[index1],
                datax[index2],
                label,
                embeddings[index1],
                embeddings[index2], 
                datay1, datay2
            ))
        return results

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=-1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def euclidean_distance_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def build_contrastive_model(self, input_shape, class_dim):
        # Specify the contrastive Model

        X1 = Input((input_shape,))
        X2 = Input((input_shape,))

        dense1 = Dense(300, name="dense1")
        bn1 = BatchNormalization(name="bn1")
        d1 = Dropout(rate=0.1, name="d1")
        a1 = Activation("tanh", name="a1")

        dense2 = Dense(200, name="dense2")
        bn2 = BatchNormalization(name="bn2")
        d2 = Dropout(rate=0.1, name="d2")
        a2 = Activation("tanh", name="a2")

        dense3 = Dense(100, name="dense3")
        bn3 = BatchNormalization(name="bn3")
        d3 = Dropout(rate=0.1, name="d3")
        a3 = Activation("tanh", name="a3")
        
        dense4 = Dense(50, name="dense4")
        bn4 = BatchNormalization(name="bn4")
        d4 = Dropout(rate=0.1, name="d4")
        a4 = Activation("tanh", name="a4")

        embedding = Dense(2, name="Embedding")
        contrastive = Lambda(
            self.euclidean_distance, 
            output_shape=self.euclidean_distance_shape,
            name="Contrastive"
        )
        class_pred = Dense(class_dim, activation="softmax", name="Classifier")

        H11 = a1(d1(bn1(dense1(X1))))
        H12 = a1(d1(bn1(dense1(X2))))

        H21 = a2(d2(bn2(dense2(H11))))
        H22 = a2(d2(bn2(dense2(H12))))

        H31 = a3(d3(bn3(dense3(H21))))
        H32 = a3(d3(bn3(dense3(H22))))

        H41 = a4(d4(bn4(dense4(H31))))
        H42 = a4(d4(bn4(dense4(H32))))

        E1 = embedding(H41)
        E2 = embedding(H42)

        Y = contrastive([E1, E2])
        YClass1 = class_pred(E1)
        YClass2 = class_pred(E2)

        model = Model(inputs=[X1, X2], outputs=[Y, E1, E2, YClass1, YClass2])
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        model_filename = "./sessData/studies/" + date_time + "_model.json"
        model_in_json = model.to_json()
        with open(model_filename, 'w') as outfile:
            json.dump(model_in_json, outfile)
        model.summary()
        return model

    # Capture the embeddings from the trained model layers
    def get_embedding_model(self, model, input_shape):
        X = Input((input_shape,))
        
        dense1 = model.get_layer("dense1")
        bn1 = model.get_layer("bn1")
        d1 = model.get_layer("d1")
        a1 = model.get_layer("a1")

        dense2 = model.get_layer("dense2")
        bn2 = model.get_layer("bn2")
        d2 = model.get_layer("d2")
        a2 = model.get_layer("a2")

        dense3 = model.get_layer("dense3")
        bn3 = model.get_layer("bn3")
        d3 = model.get_layer("d3")
        a3 = model.get_layer("a3")

        dense4 = model.get_layer("dense4")
        bn4 = model.get_layer("bn4")
        d4 = model.get_layer("d4")
        a4 = model.get_layer("a4")

        embedding = model.get_layer("Embedding")
        class_pred = model.get_layer("Classifier")

        H1 = a1(d1(bn1(dense1(X))))
        H2 = a2(d2(bn2(dense2(H1))))
        H3 = a3(d3(bn3(dense3(H2))))
        H4 = a4(d4(bn4(dense4(H3))))

        Y = embedding(H4)
        Class = class_pred(Y)
        # pdb.set_trace()

        result = Model(inputs=X, outputs=[Y, Class])
        result.summary()
        return result

    def as_cat(self, target, n_class):
        result = np.zeros(n_class)
        result[target] = 1
        return result

    def to_categorical(self,targets, n_class):
        return np.array([self.as_cat(target, n_class) for target in targets])

    # Get inputs and outpus from contrastive data
    def get_inputs_and_outputs(self, contrastive_data):
        dx1 = [x1 for (x1,_,_,_,_,_,_) in contrastive_data]
        dx2 = [x2 for (_,x2,_,_,_,_,_) in contrastive_data]
        dy =  [y  for (_,_,y,_,_,_,_)  in contrastive_data]
        de1 = [e1 for (_,_,_,e1,_,_,_) in contrastive_data]
        de2 = [e2 for (_,_,_,_,e2,_,_) in contrastive_data]
        dycat1 = [ycat1 for (_,_,_,_,_,ycat1,_) in contrastive_data]
        dycat2 = [ycat2 for (_,_,_,_,_,_,ycat2) in contrastive_data]
        return dx1, dx2, dy, de1, de2, dycat1, dycat2

            