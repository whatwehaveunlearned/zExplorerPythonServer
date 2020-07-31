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
        self.text_max_length = 0;
        self.lastStepDocuments = pd.DataFrame()
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
            self.topic_UMAP = umap.UMAP(n_neighbors=3, n_components=2, metric='euclidean')
        #Filter False Values
        clean_bert_vectors = doc_dictionary['abstract_vector'][doc_dictionary['abstract_vector']!= False]
        clean_topic_vectors = doc_dictionary['topics_vector'][doc_dictionary['topics_vector']!=False]
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
                    doc_vector.extend(doc_vector[:size_to_extend])
                    if len(doc_vector) == self.text_max_length:
                        break
        #topics have different sizes too
        if self.topic_min_length == 0:
            topic_length = clean_topic_vectors.apply(lambda x: len(x))
            self.topic_min_length = topic_length.min()
        clean_topic_vectors = clean_topic_vectors.apply(lambda x: x[:self.topic_min_length])
        if self.encoder == False :
            print('Encoder has not been trained yet so Using UMAP for fitting')
            vectors_list = clean_bert_vectors.values.tolist()
            topics_list  = clean_topic_vectors.values.tolist()
            self.UMAP = self.UMAP.fit(vectors_list)
            self.topic_UMAP = self.topic_UMAP.fit(topics_list)
            vec_2d = self.UMAP.transform(vectors_list)
            vec_2d_UMAP = vec_2d
            vec_topic_2d = self.topic_UMAP.transform(topics_list)
        else:
            print('Using Enconder to Fit')
            papers_to_fit_indexes = doc_dictionary[doc_dictionary.isna().any(axis=1)].index.values.tolist()
            papers_to_fit = clean_bert_vectors.loc[papers_to_fit_indexes]
            try:
                new_projections = self.encoder.transform(np.array(papers_to_fit.values.tolist()))
            except ValueError:
                print('error')
                pdb.set_trace()
            vec_2d = np.append(self.vectors2D,new_projections,axis=0)
            papers_to_fit_topics = clean_topic_vectors.loc[papers_to_fit_indexes]
            new_projections = self.topic_UMAP.transform(papers_to_fit_topics.values.tolist())
            vec_topic_2d = np.append(self.vectors2D_topics,new_projections,axis=0)
            # We also train UMAP for comparison
            vectors_list = clean_bert_vectors.values.tolist()
            # self.UMAP = self.UMAP.fit(vectors_list)
            vec_2d_UMAP = self.UMAP.transform(vectors_list)
        #Store values for subsequent runs
        self.vectors2D = vec_2d
        self.vectors2D_topics = vec_topic_2d
        #add to pandas columns
        doc_dictionary['vec_2d'] = vec_2d.tolist()
        doc_dictionary['vec_topic_2d']= vec_topic_2d.tolist()
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

    def update_model(self,new_data):
        self.already_encoded_papers = new_data
        #loop list of papers
        for index, row in new_data.iterrows():
            #Get from paper in session
            this_paper = self.documents.loc[row['key']]
            #Set new x,y coordinates from paper in session
            this_paper.vec_2d[0] = new_data['x'][index]
            this_paper.vec_2d[1] = new_data['y'][index]
        #Get 300Vectors and 2D Vectors
        clean_bert_vectors = self.documents['abstract_vector']
        vec_2d = np.array(self.documents['vec_2d'].values.tolist())
        self.train_encoder(clean_bert_vectors,vec_2d)
    
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

            