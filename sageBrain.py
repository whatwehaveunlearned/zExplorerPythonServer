#!/usr/bin/python

#server sockets
import asyncio
import websockets

#Server stuff
import zerorpc
import logging
logging.basicConfig()
#Data Science Imports
import pandas as pd
#Debugg
import pdb
#json parser
import json
#To Import from Arxiv
import arxiv

#Import Brain Classes
from classes.session_class import Session
from classes.document_class import Document
from classes.topic_class import Topic #Not using this now
from classes.zotero_class import Zotero

#Interface Class to talk with Node
class SageBrain(object):
    """ Sage Brain Class"""
    def __init__(self, session_id):
        self.id = "brainInterface"
        self.sessions = []
        self.session_counter = -1
        self.actualSession = -1
        self.session_id = session_id
        self.addSession(session_id)
        self.zotero = Zotero('2476068','user','ravDnfy0bMKyuDrKq5kNz5Rh')
        self.sess_folder = './sessData/' + session_id
        self.globalSess = Session('globalSess')
       
    
    def reset(self):
        self.id = "brainInterface"
        self.sessions = []
        self.session_counter = -1
        self.actualSession = -1
        self.addSession(self.session_id)
        self.zotero = Zotero('2476068','user','ravDnfy0bMKyuDrKq5kNz5Rh')
        self.sess_folder = './sessData/' + self.session_id
        self.globalSess = Session('globalSess')

    def Zotero(self,function_name,collection_key,itemKeys,collection_items):
        value_to_return = False
        if function_name == 'getCollections':
            value_to_return = self.zotero.getColletions()
        elif function_name == 'getCollectionItems':
            value_to_return = self.zotero.getCollectionItems(collection_key)
        elif function_name == 'downloadItems':
            value_to_return = self.zotero.downloadItems(itemKeys,collection_items,self.sess_folder)
        
        return value_to_return

    def DocInterface(self, fileName, doc_id,dataType,metadata,step_type):
        """ Brain Interface """
        if dataType == 'zoteroCollection':
            self.actualSession.lastStepDocuments = pd.DataFrame()
            # pdb.set_trace()
            for index,each_doc in enumerate(metadata):
                doc_in_sess = self.actualSession.docInSess(metadata[index]['key'])
                doc_in_global = self.globalSess.docInSess(metadata[index]['key'])
                if doc_in_sess == True:
                    print ("Doc in sess")
                elif doc_in_global == False:
                    print ("New Document")
                    doc = Document(self.actualSession, metadata[index]['name'],  metadata[index]['key'], 'zotero', "user", each_doc)
                    self.globalSess.addDoc(doc)
                    self.actualSession.addDoc(doc)
                else:
                    print ("Doc in global")
                    doc_from_global = self.globalSess.returnDoc(metadata[index]['key'])
                    doc = Document(self.actualSession, 'name',  'key', 'inSession', "user", doc_from_global)
                    self.actualSession.addDoc(doc)
                # Lastly we add to last step documents to know what documents are from this step.
                self.actualSession.addDocLastStep(doc)

            #We get the topic and words Using Umap NSA algorithm and we include them into session
            # self.actualSession.get_topics(self.actualSession.documents)
        else:
            doc = Document(self.actualSession, fileName, doc_id, "doc", "user", False)
            self.actualSession.addDoc(doc)
        #Get Umap fit
        self.get_projections(step_type)
        current_data = self.send_current(step_type)
        # pdb.set_trace()
        return current_data

    def add_data_for_user_studies(self):
        #Incrementally add new data from the collection for user studies
        pdb.set_trace()

    def UpdateModel(self,new_paper_data,step):
        self.actualSession.update_model(new_paper_data,step)
        return self.actualSession.documents

        
    def addCitations(self):
        documents_msg = []
        for each_doc in self.actualSession.documents:
            print (each_doc.title)
            each_doc.GetScholarInfo()
            documents_msg.append(each_doc.create_document_msg())
        # pdb.set_trace()
        return {"documents":documents_msg}

    def get_projections(self,step_type):
        #We get the projections of the papers in the session
        self.actualSession.documents = self.actualSession.train_fit_UMAP_2D(self.actualSession.documents)
        self.globalSess.storeSessData()
        self.actualSession.storeSessData()


    def addSession(self, session_id):
        """ Function to add new sessions """
        self.session_counter = self.session_counter + 1
        self.sessions.append(Session(session_id))  
        self.actualSession = self.sessions[self.session_counter]

    def send_current(self,step_type):
        #Assign Session Papers to Session Topics
        # self.actualSession.assign_topics_to_documents()
        #We get the years
        years = self.actualSession.get_years()
        if(step_type == 'training'):
            return {"documents":self.actualSession.documents,"years":years,"authors":self.actualSession.authorList,"doc_topics":{'topics':self.actualSession.topics, 'topic_params':self.actualSession.topic_params, 'order':self.actualSession.topics.columns.values.tolist(), 'words':self.actualSession.words.to_json(orient='records')}}
        else:
            # pdb.set_trace()
            # Separate new papers
            this_step_papers = self.actualSession.documents[self.actualSession.documents.index.isin(self.actualSession.lastStepDocuments.index)]
            return { "documents": this_step_papers}

            # Send 
        

async def handler(websocket,path):
    print('Server Running ...')
    server.reset()
    step_count = 0;
    data_zotero = server.Zotero('getCollections',False,False,False)
    await websocket.send(json.dumps({
        "message" : data_zotero,
        "type" : "collections"
    }))
    # pdb.set_trace()
    consumer_task = asyncio.ensure_future(handle_message(websocket,path,server,step_count))
    done, pending = await asyncio.wait(
        [consumer_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
            task.cancel()

async def handle_message(websocket, path,server,step_count):
    # #HardCoded Collections for User Study
    # collection1 = 'DNUCZS2H'
    # collection2 = '5WYEJX3C'
    # collection3 = 'QFD2CSZY'
    # collection4 = 'FCKVGI5M'
    # collection5 = 'VGWP6JWE'

    #HardCoded Collections for User Study
    collection1 = 'VGWP6JWE'
    collection2 = 'FCKVGI5M'
    collection3 = 'QFD2CSZY'
    collection4 = '5WYEJX3C'
    collection5 = 'DNUCZS2H'

    # collection1 = 'B3RFB6H9'
    # collection2 ='AVUHSXNZ'
    # collection3 = '3JZ34V2D'

    #Retrieve Collection Elemets Zotero
    data_zotero = server.Zotero("getCollectionItems",collection1,False,False)
    collection_items = json.loads(data_zotero)
    # Load Pdfs
    pdf_ids = []
    for  i in range(0,len(collection_items)):
        pdf_ids.append(collection_items[i]['pdf_file'])
    #Process Papers in Server
    server.Zotero("downloadItems",False,pdf_ids,collection_items)
    #send Papers to Client
    message = server.DocInterface(False,'0','zoteroCollection',collection_items,'training')
    await websocket.send(json.dumps({
        # "message": {'documents':message['documents'].to_json(),'doc_topics':{'topics':message['doc_topics']['topics'].to_json(),'topic_params':message['doc_topics']['topic_params'].to_json(), 'order': message['doc_topics']['order'], 'words':message['doc_topics']['words']},'years':message['years'].to_json(),'authors':message['authors'].to_json()},
        "message": {'documents':message['documents'].to_json(),'doc_topics':{'topics':"message['doc_topics']['topics'].to_json()",'topic_params':"message['doc_topics']['topic_params'].to_json()", 'order': message['doc_topics']['order'], 'words':message['doc_topics']['words']},'years':message['years'].to_json(),'authors':message['authors'].to_json()},
        "type" : "sageBrain_data"
    }))
    while True:
        message = await websocket.recv()
        json_message = json.loads(message)
        if json_message['type']=='add_next_documents_for_user_studies':
            print('Add next documents for user studies')
            # pdb.set_trace()
            step_count = step_count + 1
            if step_count == 1:
                data_zotero = server.Zotero("getCollectionItems",collection2,False,False)
            elif step_count == 2:
                data_zotero = server.Zotero("getCollectionItems",collection3,False,False)
            elif step_count == 3:
                data_zotero = server.Zotero("getCollectionItems",collection4,False,False)
            elif step_count == 4:
                data_zotero = server.Zotero("getCollectionItems",collection5,False,False)
            collection_items = json.loads(data_zotero)
            pdf_ids = []
            for  i in range(0,len(collection_items)):
                pdf_ids.append(collection_items[i]['pdf_file'])
            #Download papers
            server.Zotero("downloadItems",False,pdf_ids,collection_items)
            message = server.DocInterface(False,'0','zoteroCollection',collection_items,'comparison')
            await websocket.send(json.dumps({
                "message": {'documents':message['documents'].to_json()},
                "type" : "Comparison_data"
            }))
        elif json_message['type']=='update_model':
            # pdb.set_trace()
            #TOPICS AND WORDS DONT CHANGE SO WE JUST PASS THEM BACK
            new_paper_data = pd.DataFrame(json_message['msg']['papers'])
            # pdb.set_trace()
            documents = server.UpdateModel(new_paper_data,step_count)
            # pdb.set_trace()
            await websocket.send(json.dumps({
                "message": {'documents':documents.to_json()},
                "type" : "update_model"
            }))
        elif json_message['type']=='next_cicle':
            # pdb.set_trace()
            message = server.send_current('training')
            await websocket.send(json.dumps({
                # "message": {'documents':message['documents'].to_json(),'doc_topics':{'topics':message['doc_topics']['topics'].to_json(),'topic_params':message['doc_topics']['topic_params'].to_json(), 'order': message['doc_topics']['order'], 'words':message['doc_topics']['words']},'years':message['years'].to_json(),'authors':message['authors'].to_json()},
                "message": {'documents':message['documents'].to_json(),'doc_topics':{'topics':"message['doc_topics']['topics'].to_json()",'topic_params':"message['doc_topics']['topic_params'].to_json()", 'order': message['doc_topics']['order'], 'words':message['doc_topics']['words']},'years':message['years'].to_json(),'authors':message['authors'].to_json()},
                "type" : "sageBrain_data"
            }))

def main():
    start_server = websockets.serve(handler, "0.0.0.0", 3000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

server = SageBrain('sess5')
#we use this for user inividual steps
step_count = 0

#start process
if __name__ == "__main__":
    main()