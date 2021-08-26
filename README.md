Zexplorer SAGEBRAIN (User Study)
=================================

This is the zEplorer Module for SAGEBRAIN.
------------
Project Organization
------------

    ├── sageBrain.py                <- Main file. Server code. Need to change Zotero to point to your own zotero collection by editing variable self.Zotero
    ├── classes
    │   ├── author_class.py      
    │   ├── document_class.py       <- Document object parameters and functions.
    │   ├── encoder_class.py        <- Encoder model class. (Not used in user study. I user study we used latent_space_class.py see below)
    │   ├── session_class.py        <- Session object parameters and fuctions.
    │   ├── topic_class.py 
    │   ├── zotero_class.py         <- Class to search and retrieve information from a zotero collection.
    │   ├── latent_space_class.py   <- Contrastive model class. Tensorflow. Keras.
    │   └── nsaSrc
    |   │   ├── data      
    │   │   │   ├── external        <- Need to go to this directory and put the file downloaded from: http://magnitude.plasticity.ai/fasttext+approx/wiki-news-300d-1M.magnitude
    │   │   │   ├── extracted
    │   │   │   ├── interim
    │   │   │   ├── processed
    │   │   │   └── raw
    |   │   ├── models

------------
