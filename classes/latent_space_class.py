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

def get_contrastive_data(datax, datay, n):
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
            label, datay1, datay2
        ))
    return results

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=-1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def euclidean_distance_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def build_contrastive_model(input_shape, class_dim):
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
        euclidean_distance, 
        output_shape=euclidean_distance_shape,
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

    model = Model(inputs=[X1, X2], outputs=[Y, YClass1, YClass2])
    model.summary()
    return model

# Capture the embeddings from the trained model layers
def get_embedding_model(model, input_shape):
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

    result = Model(inputs=X, outputs=[Y, Class])
    result.summary()
    return result

def as_cat(target, n_class):
    result = np.zeros(n_class)
    result[target] = 1
    return result

def to_categorical(targets, n_class):
    return np.array([as_cat(target, n_class) for target in targets])

# Get inputs and outpus from contrastive data
def get_inputs_and_outputs(contrastive_data):
    dx1 = [x1 for (x1,_,_,_,_) in contrastive_data]
    dx2 = [x2 for (_,x2,_,_,_) in contrastive_data]
    dy =  [y  for (_,_,y,_,_)  in contrastive_data]
    dycat1 = [ycat1 for (_,_,_,ycat1,_) in contrastive_data]
    dycat2 = [ycat2 for (_,_,_,_,ycat2) in contrastive_data]
    return dx1, dx2, dy, dycat1, dycat2


    