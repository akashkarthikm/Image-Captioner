import string
import numpy as np
import pandas as pd
from PIL import Image
from pickle import dump, load

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model, load_model
from tensorflow.keras.applications.resnet import ResNet101
from keras.applications.vgg16 import VGG16
from tensorflow.keras import models as kModelsimport 

home = './'

# Loading a text file into memory
def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_doc_bin(filename):
    # Opening the file as read only
    file = open(filename, 'rb')
    text = load(file)
    file.close()
    return text

# get all imgs with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    # print(captions[3])
    a=1
    descriptions ={}
    # print(captions[1].split(":"))
    for caption in captions[1:]:
        # print(caption.split(','))
        # print(caption)
        a+=1
        cap=caption.split(":")
        img = cap[0].strip()
        caption_ = cap[1].strip()
        # print(img)
        if img not in descriptions:
            descriptions[img] = [caption_]
        else:
            descriptions[img].append(caption_)
    # print(descriptions)
    return descriptions

##Data cleaning- lower casing, removing puntuations and words containing numbers
import string
def cleaning_text(captions):
    table = str.maketrans('','', string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):

            img_caption.replace("-"," ")
            desc = img_caption.split()

            #converts to lower case
            desc = [word.lower() for word in desc]
            #remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            #remove hanging 's and a 
            desc = [word for word in desc if(len(word)>1)]
            #remove tokens with numbers in them
            desc = [word for word in desc if(word.isalpha())]
            #convert back to string

            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    # print(list(captions.values())[0])
    return captions

def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()
    
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    
    # print(vocab)
    return vocab

#All descriptions in one file 
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc )
    data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()

filename = home + "captions_8k.txt"
descriptions = all_img_captions(filename)

with open(home + 'descDict.p', 'wb') as descDict:
    dump(descriptions, descDict)


print("Length of descriptions =" ,len(descriptions))
clean_descriptions = cleaning_text(descriptions)
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
save_descriptions(clean_descriptions, home + "descriptions.txt")

# from tensorflow
# resnetModel = ResNet101(
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
# # resnet.summary()

# resnetModel.save(home + 'resnetModel.h5')

vggModel = VGG16()
vggModel = Model(inputs = vggModel.inputs, outputs = vggModel.layers[-2].output)
vggModel.save(home + 'vggModel.h5')


def extract_features(directory):
        features = {}
        img_dir = glob(directory + '\*.jpg')
        # print(len(img_dir))
        for image in img_dir:
            filename = image

            # image = cv2.imread(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(filename)
            image = image.resize((224,224))
            image = np.expand_dims(image, axis=0)
            # print(image.shape)
            #image = preprocess_input(image)
            # image = image/127.5
            # image = image - 1.0

            filename = filename.split('\\')[1]
            
            # print(image.shape)
            feature = resnet.predict(image)
            # print(feature.shape)
            features[filename] = feature
        return features

# ## 2048 feature vector
def dumpFeatureVectors():
    features = extract_features('./train2')
    dump(features, open(home + "mod_train_8K_6500.p","wb"))

def loadFeatureVectors():
    features = load(open("mod_train_8K_6500.p","rb"))
    print(len(features))
    return features

# features_3000 = load_doc_bin('features4k.p')
# features = {k:features_3000[k] for k in list(features_3000)[:2]}

# with open('features_10k.p', 'wb') as feat_10k:
#     dump(features, feat_10k)

# f_7000 = load(open('features_10k.p', 'rb'))

# with open('features_10k.p', 'wb') as feat_10k:
#     dump({**features , **features_3000}, feat_10k)

# print(len(load(open('features_10k.p', 'rb'))))

#load the data  
def load_photos(filename): 
    file = load_doc(filename) 
    photos = file.split("\n")[:-1] 
    return photos 
 
 
def load_clean_descriptions(filename):    
    #loading clean_descriptions 
    file = load_doc(filename) 
    descriptions = {} 
    for line in file.split("\n"):         
        words = line.split() 
        if len(words)<1 : 
            continue      
        image, image_caption = words[0], words[1:]          
        if image not in train_features.keys(): 
            continue  
        if image not in descriptions: 
            descriptions[image] = [] 
        desc = '<start> ' + " ".join(image_caption) + ' <end>' 
        descriptions[image].append(desc) 
    return descriptions 
 
def load_features(): 
    #loading all features 
    all_features = load_doc_bin(home + "train_8K_6500.p") 
    return all_features

# filename = dataset_text + "/" + "Flickr_8k.trainImages.txt" 
 
# train = loading_data(filename) 
# train_imgs = load_photos(filename) 
train_features = load_features() 
train_descriptions = load_clean_descriptions(home + "descriptions.txt") 
features = train_features
len(features)

#converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

#creating tokenizer class 
#this will vectorise text corpus
#each integer will represent token in dictionary 


def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

# give each word a index, and store that into tokenizer.p pickle file
tokenizer = load_doc_bin(home + 'tokenizer.p')
vocab_size = len(tokenizer.word_index) + 1
vocab_size
# give each word a index, and store that into tokenizer.p pickle file


#calculate maximum length of descriptions
def max_length_func(descriptions):
    desc_list = dict_to_list(descriptions)
    max_ = 0
    for d in desc_list:
        if len(d.split()) > max_:
            max_ = len(d.split())
            # print(max_, d)
    return max_

max_length = max_length_func(descriptions)
max_length
#((47, 2048), (47, 32), (47, 7577))

# train our model
# print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)

resnet = kModels.load_model(home + 'vggModel.h5')
model = kModels.load_model(home + 'model_20.h5')

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def generate_feature(filename):
    image = Image.open(filename)
    image = image.resize((224,224))
    image = np.expand_dims(image, axis=0)
    return resnet.predict(image)

def predictUsingImage(filename):
    # filename = "./mod_testdata/3082934678_58534e9d2c.jpg"
    description = generate_desc(model, tokenizer, generate_feature(filename), max_length)
    description = description[6:-3].capitalize()
    print(description)
    return description