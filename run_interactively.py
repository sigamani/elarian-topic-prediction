from deeppavlov import build_model
import sys
import nltk
import os
import pandas as pd

CWD = os.getcwd()
CONFIG = CWD + '/config.json'
MODEL = build_model(CONFIG, download=False, load_trained=True)

def run_bulk():
    data = pd.read_csv("data/valid.csv") 
    for index, row in data.iterrows():
        message = row['x']  
        truth   = row['y'] 
        predictions = MODEL([message])                                                       

        print(f"{message} | {predictions[1][0]} | {truth}")


def run_interactively():
    mappings = load_class_mappings()
    while True:
        try:
            sys.stdout.write('>>')
            message = input()
            if message == 'exit':
                break
            if greater_than_length_threshold(message):
                predictions = MODEL([message])
                topic_thresholds = map_labels_to_threshold(predictions, mappings)
                print(topic_thresholds)

        except KeyboardInterrupt:
            break

def map_labels_to_threshold(predictions, mappings):
    topic_thresholds = {}
    labels = predictions[1][0]
    predictions_matrix = predictions[0][0]
    if len(labels) > 0:
        for l in labels:
            index = mappings.index(l)
            print(predictions_matrix)
            probability = predictions_matrix[mappings.index(l)]
            topic_thresholds[l] = probability
    return topic_thresholds

def greater_than_length_threshold(message):
    tokens = nltk.word_tokenize(message)
    if len(tokens) > 2:return True
    return False

def load_class_mappings():
    classes_dir = CWD.replace('/app','') + '/tf_model/classes.dict'
    mappings = [line.rstrip('\n') for line in open(classes_dir)] 
    mappings = [line.split()[0] for line in mappings]
    return mappings
   
if __name__ == '__main__':
    run_interactively()
   # run_bulk()
