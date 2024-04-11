import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
import multiprocessing
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
import torch
import transformers
from pydantic import BaseModel
from typing import Dict
import os
import time
import string
from fastapi.staticfiles import StaticFiles

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
sms_data = pd.read_csv(url, sep='\t', names=['label', 'message'])

url = "text_classification_dataset.csv"
news_data = pd.read_csv(url, encoding='latin1')
news_data = news_data[['text', 'type']]
news_data.columns = ['message', 'label'] 

url = "Training_Essay_Data.csv"
ai_generated_data = pd.read_csv(url)
ai_generated_data = ai_generated_data[['text', 'generated']]
ai_generated_data.columns = ['message', 'label'] 

url = "emails.csv"
email_spam_data = pd.read_csv(url)
email_spam_data = email_spam_data[['text', 'spam']]
email_spam_data.columns = ['message', 'label'] 

reserved_tokens = ['happy', 'hi', 'see']

augmenters = {
    'deletion': {'augmenter': naw.RandomWordAug(action="delete"), 'aug_p': 0.5},
    'swap': {'augmenter': naw.RandomWordAug(action="swap"), 'aug_p': 0.5},
    'crop': {'augmenter': naw.RandomWordAug(action="crop"), 'aug_p': 0.5},
    'spelling': {'augmenter': naw.SpellingAug(), 'aug_p': 0.5},
    'split': {'augmenter': naw.SplitAug(), 'aug_p': 0.5},
    'tfidf': {'augmenter': naw.TfIdfAug(action="insert"), 'aug_p': 0.5},
    'insert': {'augmenter': nac.RandomCharAug(action='insert'), 'aug_p': 0.5},
    'substitute': {'augmenter': nac.RandomCharAug(action='substitute'), 'aug_p': 0.5},
    'delete': {'augmenter': nac.RandomCharAug(action='delete'), 'aug_p': 0.5},
    'abstractive': {'augmenter': nas.AbstSummAug(), 'aug_p': 0.5},
    'reserved' : {'augmenter' : naw.ReservedAug(reserved_tokens=reserved_tokens), 'aug_p': 0.5} 
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="storage"), name="static")

clf = MultinomialNB()
vectorizer = CountVectorizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

def augment_data(data, augmenter, num_samples):
    augmented_data = []
    for text in data:
        try:
            augmented_texts = augmenter.augment(text, n=num_samples)
            augmented_data.extend(augmented_texts)
        except Exception as e:
            print(e)
    return augmented_data

def process_batch(batch, augmenters_list):
    augmented_data_list = [] 
    for augmenter_name, augmenter_info in augmenters_list.items():
        augmenter = augmenter_info['augmenter']
        aug_p = augmenter_info['aug_p']
        augmented_texts = augment_data(batch['message'], augmenter, 5)
        augmented_data_list.extend([
            {
                'label': label,
                'message': text
            }
            for label, text in zip(batch['label'], augmented_texts)
        ])
        augmented_batch = pd.DataFrame(augmented_data_list)
    return augmented_batch

class AugmentationConfig(BaseModel):
    deletion: bool = False
    swap: bool = False
    crop: bool = False
    spelling: bool = False
    split: bool = False
    tfidf: bool = False
    insert: bool = False
    substitute: bool = False
    delete: bool = False
    abstractive: bool = False
    reserved: bool = False

class QLearningAgent:
    def __init__(self, augmenters, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.augmenters = augmenters
        self.q_values = {augmenter: 0.5 for augmenter in augmenters}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.prev_action = None
        self.prev_reward = None

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice(list(self.augmenters.keys()))
        else:  # Exploitation
            return max(self.q_values, key=self.q_values.get)

    def update(self, action, reward):
        if self.prev_action is not None:
            prev_value = self.q_values[self.prev_action]
            best_next_action = max(self.q_values, key=self.q_values.get)
            new_value = prev_value + self.alpha * (reward + self.gamma * self.q_values[best_next_action] - prev_value)
            self.q_values[self.prev_action] = new_value
        self.prev_action = action
        self.prev_reward = reward

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_augmenter_probs(self):
        for augmenter_name, augmenter_info in self.augmenters.items():
            augmenter_info["aug_p"] = self.q_values[augmenter_name]

def train_and_evaluate(X_train, X_test, y_train, y_test):
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    clf.fit(X_train_counts, y_train)
    predictions = clf.predict(X_test_counts)

    return accuracy_score(y_test, predictions)

def process_batch_partial(args):
    batch, augmenters_list = args
    return process_batch(batch, augmenters_list)

def aug_dataset(dataset, augmenters):
    batch_size = 500
    num_batches = len(dataset) // batch_size + 1
    batches = np.array_split(dataset, num_batches)
    pool = multiprocessing.Pool()
    augmented_batches = pool.map(process_batch_partial, [(batch, augmenters) for batch in batches])
    pool.close()
    pool.join()
    augmented_data = pd.concat(augmented_batches, ignore_index=True)
    return augmented_data

def train_model(selected_augmenters,dataset):

    agent = QLearningAgent(selected_augmenters)
    num_episodes = 10
    augmented_dataset = None
    X_train, X_test, y_train, y_test = train_test_split(dataset['message'], dataset['label'], test_size=0.2, random_state=42)

    print("Training model auto \n")
    last_accuracy = None
    # Main training loop
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        print(selected_augmenters)
        augmented_dataset = aug_dataset(dataset,selected_augmenters)
        X_train, X_test, y_train, y_test = train_test_split(augmented_dataset['message'], augmented_dataset['label'], test_size=0.2, random_state=42)
        accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
        agent.update(agent.choose_action(), accuracy)
        agent.update_augmenter_probs()
        agent.decay_epsilon()
        last_accuracy = accuracy

    best_params = {augmenter: agent.q_values[augmenter] for augmenter in selected_augmenters}
    print("Best Parameters:")
    print(best_params)

    timestamp = int(time.time())
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    filename = f"augmented_data_{timestamp}_{random_str}.csv"
    file_path = f"storage/{filename}"
    augmented_dataset.to_csv(file_path, index=False)

    X_test_counts = vectorizer.transform(X_test)
    predictions = clf.predict(X_test_counts)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    baseURL = "http://127.0.0.1:8000"

    return {
        "status": "Training completed",
        "best_params": best_params,
        "accuracy": last_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "csvlink": f"{baseURL}/static/{filename}" 
    }

@app.get("/")
def read_root():
    return {"message": "AutoDA_service_is_running"}

@app.post("/model")
def model_training(config: AugmentationConfig):
    selected_augmenters = {key: augmenters[key] for key, value in config.dict().items() if value}

    if not selected_augmenters:
        raise HTTPException(status_code=400, detail="At least one augmentation method must be selected.")
    
    return train_model(selected_augmenters,sms_data)


@app.post("/model2")
def model_training(config: AugmentationConfig):
    selected_augmenters = {key: augmenters[key] for key, value in config.dict().items() if value}

    if not selected_augmenters:
        raise HTTPException(status_code=400, detail="At least one augmentation method must be selected.")
    
    return train_model(selected_augmenters,news_data)

@app.post("/model3")
def model_training(config: AugmentationConfig):
    selected_augmenters = {key: augmenters[key] for key, value in config.dict().items() if value}

    if not selected_augmenters:
        raise HTTPException(status_code=400, detail="At least one augmentation method must be selected.")
    
    return train_model(selected_augmenters,ai_generated_data)

@app.post("/model4")
def model_training(config: AugmentationConfig):
    selected_augmenters = {key: augmenters[key] for key, value in config.dict().items() if value}

    if not selected_augmenters:
        raise HTTPException(status_code=400, detail="At least one augmentation method must be selected.")
    
    return train_model(selected_augmenters,email_spam_data)
