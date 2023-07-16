from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
import requests
from geopy.geocoders import Nominatim

app = Flask(__name__)

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)


class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


input_size = len(training[0])
hidden_size = 8
output_size = len(output[0])

model = ChatbotModel(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train_model(model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = torch.tensor(training, dtype=torch.float32)
        targets = torch.tensor(np.argmax(output, axis=1), dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

train_model(model, optimizer, criterion, epochs=1000)

torch.save(model.state_dict(), "model.pt")
def generate_response(user_input):
    inp = user_input
    input_data = torch.tensor(bag_of_words(inp, words), dtype=torch.float32).unsqueeze(0)
    output = model(input_data)
    _, predicted = torch.max(output, dim=1)
    tag = labels[predicted.item()]

    if tag == "nearest_hospitals":
        try:
            hospitals = get_nearby_hospitals(location_str="Dublin, California, United States of America")
            
            if len(hospitals) > 0:
                response = random.choice(data["intents"][9]["responses"]).format(hospitals=", ".join(hospitals))
            else:
                response = "Sorry, no hospitals found nearby."
                
            return response
        except Exception as e:
            response = f"An error occurred while fetching nearby hospitals: {str(e)}"
            return response 
    else:
        for intent in data["intents"]:
            if intent['tag'] == tag:
                responses = intent['responses']
                break
        else:
            responses = []

        response = random.choice(responses)
        return response

def get_coordinates(location):
    geolocator = Nominatim(user_agent="myGeocoder")
    location = geolocator.geocode(location)
    return f"{location.latitude},{location.longitude}" if location else None

def get_nearby_hospitals(location_str):
    GOOGLE_MAPS_API_KEY = "AIzaSyCLCReLzKVF6W6MvJw6WwwcMtZj5in25TA"
    GOOGLE_MAPS_API_ENDPOINT = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    location = get_coordinates(location_str)

    if location:
        params = {
            "location": location,
            "radius": 5000,
            "type": "hospital",
            "key": GOOGLE_MAPS_API_KEY,
        }

        try:
            response = requests.get(GOOGLE_MAPS_API_ENDPOINT, params=params)
            data = response.json()

            hospitals = []
            if data.get("results"):
                for result in data["results"]:
                    hospitals.append(result["name"])

            return hospitals
        except Exception as e:
            print(f"Error fetching nearby hospitals: {str(e)}")
    else:
        print("Invalid location")



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']

    response = generate_response(user_input)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
