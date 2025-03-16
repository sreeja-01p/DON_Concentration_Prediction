from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import pickle

class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(128, 64)           
        self.dropout = nn.Dropout(0.3)           
        self.fc3 = nn.Linear(64, 32)             
        self.fc4 = nn.Linear(32, 1)              
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

app = Flask(__name__)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_dim = scaler.mean_.shape[0]
model = RegressionNN(input_dim)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    #JSON payload with new spectral data
    data = request.json
    #convert JSON data to DataFrame
    df = pd.DataFrame(data)
    #standardize features using the saved scaler
    df_scaled = scaler.transform(df)
    #convert the scaled DataFrame to a PyTorch tensor
    X = torch.tensor(df_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X).numpy().flatten().tolist()
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
