import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from pathlib import Path
from typing import List, Optional
import datetime
from ws_download import WeatherDataLoader

class HeatwaveGNN(torch.nn.Module):
    def __init__(self, num_features: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)  # Binary classification
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def prepare_graph_data(df: pd.DataFrame, threshold_temp: float = 30.0) -> Data:
    """
    Prepare graph data structure from weather dataframe.
    A heatwave is defined as temperature exceeding threshold_temp.
    """
    # Extract relevant features
    features = df[['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 
                  'Heat Deg Days (°C)', 'Cool Deg Days (°C)', 
                  'Total Rain (mm)', 'Total Snow (cm)']].values
    
    # Create temporal edges (connecting consecutive days)
    edge_index = []
    for i in range(len(df) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])  # Bidirectional
    
    # Create labels (1 for heatwave days)
    labels = (df['Max Temp (°C)'] > threshold_temp).astype(float).values
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index).t()
    y = torch.FloatTensor(labels)
    
    return Data(x=x, edge_index=edge_index, y=y)

def train_model(model: HeatwaveGNN, data: Data, 
                num_epochs: int = 100) -> List[float]:
    """Train the GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    return losses

def save_model(model: HeatwaveGNN, save_dir: str = "ECCC"):
    """Save the trained model."""
    save_path = Path(save_dir) / "heatwave_gnn_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def load_model(model: HeatwaveGNN, save_dir: str = "ECCC"):
    """Load a trained model."""
    load_path = Path(save_dir) / "heatwave_gnn_model.pt"
    model.load_state_dict(torch.load(load_path))
    return model

if __name__ == "__main__":
    # Initialize data loader
    loader = WeatherDataLoader(data_dir="ECCC")
    
    # Fetch data for a specific station and time period
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    station_id = "1010066"  # Example station ID
    
    # Fetch and prepare data
    print(f"Fetching data for station {station_id} from {start_date} to {end_date}")
    df = loader.fetch_date_range(station_id, start_date, end_date)
    
    if df is not None:
        print("Preparing graph data...")
        graph_data = prepare_graph_data(df)
        
        # Initialize and train model
        print("Initializing and training model...")
        model = HeatwaveGNN(num_features=7, hidden_channels=64)
        losses = train_model(model, graph_data)
        
        # Save the trained model
        save_model(model)
        
        print("Training complete! Model saved in ECCC directory.")
class HeatwaveGNN(torch.nn.Module):
    def __init__(self, num_features: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)  # Binary classification
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def prepare_graph_data(df: pd.DataFrame, threshold_temp: float = 30.0) -> Data:
    """
    Prepare graph data structure from weather dataframe.
    A heatwave is defined as temperature exceeding threshold_temp.
    """
    # Extract relevant features
    features = df[['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 
                  'Heat Deg Days (°C)', 'Cool Deg Days (°C)', 
                  'Total Rain (mm)', 'Total Snow (cm)']].values
    
    # Create temporal edges (connecting consecutive days)
    edge_index = []
    for i in range(len(df) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])  # Bidirectional
    
    # Create labels (1 for heatwave days)
    labels = (df['Max Temp (°C)'] > threshold_temp).astype(float).values
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index).t()
    y = torch.FloatTensor(labels)
    
    return Data(x=x, edge_index=edge_index, y=y)

def train_model(model: HeatwaveGNN, data: Data, 
                num_epochs: int = 100) -> List[float]:
    """Train the GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    return losses

# Usage example:
if __name__ == "__main__":
    # Initialize data loader
    loader = WeatherDataLoader()
    
    # Fetch data for a specific station and time period
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)
    station_id = "1010066"  # Example station ID
    
    # Fetch and prepare data
    df = loader.fetch_date_range(station_id, start_date, end_date)
    if df is not None:
        # Prepare graph data
        graph_data = prepare_graph_data(df)
        
        # Initialize and train model
        model = HeatwaveGNN(num_features=7, hidden_channels=64)
        losses = train_model(model, graph_data)
        
        print("Training complete!")