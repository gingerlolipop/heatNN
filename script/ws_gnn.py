# heatwave_gnn.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from pathlib import Path
from typing import List, Optional
import datetime

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

class WeatherGNNProcessor:
    def __init__(self, data_dir: str = "ECCC"):
        self.data_dir = Path(data_dir)
        self.model = None
        self.scaler = None
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load the combined dataset."""
        data_path = self.data_dir / "combined_all_stations.csv"
        if not data_path.exists():
            print(f"Combined dataset not found at {data_path}")
            return None
            
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            print(f"Error reading combined dataset: {e}")
            return None
        
    def prepare_graph_data(self, df: pd.DataFrame, threshold_temp: float = 30.0) -> Data:
        """
        Prepare graph data structure from weather dataframe.
        A heatwave is defined as temperature exceeding threshold_temp.
        """
        # Extract relevant features
        feature_columns = [
            'Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)',
            'Heat Deg Days (°C)', 'Cool Deg Days (°C)',
            'Total Rain (mm)', 'Total Snow (cm)'
        ]
        
        features = df[feature_columns].values
        
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

    def train_model(self, data: Data, num_epochs: int = 100) -> List[float]:
        """Train the GNN model."""
        self.model = HeatwaveGNN(num_features=data.x.size(1), hidden_channels=64)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        losses = []
        self.model.train()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
        return losses

    def evaluate_model(self, data: Data) -> dict:
        """Evaluate the model performance."""
        if self.model is None:
            print("No model to evaluate")
            return {}
            
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data.x, data.edge_index)
            predictions = (predictions.squeeze() > 0.5).float()
            
            # Calculate metrics
            correct = (predictions == data.y).sum().item()
            total = len(data.y)
            accuracy = correct / total
            
            # Calculate precision, recall, f1 for heatwave detection
            true_positives = ((predictions == 1) & (data.y == 1)).sum().item()
            false_positives = ((predictions == 1) & (data.y == 0)).sum().item()
            false_negatives = ((predictions == 0) & (data.y == 1)).sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    def save_model(self, filename: str = "heatwave_gnn_model.pt"):
        """Save the trained model."""
        if self.model is None:
            print("No model to save")
            return
        
        save_path = self.data_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, filename: str = "heatwave_gnn_model.pt"):
        """Load a trained model."""
        load_path = self.data_dir / filename
        if not load_path.exists():
            print(f"Model file not found: {load_path}")
            return
        
        if self.model is None:
            print("Initialize model first with correct number of features")
            return
            
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

def main():
    # Initialize processor
    gnn_processor = WeatherGNNProcessor()
    
    # Load data
    print("Loading weather data...")
    data = gnn_processor.load_data()
    
    if data is not None:
        print(f"\nPreparing graph data...")
        graph_data = gnn_processor.prepare_graph_data(data)
        
        print("\nTraining model...")
        losses = gnn_processor.train_model(graph_data)
        
        print("\nEvaluating model...")
        metrics = gnn_processor.evaluate_model(graph_data)
        print(f"Model performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nSaving model...")
        gnn_processor.save_model()
        
        print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    main()