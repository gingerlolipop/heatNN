# weather_downloader.py
import requests
import pandas as pd
import datetime
from pathlib import Path
from typing import Optional
import os
from bs4 import BeautifulSoup
import re
import time
import chardet

class WeatherDataLoader:
    def __init__(self, base_url: str = "https://dd.weather.gc.ca/climate/observations/daily/csv/BC/",
                 data_dir: str = "ECCC"):
        self.base_url = base_url
        self.data_dir = Path(data_dir)
        # Create ECCC directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
    def list_available_files(self) -> list:
        """List available CSV files from the webpage."""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            
            # Parse HTML properly using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links that end with _P1D.csv and start with climate_daily_BC_
            csv_pattern = re.compile(r'^climate_daily_BC_.*_P1D\.csv$')
            files = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if csv_pattern.match(href):
                    files.append(href)
            
            # Sort files to process them in order
            files.sort()
            print(f"Found {len(files)} valid CSV files")
            return files
        
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def detect_encoding(self, file_path: Path) -> str:
        """Detect the encoding of a file using chardet."""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    
    def download_file(self, filename: str, retry_count: int = 3) -> bool:
        """Download a specific file with retries."""
        local_path = self.data_dir / filename
        url = f"{self.base_url}{filename}"
        
        for attempt in range(retry_count):
            try:
                response = requests.get(url)
                response.raise_for_status()
                # Try to decode content first to check encoding
                content = response.content.decode('latin-1')  # Using latin-1 for initial download
                with open(local_path, 'w', encoding='latin-1') as f:
                    f.write(content)
                print(f"Successfully downloaded: {filename}")
                return True
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")
                    time.sleep(1)  # Wait a second before retrying
                else:
                    print(f"Error downloading {filename} after {retry_count} attempts: {e}")
                    return False

    def download_files(self, start_idx: int = 0, num_files: int = 5):
        """Download a specific number of files starting from start_idx."""
        files = self.list_available_files()
        
        if not files:
            print("No files found to download")
            return
            
        end_idx = min(start_idx + num_files, len(files))
        files_to_download = files[start_idx:end_idx]
        
        print(f"Attempting to download files {start_idx} to {end_idx-1}")
        
        for i, filename in enumerate(files_to_download, 1):
            local_path = self.data_dir / filename
            if not local_path.exists():
                print(f"Downloading file {i}/{len(files_to_download)}: {filename}")
                self.download_file(filename)
            else:
                print(f"File already exists: {filename}")

    def read_csv_with_encoding(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Try to read CSV file with different encodings."""
        encodings = ['latin-1', 'utf-8', 'utf-16', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {file_path.name} with {encoding} encoding: {e}")
                continue
        
        print(f"Failed to read {file_path.name} with any encoding")
        return None

    def combine_downloaded_files(self) -> Optional[pd.DataFrame]:
        """Combine all downloaded CSV files into a single DataFrame."""
        print("Combining downloaded files...")
        dfs = []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and 'combined' not in f]
        total_files = len(csv_files)
        
        if total_files == 0:
            print("No CSV files found in the directory")
            return None
        
        for i, file in enumerate(csv_files, 1):
            try:
                file_path = self.data_dir / file
                print(f"Processing file {i}/{total_files}: {file}")
                
                df = self.read_csv_with_encoding(file_path)
                if df is not None:
                    dfs.append(df)
                    print(f"Successfully read {file}")
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if not dfs:
            print("No files were successfully processed")
            return None
            
        print("Concatenating all dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save combined dataset
        combined_path = self.data_dir / "combined_all_stations.csv"
        print(f"Saving combined dataset to: {combined_path}")
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')
        
        return combined_df

if __name__ == "__main__":
    # Example usage
    loader = WeatherDataLoader()
    
    print("Starting weather data download process...")
    
    # Download first 5 files for testing
    loader.download_files(start_idx=0, num_files=5)
    
    print("\nCombining downloaded files...")
    combined_data = loader.combine_downloaded_files()
    
    if combined_data is not None:
        print(f"\nSuccessfully combined data. Shape: {combined_data.shape}")
        print("\nColumns in the dataset:")
        print(combined_data.columns.tolist())
        print("\nFirst few rows:")
        print(combined_data.head())