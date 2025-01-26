import pandas as pd
import requests
from pathlib import Path

def get_sample_predictions(data_path: Path, predict_url: str) -> pd.DataFrame:
    # Load the data
    data = pd.read_csv(data_path)
    
    # Get the sample data (removes NaN and takes a random sample of 5 rows)
    sample_data = data.dropna().sample(1)
    
    # Convert the sample data into a JSON-serializable format (list of dictionaries)
    data_dict = sample_data.drop(columns=[sample_data.columns.tolist()[-1]]).squeeze().to_dict()
    
    # Get the predictions from the API
    try:
        response = requests.post(url=predict_url, json=data_dict)
        response.status_code  # Raise an error for bad status codes (e.g., 404, 500)
        sample_data["predictions"] = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {predict_url}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
       
    return sample_data

# Define paths
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"
predict_url = "http://127.0.0.1:8000/predict"

# Get sample raw data with predictions
sample_raw_data = get_sample_predictions(data_path, predict_url)

print(sample_raw_data.iloc[:, -2:])  # Display the last two columns of the sample data
