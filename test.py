import requests
from datasets import load_dataset

if __name__ == "__main__":
  # download
  dataset = load_dataset("allenai/cosmos_qa", trust_remote_code=True)
  dataset.save_to_disk('cosmos_qa')