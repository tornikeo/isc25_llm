import requests
from datasets import load_dataset

if __name__ == "__main__":
  # download
  dataset = load_dataset("cosmos_qa")
  print(dataset)
  print(dataset['train'])