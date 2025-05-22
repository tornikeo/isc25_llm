import requests
from datasets import load_dataset

dataset = load_dataset("allenai/cosmos_qa", trust_remote_code=True)
dataset.save_to_disk('tmp-dataset')