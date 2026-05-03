import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = './'
path = kagglehub.dataset_download("shubhamsingh42/flight-delay-dataset-2018-2024")

print("Path to dataset files:", path)