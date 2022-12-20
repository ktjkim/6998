from google.cloud import storage
from constants import *


def download_model_from_gcs():
    # Initialise a client
    storage_client = storage.Client(PROJECT)
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(BUCKET)
    # Create a blob object from the filepath
    blob = bucket.blob(MODEL_PATH)
    # Download the file to a destination
    blob.download_to_filename(DESTINATION_PATH)