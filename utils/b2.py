import mimetypes
import pandas as pd
import boto3
from botocore.config import Config


class B2(object):
    def __init__(self, endpoint, key_id, secret_key):
        """
        Set up a connection between the current instance and Backblaze.

        Parameters
        ----------
        endpoint : str
            The endpoint, usually starting with "https://s3..."
        key_id : str
            The "Key ID" for the application key from Backblaze.
        secret_key : str
            The Key secret, or "Key" for the Backblaze app key itself.
        """
        # Return a boto3 resource object for B2 service
        self.b2 = boto3.resource(
            service_name='s3',
            endpoint_url=f"https://{endpoint}",
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )
        
    def set_bucket(self, bucket_name):
        """
        Select a bucket accessible by the chosen app key.

        Parameters
        ----------
        bucket_name : str
            Name of Bucket
        """
        self.bucket = self.b2.Bucket(bucket_name)

    def list_files(self, verbose=False):
        """
        List all files in the bucket.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, return detailed file information, by default False.
        """
        if verbose:
            return [f.get() for f in self.bucket.objects.all()]
        else:
            return [f.key for f in self.bucket.objects.all()]

    def get_df(self, remote_path):
        """
        Get a CSV file from Backblaze as a DataFrame.

        Parameters
        ----------
        remote_path : str
            The path to the file in the Backblaze bucket.
        """
        # Get file
        obj = self.bucket.Object(remote_path)
        df = pd.read_csv(obj.get()['Body'])
        return df
    
    def get_object(self, remote_path):
        """
        Get an object from Backblaze bucket.

        Parameters
        ----------
        remote_path : str
            The path to the file in the Backblaze bucket.
        """
        obj = self.bucket.Object(remote_path)
        return obj.get()['Body']

    def file_to_b2(self, local_path, remote_path):
        """
        Upload a file to Backblaze.

        Parameters
        ----------
        local_path : str
            The path to the local file.
        remote_path : str
            The destination path in the bucket.
        """
        # Guess the type of a file based on its URL
        mimetype, _ = mimetypes.guess_type(local_path)

        if mimetype is None:
            raise Exception("Failed to guess mimetype")
        
        if remote_path in [f.key for f in self.bucket.objects.all()]:
            print(f'Overwriting {remote_path} ...')
        else:
            print(f'Uploading {remote_path} ...')
        
        self.bucket.upload_file(
            Filename=local_path,
            Key=remote_path,
            ExtraArgs={
                "ContentType": mimetype
            }
        )
