from abc import ABC, abstractmethod
import boto3
import os

base_ext_data_path = "/home/jovyan/external"

class AbstractDatasetManager(ABC):
    @abstractmethod
    def get_file(self, file_id):
        pass



class S3DatasetManager(AbstractDatasetManager):
    def get_file(self, file_id):
        path = os.path.join(base_ext_data_path, file_id)
        f = open(path, 'r')
        return f

def dataset_manager(opts = {}):
    return S3DatasetManager()
