import importlib
import torch.utils.data
from datasets.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_argsion_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_argsions

def create_dataset(args):
    data_loader = CustomDatasetDataLoader(args)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, args):
        self.args = args
        dataset_class = find_dataset_using_name(args.dataset_mode)
        self.dataset = dataset_class(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=not args.serial_batches,
            num_workers=int(args.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
