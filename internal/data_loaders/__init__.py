from .levir_nvs import *
from .levir_spc_aug import *



dataset_dict = {
    'levir_nvs': LevirNVSDataset,
    'levir_spc_aug': LevirSPCAugDataset,
}