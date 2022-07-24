import os 
import pandas as pd

def metadata(data_dir):
    data_root = data_dir
    ds_external = os.path.join(data_root, "external")
    patient = pd.read_csv(os.path.join(ds_external, 'HBP_Descriptives.csv'),delimiter=";",decimal = ',')
    patient.set_index("ID_HBP", inplace=True)
    return patient