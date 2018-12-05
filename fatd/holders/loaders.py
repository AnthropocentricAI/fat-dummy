import fatd.holders
import numpy as np

def csv_loader(csv_path, header=None):
    data_matrix = np.loadtxt(csv_path, delimiter=',')

    if header is None:
        data = fatd.holders.Data(data_matrix[:,:-1],
                                 data_matrix[:,-1]
                                )
    else:
        data = fatd.holders.Data(data_matrix[:,:-1],
                                 data_matrix[:,-1],
                                 header[:-1],
                                 header[-1]
                                )

    return data
