import numpy as np
import h5py

dataset_name ="BLA"
a = np.array([1,2,3])
b = np.array([1,2])

h5f = h5py.File(data_file_path,'r')[dataset_name]
h5f = h5py.File(data_file_path,'a')
	h5f.create_dataset(dataset_name,user_shape,maxshape = (None,time_steps,number_of_products,total_features),chunks = user_shape,compression = "gzip",compression_opts = 9)