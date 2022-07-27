This folder is for generating simulated training data and data for quantifying the estimation precision and accuracy

===code: generate_image_trispot.m==============
generate training data


====code: train_test_set_processing==========
combine the data generated from 'generate_image_pol.m' into larger .mat file. 
The output of this code will be used for training

====code: generate_validation_data_trispot.m=======
generate data for quantifying the estimation precision and accuracy


=====code: generate_validation_trispot_large_dataset.m=====
generate training data similar as code 'generate_image_trispot.m', but the saving method is different. The saved data will be used for code 'Training_run_large_dataset_tripost.py' in NN_code folder
