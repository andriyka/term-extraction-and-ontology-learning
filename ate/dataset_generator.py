import numpy as np
import ConfigParser
import os

from os import listdir
from os.path import isfile, join


def create_data_set(txt_source_dir, dataset_dest_key, sample_type = 'rev_chrono', all_files = False, sample_size = 30):
    file_names = sorted([os.path.join(txt_source_dir, f) for f in listdir(txt_source_dir) if isfile(join(txt_source_dir, f))
                          and f.lower().endswith(".txt")])
    sample_size = sample_size if not all_files else len(file_names)
    if sample_type == 'chrono':
        sample = file_names[:sample_size]
    elif sample_type == 'rev_chrono':
        sample = list(reversed(file_names[-sample_size:]))
    elif sample_type == 'bi_dir':
        inds = range(0, sample_size)
        sample = []
        for i in inds:
            if i % 2 == 0:
                sample.append(file_names[i])
            else:
                sample.append(file_names[-i])
    elif sample_type == 'rand':
        sample = np.random.choice(file_names, sample_size, replace = False)

    dataset_txt = ''
    for file_path in sample:

        print(file_path)

        with open(file_path, 'r') as txt_file:
            dataset_txt += txt_file.read()
            dataset_txt += os.linesep

    with open(dataset_dest_key + sample_type + '_' + str(sample_size) + '.txt',
              'w') as out_file:
        out_file.write(dataset_txt)




#if __name__ == '__main__':
    #create_data_set(sample_type = 'bi_dir', sample_size = 50)




