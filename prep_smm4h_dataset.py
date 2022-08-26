import pandas as pd
import math
import random

random.seed(2)


if __name__ == '__main__':
    source_file_path = "data/seq_labeling/raw/smm4h_full_raw.csv"
    train_out_file_path = "data/seq_labeling/raw/train/smm4h_75_25_train"
    test_out_file_path = "data/seq_labeling/raw/test/smm4h_75_25_test"

    rows = pd.read_csv(source_file_path, na_filter=False)

    new_rows = []
    test_percent = .25
    train_percent = 1 - test_percent

    for index, row in rows.iterrows():

        if len(row['drug']) > 0:
            new_rows.append(row.to_dict())

    all_indexes = range(0, len(new_rows))
    test_indices = random.sample(all_indexes, math.ceil(len(new_rows) * test_percent))
    test_rows = [new_rows[index] for index in test_indices]
    test_data = pd.DataFrame(test_rows, columns=list(rows))

    train_indices = [index for index in all_indexes if index not in test_indices]
    train_rows = [new_rows[index] for index in train_indices]
    training_data = pd.DataFrame(train_rows, columns=list(rows))

    training_data.to_csv(train_out_file_path, index=False)
    test_data.to_csv(test_out_file_path, index=False)


    # Generate smaller train dataset for evaluation. The test set remains same.
    # This part is only for the evaluation on the smaller training datasets
    train_all_indices = range(0, len(train_rows))

    train_percents = [.25, .50, .75] # 25%, 50% and 75% of the training data

    for train_percent in train_percents:
        small_train_output_file = "data/seq_labeling/raw/train/smm4h_"+ str(train_percent)[2:] +"p_small_train"
        small_train_indices = random.sample(train_all_indices, math.ceil(len(train_rows) * train_percent))
        small_train_rows = [train_rows[index] for index in small_train_indices]
        small_train_data = pd.DataFrame(small_train_rows, columns=list(rows))
        small_train_data.to_csv(small_train_output_file, index=False)





