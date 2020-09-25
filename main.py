import os

import pandas as pd

import argparse
from dataset import Dataset
from models.pfp_model import PFPModel

# parser = argparse.ArgumentParser(description='pulmonary fibrosis')
# parser.add_argument('--', type=int, default=2020,
#                     help='re-produce the results with seed random')


def preprocessing(data):
    # Get Mex Percent
    data['MaxFvc'] = data['FVC'] / data['Percent'] * 100
    data.drop(['Percent'], axis=1, inplace=True)

    # min-max (MaxFvc, Age, FVC)
    fvc_min_max = (data['FVC'].min(), data['FVC'].max())
    weeks_min_max = (-5, 133)

    col_list = ['MaxFvc', 'Age', 'FVC', 'Weeks']
    for col in col_list:
        if col == 'Weeks':
            data[col] = (data[col] - weeks_min_max[0]) / (weeks_min_max[1] - weeks_min_max[0])
        else:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())



    # One-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'SmokingStatus'])
    return data, fvc_min_max, weeks_min_max


def create_test_input(x, train_data):
    patient, weeks = x.split('_')
    weeks = int(weeks)
    weeks_min_max = (-5, 133)
    converted_weeks = (weeks - weeks_min_max[0]) / (weeks_min_max[1] - weeks_min_max[0])

    patient_info_list = train_data[train_data['Patient'] == patient]
    patient_info = patient_info_list[patient_info_list['Weeks'] == converted_weeks]
    if len(patient_info) == 0:
        patient_info = patient_info_list.iloc[0].copy()
        patient_info.update({'Weeks': converted_weeks, 'FVC': 'None'})

    else:
        patient_info = patient_info.iloc[0]

    return patient_info


def main(is_train=False):
    # init data path
    root_dir = './dataset/OSIC'
    image_dir = os.path.join(root_dir, 'osic-processed-image-saved-to-npy')
    csv_dir = os.path.join(root_dir, 'osic-pulmonary-fibrosis-progression')
    train_csv_path = os.path.join(csv_dir, 'train.csv')
    test_csv_path = os.path.join(csv_dir, 'test.csv')
    submission_csv_path = os.path.join(csv_dir, 'sample_submission.csv')

    # load data
    # Read CSV
    train_data = pd.read_csv(train_csv_path)
    # test_data = pd.read_csv(test_csv_path)
    submission = pd.read_csv(submission_csv_path)

    # Preprocessing (Train, Test)
    train_data, fvc_min_max, weeks_min_max = preprocessing(train_data)
    test_data = submission['Patient_Week'].apply(lambda x: create_test_input(x, train_data))

    # load model
    model = PFPModel()

    if is_train:
        # Train Dataset
        train_label = train_data['FVC'].to_numpy()
        train_patient = train_data['Patient'].to_numpy()
        train_data_copy = train_data.drop(['FVC', 'Patient'], axis=1)
        train_dataset = Dataset(train_data_copy, train_patient, label_list=train_label, batch_size=10,
                                root_dir=image_dir)

        model.fit(train_dataset, epoch_num=100, print_epoch=10)
        model.save_weights('checkpoints/pfpModel')
    else:
        # Test Dataset
        test_label = test_data['FVC'].to_numpy()
        test_patient = test_data['Patient'].to_numpy()
        test_data_copy = test_data.drop(['FVC', 'Patient'], axis=1)
        test_dataset = Dataset(test_data_copy, test_patient, root_dir=image_dir)

        # Load pretrained model
        # model.load_weights('checkpoints/pfpModel')

        test_result = test_label.copy()
        print(test_result)
        # test_result = []

        for (img, x, y, index) in test_dataset.dataset:
            index_val = index.numpy()[0]
            patient = test_patient[index_val]
            out = model((img, x)).numpy()
            fvc = out[:, 1] if test_result[index_val] == 'None' else test_result[index_val]
            confidence = out[:, 2] - out[:, 0]

            test_result.append([fvc, confidence])

        print(test_result)


if __name__ == '__main__':
    is_train = True
    main(is_train)
