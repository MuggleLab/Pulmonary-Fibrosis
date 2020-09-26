import os

import pandas as pd

import argparse
from dataset import Dataset
from models.pfp_model import PFPModel

# parser = argparse.ArgumentParser(description='pulmonary fibrosis')
# parser.add_argument('--', type=int, default=2020,
#                     help='re-produce the results with seed random')
min_max_info = {
    'MaxFvc': (1598, 4923),
    'Age': (49, 88),
    # 'FVC': (827, 6399),
    'Weeks': (-5, 133),
}


def preprocessing(data):
    # Get Mex Percent
    data['MaxFvc'] = data['FVC'] / data['Percent'] * 100
    data.drop(['Percent'], axis=1, inplace=True)

    # min-max (MaxFvc, Age, FVC)

    for col in min_max_info.keys():
        min_val, max_val = min_max_info[col]
        data[col] = (data[col] - min_val) / (max_val - min_val)

    # One-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'SmokingStatus'])
    return data


def create_test_input(x, train_data):
    patient, weeks = x.split('_')
    weeks = int(weeks)
    weeks_min_max = min_max_info['Weeks']
    converted_weeks = (weeks - weeks_min_max[0]) / (weeks_min_max[1] - weeks_min_max[0])

    patient_info_list = train_data[train_data['Patient'] == patient]
    patient_info = patient_info_list[patient_info_list['Weeks'] == converted_weeks]
    if len(patient_info) == 0:
        patient_info = patient_info_list.iloc[0].copy()
        patient_info.update({'Weeks': converted_weeks, 'FVC': 'None'})

    else:
        patient_info = patient_info.iloc[0]

    return patient_info


def main(model_name, is_train=False):
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
    train_data = preprocessing(train_data)
    test_data = submission['Patient_Week'].apply(lambda x: create_test_input(x, train_data))

    # load model
    model = PFPModel()

    if is_train:
        # Train Dataset
        train_label = train_data['FVC'].to_numpy()
        train_patient = train_data['Patient'].to_numpy()
        train_data_copy = train_data.drop(['FVC', 'Patient'], axis=1)
        train_dataset = Dataset(train_data_copy, train_patient, label_list=train_label, batch_size=10,
                                root_dir=image_dir, shuffle=True)

        model.fit(train_dataset, epoch_num=1000, print_epoch=10)
        model.save_weights(f'checkpoints/{model_name}')
    else:
        # Test Dataset
        test_label = test_data[['FVC']].copy()
        test_patient = test_data['Patient'].to_numpy()
        test_data_copy = test_data.drop(['FVC', 'Patient'], axis=1)
        test_dataset = Dataset(test_data_copy, test_patient, root_dir=image_dir)

        # Load pretrained model
        model.load_weights(f'checkpoints/{model_name}')

        test_label['Confidence'] = 70
        for (img, x, y, index) in test_dataset.dataset:
            index_val = index.numpy()[0]
            patient = test_patient[index_val]
            out = model((img, x)).numpy()

            # FVC
            if test_label.iloc[index_val, 0] == 'None':
                test_label.iloc[index_val, 0] = out[:, 1]

            # Confidence
            test_label.iloc[index_val, 1] = out[:, 2] - out[:, 0]

        submission = pd.read_csv(submission_csv_path)
        #fvc_min, fvc_max = min_max_info['FVC']
        for idx, row in test_label.iterrows():
            fvc = row[0]
            confidence = row[1]
            submission.iloc[idx, 1] = float(fvc)
            submission.iloc[idx, 2] = float(confidence)

            #submission.iloc[idx, 1] = float(fvc * (fvc_max - fvc_min) + fvc_min)
            #submission.iloc[idx, 2] = float(confidence * (fvc_max - fvc_min) + fvc_min)

        submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    is_train = True
    model_name = 'pfpModel_0927_0330'
    main(model_name=model_name, is_train=is_train)
