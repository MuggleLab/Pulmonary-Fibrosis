def main():
    import numpy as np
    import pandas as pd
    import os
    import random
    # import matplotlib.pyplot as plt

    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import KFold

    import tensorflow as tf
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M

    def seed_everything(seed=2020):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    seed_everything(42)

    # ROOT = '../input/osic-pulmonary-fibrosis-progression'
    ROOT = "./dataset"

    # %%

    tr = pd.read_csv(f"{ROOT}/train.csv")
    tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
    chunk = pd.read_csv(f"{ROOT}/test.csv")

    print("add infos")
    sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
    sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
    sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
    sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
    sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")


    tr['WHERE'] = 'train'
    chunk['WHERE'] = 'val'
    sub['WHERE'] = 'test'
    data = tr.append([chunk, sub])

    print(tr.shape, chunk.shape, sub.shape, data.shape)
    print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(),
          data.Patient.nunique())

    # get min week
    data['min_week'] = data['Weeks']
    data.loc[data.WHERE == 'test', 'min_week'] = np.nan
    data['min_week'] = data.groupby('Patient')['min_week'].transform('min')


    base = data.loc[data.Weeks == data.min_week]
    base = base[['Patient', 'FVC']].copy()
    base.columns = ['Patient', 'min_FVC']
    base['nb'] = 1
    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
    base = base[base.nb == 1]
    base.drop('nb', axis=1, inplace=True)

    data = data.merge(base, on='Patient', how='left')
    data['base_week'] = data['Weeks'] - data['min_week']
    del base

    # get_dummy
    COLS = ['Sex', 'SmokingStatus']  # ,'Age'
    FE = []
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)
    # =================

    data['full_FVC'] = (data['FVC'] / data['Percent']) * 100

    #min-max normalization
    # age: 나이
    # BASE: 가장 처음 측정된 FVC
    # FULL_BASE: 동일한 분류의 사람의 정상 폐황량이 100%일 때 FVC 값
    # week: 가장 처음 측정된 week
    # percent: 동일한 부류의 사람의 정상 폐활량이 100%일 때 해당 환자의 폐활량에 따른 percentage

    data['age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    data['BASE'] = (data['min_FVC'] - data['min_FVC'].min()) / (data['min_FVC'].max() - data['min_FVC'].min())
    data['FULL_BASE'] = (data['full_FVC'] - data['full_FVC'].min()) / (data['full_FVC'].max() - data['full_FVC'].min())
    data['week'] = (data['base_week'] - data['base_week'].min()) / (data['base_week'].max() - data['base_week'].min())
    data['percent'] = (data['Percent'] - data['Percent'].min()) / (data['Percent'].max() - data['Percent'].min())
    FE += ['age', 'percent', 'week', 'BASE', 'FULL_BASE']

    # %%

    tr = data.loc[data.WHERE == 'train']
    chunk = data.loc[data.WHERE == 'val']
    sub = data.loc[data.WHERE == 'test']
    del data


    ### BASELINE NN


    C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

    # =============================#
    def score(y_true, y_pred):
        tf.dtypes.cast(y_true, tf.float32)
        tf.dtypes.cast(y_pred, tf.float32)
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]

        # sigma_clip = sigma + C1
        sigma_clip = tf.maximum(sigma, C1)
        delta = tf.abs(y_true[:, 0] - fvc_pred)
        delta = tf.minimum(delta, C2)
        sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
        metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
        return K.mean(metric)

    # ============================#
    def qloss(y_true, y_pred):
        # Pinball loss for multiple quantiles
        qs = [0.2, 0.50, 0.8]
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    # =============================#
    def mloss(_lambda):
        def loss(y_true, y_pred):
            return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

        return loss

    # =================
    def make_model(nh):
        z = L.Input((nh,), name="Patient")
        x = L.Dense(100, activation="relu", name="d1")(z)
        x = L.Dense(100, activation="relu", name="d2")(x)
        # x = L.Dense(100, activation="relu", name="d3")(x)
        p1 = L.Dense(3, activation="linear", name="p1")(x)
        p2 = L.Dense(3, activation="relu", name="p2")(x)
        preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

        model = M.Model(z, preds, name="CNN")
        # model.compile(loss=qloss, optimizer="adam", metrics=[score])
        model.compile(loss=mloss(0.8),
                      optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01,
                                                         amsgrad=False), metrics=[score])
        return model

    # %%

    # y = tr['FVC'].values
    y = tr['FVC'].values.astype(float)
    z = tr[FE].values
    ze = sub[FE].values
    nh = z.shape[1]
    pe = np.zeros((ze.shape[0], 3))
    pred = np.zeros((z.shape[0], 3))

    # %%

    # net = make_model(nh)
    # print(net.summary())
    # print(net.count_params())

    # %%

    NFOLD = 5
    kf = KFold(n_splits=NFOLD)

    count = 0
    EPOCHS = 100
    BATCH_SIZE = 128

    for tr_idx, val_idx in kf.split(z):
        count += 1
        # print(f"FOLD {count}")
        net = make_model(nh)
        net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS,
                validation_data=(z[val_idx], y[val_idx]), verbose=0)  #
        print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
        print("val  ", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
        # print("predict val...")
        pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
        # print("predict test...")
        pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    # ==============

    # %%

    sigma_opt = mean_absolute_error(y, pred[:, 1])
    unc = pred[:, 2] - pred[:, 0]
    sigma_mean = np.mean(unc)
    print('sigma_opt: ', sigma_opt, 'sigma_mean: ', sigma_mean)

    # print(unc.min(), unc.mean(), unc.max(), (unc >= 0).mean())

    # %%

    # %% md

    ### PREDICTION

    # %%

    sub.head()

    # %%

    sub['FVC1'] = 0.996 * pe[:, 1]
    sub['Confidence1'] = pe[:, 2] - pe[:, 0]

    # %%

    subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()

    # %%

    subm.loc[~subm.FVC1.isnull()].head(10)

    # %%

    subm.loc[~subm.FVC1.isnull(), 'FVC'] = subm.loc[~subm.FVC1.isnull(), 'FVC1']
    if sigma_mean < 70:
        subm['Confidence'] = sigma_opt
    else:
        subm.loc[~subm.FVC1.isnull(), 'Confidence'] = subm.loc[~subm.FVC1.isnull(), 'Confidence1']

    # %%

    # otest = pd.read_csv(f"{ROOT}/test.csv")
    # for i in range(len(otest)):
    #     subm.loc[subm['Patient_Week'] == otest.Patient[i] + '_' + str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    #     subm.loc[subm['Patient_Week'] == otest.Patient[i] + '_' + str(otest.Weeks[i]), 'Confidence'] = 0.1

    # %%

    subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)

    # %%


if __name__ == '__main__':
    main()
