
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def fit_CENNTIPEDE_Effect_SNP_model(model, data1, data2):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    model.fit([data1['train_data_X'],data2['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'],data2['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)

def fit_CENNTIPEDE_CNNtipede_model(model, data1, data2, data3):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    model.fit([data1['train_data_X'],data2['train_data_X'],data3['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'],data2['val_data_X'],data3['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)

def fit_CENNTIPEDE_CNNtipede_pwm_model(model, data1, data2, data3,good_pwms_idx):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    model.fit([data1['train_data_X'][:,good_pwms_idx],data2['train_data_X'][:,good_pwms_idx],data3['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'][:,good_pwms_idx],data2['val_data_X'][:,good_pwms_idx],data3['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)