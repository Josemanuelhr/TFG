import argparse

import mne
from mne.externals.pymatreader import read_mat

import numpy as np
import pandas as pd

import os

'''
A partir del fichero RAW del SEED-VIG se genera el RawArray de MNE
Lo devuelve sin cargar en memoria, es necesario hacer un load_data() para trabajar con los datos
'''
def mat_to_raw(filename, verbose=True):
    mat_data = read_mat(filename)

    samples = mat_data['EEG']['data'].T*1e-6

    sfreq = mat_data['EEG']['sample_rate']
    ch_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1', 'Pz', 'P2', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
    ch_types = ["eeg"]*len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020')

    raw = mne.io.RawArray(samples, info, verbose=verbose)

    return raw

'''
A partir del nombre del fichero RAW se obtienen sus valores de PERCLOS
'''
def get_perclos(filename):
    perclos = read_mat(filename)
    return perclos["perclos"]

'''
Genera una copia de los datos raw pero filtrados con paso banda, resampling e ICA (opcional).
'''
def filter_data(raw, apply_ica=True, verbose=True):
    raw_filtered = raw.copy().filter(l_freq=1, h_freq=30)
    raw_filtered.resample(60)
    if (apply_ica):
        n_components = 0.999  # Tambien se podria poner 16 (n_channels -1)
        method = 'picard'
        max_iter = 500  # entre 500 y 1000 para que converja
        fit_params = dict(fastica_it=5)
        random_state = 42

        ica = mne.preprocessing.ICA(n_components=n_components,
                                    method=method,
                                    max_iter=max_iter,
                                    fit_params=fit_params,
                                    random_state=random_state)
        ica.fit(raw_filtered, verbose=verbose)
        ica.apply(raw_filtered, verbose=verbose)

    return raw_filtered
'''
Crea Epochs de la duracion especificada a partir de los datos.
Son epocas artificiales ya que el caso de estudio no tiene eventos (es un estudio cognitivo).
'''
def create_epochs(raw, duration, verbose=True):
    return mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose=verbose)

'''
Almacena en un csv las 'X' y la 'y' para los datos de eeg y perclos suministrados
'''
def create_features(epochs, perclos, filename):
    n_epochs = len(epochs)
    n_channels = len(epochs.info['ch_names'])
    data = epochs.get_data()

    Xlabels = ['mean', 'std', 'var', 'p05', 'q1', 'median', 'q3', 'p95']
    ylabel = 'perclos'
    n_features = len(Xlabels)
    total_features = n_channels*n_features

    X = np.zeros(shape=(n_epochs, total_features))
    y = np.array(perclos)

    for i in range(n_epochs):
        for channel in range(n_channels):
            X[i, n_features*channel+0] = np.mean(data[i][channel])
            X[i, n_features*channel+1] = np.std(data[i][channel])
            X[i, n_features*channel+2] = np.var(data[i][channel])
            X[i, n_features*channel+3] = np.percentile(data[i][channel], 5)
            X[i, n_features*channel+4] = np.quantile(data[i][channel], 0.25)
            X[i, n_features*channel+5] = np.median(data[i][channel])
            X[i, n_features*channel+6] = np.quantile(data[i][channel], 0.75)
            X[i, n_features*channel+7] = np.percentile(data[i][channel], 95)

    XT = X.T

    csv_dict = {}
    for i in range(n_channels):
        channel = f'c{i+1:02}_'
        for j in range(n_features):
            key = channel + Xlabels[j]
            val =  XT[n_features*i+j]

            csv_dict[key] = val

    csv_dict[ylabel] = y

    _ ,file = os.path.split(filename)
    file_no_ext = os.path.splitext(file)[0]
    out_file = f'./features/{file_no_ext}.csv'
    df = pd.DataFrame(csv_dict)
    df.to_csv(out_file, index=False)

'''
METODO PRINCIPAL
'''
def generate_csv_with_features():
    raw = mat_to_raw(args.eeg, verbose=args.v)
    perclos_data = get_perclos(args.perclos)
    
    duracion = raw.get_data().shape[1]/raw.info['sfreq']
    n_perclos = perclos_data.shape[0]
    int_perclos = duracion/n_perclos

    raw_filtered = filter_data(raw, apply_ica=True, verbose=args.v)
    epochs = create_epochs(raw_filtered, int(int_perclos), verbose=args.v)
    create_features(epochs, perclos_data, args.eeg)

'''
Recupera del documento CSV para todos los canales las features indicadas y la columna del valor perclos 'y'
n_channel: numero de canales
'''
def read_features(filename, n_channels=17, features=[0,1,2,3,4,5,6,7], feat_per_channel=8):

    features = np.array(features)
    selected_features = []
    for i in range(n_channels):
        selected_features += (features+(feat_per_channel*i)).tolist()

    selected_features.append(n_channels*feat_per_channel) # para coger la ultima columna que es la 'y'

    data = pd.read_csv(filename, usecols=selected_features)

    X = data.T.values[:-1].T
    y = data.T.values[-1]

    return (X, y)

def main():
    raw = mat_to_raw(args.eeg, verbose=args.v)
    perclos_data = get_perclos(args.perclos)
    
    duracion = raw.get_data().shape[1]/raw.info['sfreq']
    n_perclos = perclos_data.shape[0]
    int_perclos = duracion/n_perclos

    if (args.v):
        print(f'Duracion en segundos: {duracion}')
        print(f'Cantidad de registros PERCLOS: {n_perclos}')
        print (f'Cada {int_perclos} segundos se da un valor de PERCLOS')

    raw_filtered = filter_data(raw, apply_ica=True, verbose=args.v)
    epochs = create_epochs(raw_filtered, int(int_perclos), verbose=args.v)
    create_features(epochs, perclos_data, args.eeg)

    print("fin.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello World!")

    parser.add_argument("-v", default=False, help="modo verboso", action="store_true")
    parser.add_argument("-eeg", metavar="fichero_eeg", type=str, required=True, help="fichero de datos raw eeg")
    parser.add_argument("-perclos", metavar="fichero_perclos", type=str, required=True, help="fichero de valores perclos")

    args = parser.parse_args()

    main()
