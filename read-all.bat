@ECHO OFF
time /t
for /r %%i in (.\SEED-VIG\Raw_Data\*) do (
    echo Procesando: %%~ni.mat
    python data-to-csv.py -eeg ./SEED-VIG/Raw_Data/%%~ni.mat -perclos ./SEED-VIG/perclos_labels/%%~ni.mat
    echo Completado: %%~ni.mat
)
ECHO ---- Fin del script ----
time /t
PAUSE