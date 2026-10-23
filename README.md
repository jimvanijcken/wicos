# WICOS: radar hand-gesture classification (DopNet)

Classification of four hand gestures (wave, pinch, swipe, click) from micro-Doppler spectrograms of six persons.

## Contents
| path | content |
|---|---|
| `src/preprocessing.py` | noise removal, Doppler crop, scaling and resizing of one spectrogram |
| `src/features.py` | micro-Doppler envelope features (Amin et al., 2019) |
| `src/build_dataset.py` | builds `data/processed/wicos_dataset.npz`: test split, CV folds, images and features |
| `notebooks/1_data_exploration.ipynb` | data exploration and the data split |
| `notebooks/2_preprocessing.ipynb` | pre-processing choices |
| `notebooks/3_features.ipynb` | envelope features |
| `notebooks/4_models.ipynb` | LR, SVM, RF, SVM + PCA and CNN: tuning, model choice, LOSO and test set |
| `data/processed/` | the model-ready dataset (output of `build_dataset.py`) |
| `results/` | result tables; `cnn_cv.csv` caches the CNN cross-validation results |
| `reports/figures/` | the figures of the report |

## Running
1. Install the packages (Python 3.12): `pip install -r requirements.txt`
2. `notebooks/4_models.ipynb` runs on the included `data/processed/wicos_dataset.npz`. Run it from the `notebooks`
   folder. The CNN cross-validation results are read from `results/cnn_cv.csv`; delete that file to retrain them
   (slow on a laptop CPU).
3. Notebooks 1-3 and `build_dataset.py` need the raw DopNet data: unzip the CSV files of the six persons into
   `data/Person A/<gesture>/PersonAg1s1.csv` ... `data/Person F/...`, then run `python src/build_dataset.py`.
