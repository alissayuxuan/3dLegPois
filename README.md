# Deep Learning-Based Prediction of Anatomical Ponits-of-Interest from Sparse Annotations on Segmentation Masks

Welcome to the code repository of my bachelor's thesis at TUM. This code is focussing on the POI-prediction of the lower limbs. Most of the code was taken from the master's thesis 'Automated Point-of-Interest Prediction on CT Scans of Human Vertebrae Using Spine Segmentations' by Daniel-Jordi Regenbrecht. You can find the code here: https://github.com/doppelplusungut/3dVertPois.

For a quick overview of the project, you can view my [poster](doc/PosterPOIPred.pdf).

## Project Description

The project comprises of three major components:

- Data analysis and preparation
- Data augmentation using elastic deformation
- Training a POI prediction model
- An inference pipeline for a given sample/dataset

## Installation (Ubuntu)

Create and activate a virtual environment, e.g. by running

```bash
conda create -n poi-prediction python=3.10
cona activate poi-prediction
```

Set the python interpreter path in your IDE to the version in the environment, i.e. the output of

```bash
which python
```

Install the TPTBox, for 

```bash
pip install TPTBox
```

Back in the project directory, install the required packages 

```bash
pip install -r requirements.txt
```

## Training your Own Model

To train your own model, a dataset in BIDS-like format is required, i.e. the following structure is expected:

```text
dataset-folder
├── <rawdata>
    ├── subfolders
        ├── CT image file
├── <derivatives>
    ├── subfolders
        ├── Instance Segmentation Mask
        ├── Subregion Segmentaiton Mask
        ├── POI file
```

For each CT file, corresponding segmentation files and a POI file are required.

Since the size of the CT scans and segmentations is generally too large to fit into GPU memory, the legs are cut into 3 FOVs. The model then predicts the POIs on each FOV in a separate training instead of proccessing the entire leg at once. The images are brought into standard orientation and scale. To avoid repetitive computations during training, these preprocessing steps are carried out in bulk and the cutouts are saved to the disk. In order to run the bulk preprocessing, enter the src folder and run prepare_data.py.

The prepare_data.py was adapted to process the data available, which included different POI files for each leg side.
We use FOV-cuts for the training of our model. We cut the legs into a superior femur, knee area and the inferior lowerlegs.
Other cutting methods have been implemented, which can be selected using the arguments --straight_cut, --fov_cut, --side_cut. If no cutting type is specified, each bone is cut out with a default margin of 2 voxels.

WARNING: By default, this step uses 8 CPU cores in parallel to speed up the pre-processing. You can specify a different number with the --n_workers argument.

```bash
cd src
python3 prepare_data.py --data_path $PATH_TO_YOUR_DS --derivatives_name $NAME_OF_DERIVATIVES_FOLDER --save_path $PATH_TO_SAVE_CUTOUS
```

Along with the cutouts, the script saves a csv file containing the paths of all cutouts and a json file specifying the parameters used for the creation of cutouts to reliably create appropriate cutouts during inference.

Once cutouts are created, you can start training. Create a training config (samples can be found in the experiments/experiment_configs subdirectory) to specify the location of data, logging, as well as data and model configurations. Then, inside the src folder, run

```bash
train.py --config $PATH_TO_YOUR_CONFIG_FILE
```

You can also run several trainings consecutively by placing the respective config files in one directory and using the --config-dir argument instead of --config in the above call. Further, training can be run using cross-validation by using train_cv.py instead of train.py. In this mode, training and validtation split in the config will be treated the same and random splits will be created for each fold (adjustable using the --n_folds argument)

## Inferring with a Trained Model

tbc

## Example Usage (Inference on VerSe19)

tbc
