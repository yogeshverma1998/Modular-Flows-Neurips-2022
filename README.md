# Modular Flows: Differential Molecular Generation
This is the implementation of Modular Flows: Differential Molecular Generatin NeurIPS 2022 paper,
For more information visit the [website](https://yogeshverma1998.github.io/Modular-Flows-Differential-Molecular-Generation/)

## Prerequisites

- torchdiffeq : https://github.com/rtqichen/torchdiffeq.
- pysmiles
- rdkit-pypi==2021.3.5
- pytorch == 1.10.0
- torch-scatter 
- torch-sparse 
- torch-cluster 
- torch-spline-conv 
- torch-geometric 
- astropy

The environment file **Modflow_env.yml** has been attached, to replicate the necessary environment to run the method.

## Datasets
Datasets are placed in "data/" folder.

## Usage

Different scripts are provided for different datasets. To see all options, use the `-h` flag.

Training:

QM9/ZINC:
```
python train_modflow_EGNN.py/train_modflow_EGNN_3D.py --nsamples #number_of_samples_use_to_train --data #dataset(QM9/ZINC) --batch_size #batch_size --niters #iterations
```
For, Tree representation we consider 30 common ring substructures. 
First, to create ring vocabulry and finding the common subsstructures, which will create files for ring vocabulary and common ring substructures.

```
python ring_index.py --data QM9/ZINC --nsamples #number_of_samples --nrings #number_of_rings_to_be_used
python train_modflow_EGNN_2D_JT.py/train_modflow_EGNN_3D_JT.py --nsamples #number_of_samples_use_to_train --data #dataset --batch_size #batch_size --nrings #number_of_rings --niters #iterations 
```


Evaluation/generation:

QM9/ZINC:
```
python testing.py/testing_3D.py --esamples #number_of_samples_to_generate --data #dataset  --model_name #name/location_of_model_in_Models_folder
```
For, Tree representation we consider 30 common ring substructures.
```
python testing_JT.py/testing_JT_3D.py --esamples #number_of_samples_to_generate --data #dataset --nrings #number_of_rings  --model_name #name/location_of_model_in_Models_folder
```


For tasks on toy-datasets,

```
python train_modflow_toy.py --nsamples #number_of_samples --data #dataset ("4x4_chess","16x16_chess","stripes") --batch_size #batch_size --niters #iterations

python testing_toy.py --esamples #number_of_samples --data #dataset ("4x4_chess","16x16_chess","stripes") --model_name #name/location_of_model_in_Models_folder

```

For Property Optimization,

```
python prop_optimize_qed.py --nsamples #number_of_samples --data #dataset --model_name #name/location_of_model_in_Models_folder
```
