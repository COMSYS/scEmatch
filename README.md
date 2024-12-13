# sc*E*(match):  Privacy-Preserving Cluster Matching of Single-Cell Data

This repository contains our fully tested and evaluated prototype of sc*E*(Match), which provides a method for privacy-friendly comparison of clustered single-cell datasets.

## About

**Paper Abstract**

Advances in single-cell RNA sequencing (scRNA-seq) have dramatically enhanced our understanding of cellular functions and disease mechanisms.
Despite its potential, scRNA-seq faces significant challenges related to data privacy, cost, and IP protection, which hinder the sharing and collaborative use of these sensitive datasets.
In this paper, we introduce a novel method, sc*E*(Match), a privacy-preserving tool that facilitates the matching of single-cell clusters between different datasets by relying on scMap as an established projection tool, but without compromising data privacy or IP.
sc*E*(Match) utilizes homomorphic encryption to ensure that data and unique cell clusters remain confidential while enabling the identification of overlapping cell types for further collaboration and downstream analysis.
Our evaluation shows that sc*E*(Match) performantly matches cell types across datasets with high precision, addressing both practical and ethical concerns in sharing scRNA-seq data.
This approach not only supports secure data collaboration but also fosters advances in biomedical research by reliably protecting sensitive information and IP rights.


## Installation
Install the required dependencies as specified in the `requirements.txt`. This prototype has been tested with:

- Python 3.10
- Ubuntu 22.04.5 LTS

## Matching datasets
sc*E*(Match) is compatible with scRNA-Seq datasets of `.h5ad` format as utilized by Scanpy.
We provide an example with two publicly available human heart scRNA-Seq datasets.

- **Litvinukova et al. (D1)**: https://cellgeni.cog.sanger.ac.uk/heartcellatlas/data/global_raw.h5ad
- **Chaffin et al. (D2)** (Requires registration): https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy

Download the datasets to the `datasets` folder and rename the datasets to `litvinukova.h5ad` and `chaffin.h5ad`, respectively, for compatibility with our published configuration.

From a terminal, navigate to the folder `application`.
```
cd application
```

### Step 1: Feature Selection and Alignment

Run the feature selection and alignment script to derive the query and reference dataset versions for datasets D1 and D2.

```
python3 generate_feature_aligned_datasets.py
```

### Step 2: Run sc*E*(match)

Execute the following command to run sc*E*(match) between the datasets D1 and D2.

```
python3 apply_cluster.py scenario='example.yaml' mode='homomorphic'
```

### Configuration Options

Adjust the configuration parameters in the `generate_feature_aligned_datasets.py` script located in the `application` folder and the scenario YAML files located in `application/scenario_configs`.
Results will be saved in `application/artifacts/`.

## Publication

* Johannes Lohmöller, Jannis Scheiber, Rafael Kramann, Klaus Wehrle, Sikander Hayat, and Jan Pennekamp: _scE(match): Privacy-Preserving Cluster Matching of Single-Cell Data_. 23rd IEEE International Conference on Trust, Security and Privacy in Computing and Communications 2024 (TrustCom-2024), IEEE, 2024.

```
@inproceedings{lohmoeller24scematch,
 address = {Sanya, China},
 author = {Johannes Lohmöller and Jannis Scheiber and Rafael Kramann and Klaus Wehrle and Sikander Hayat and Jan Pennekamp},
 booktitle = {23rd IEEE International Conference on Trust, Security and Privacy in Computing and Communications 2024 (TrustCom-2024)},
 month = {December},
 title = {scE(match): Privacy-Preserving Cluster Matching of Single-Cell Data},
 year = {2024}
}
```
