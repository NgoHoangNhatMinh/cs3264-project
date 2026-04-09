# CS3264 Project

This repository contains the code and documentation for our CS3264 project on accident detection in elderly homes using deep learning. The project includes data collection, model training, and deployment components.

## Setting Up the Environment

To set up the Python environment for this project, follow these steps:

1. Install Anaconda or Miniconda if you haven't already.
2. Clone this repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

5. Activate the environment:

```bash
conda activate cs3264-project-env

# To deactivate the environment later, use:
conda deactivate
```

6. You can now run the Jupyter notebooks and Python scripts included in this repository by

```
jupyter notebook
```

## Preparing the Data

## Training the Model

It is recommended to train the model using SOC computing clusters.

### Install Miniconda on the Cluster

### Exporting Jupyter Notebook as a Python Script

Prepare your training script (e.g., `model.py`) and the environment file (`environment.yml`) for uploading to the cluster.

The cluster doesn't run Jupyter Notebooks easily. You need to export your model.ipynb as a Python script into the `scripts/` directory.

In Jupyter: File > Save and Export Notebook As... > Executable Script.

### Uploading Code to the Cluster

Upload the code and environment file to the cluster:

```bash
scp -r scripts/ environment.yml your_soc_unix_id@xlogin.comp.nus.edu.sg:~/your_project_directory/
```

And upload the data (this will take a while):

```bash
scp -r data your_soc_unix_id@xlogin.comp.nus.edu.sg:~/your_project_directory/
```

### Submit the Job

Submit the job to the cluster:

```bash
sbatch scripts/submit.sh

# Check the status of your job:
squeue -u your_soc_unix_id

# To see the training progress in real-time
tail -f training_[JOB_ID].log
```
