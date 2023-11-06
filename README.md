# AI Dashboard Service

## Description

An AI dashboard service that enables the inference of videos from the cloud.

## Requirements

```bash
python 3.9
```

## Install

To get started, install the proper dependencies

### Virtual environemnt (Conda)

#### Install Conda
Follow the instructions in the link below to install Conda
##### Windows
```bash
https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
```
##### Linux
```bash
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
```

#### Create Virtual Environment
Run the below code in terminal from the source directory of the project
```bash 
conda env create -n fitchain -f environment.yml python=3.9
```

#### Activate Virtual Environment
```bash 
conda activate fitchain
```

### Init Directories

#### Windows

Run the below batch file in from the source directory of the project
```bash 
init_dir.bat
```

#### Linux

Write the following command in the terminal in the source directory of the project with the virtual environment activated
```bash
source init_dir.sh
```

## Usage

```bash
 uvicorn main:app --reload --host 0.0.0.0
```

Browse to http://localhost:8000/docs

## Available endpoints

![img.png](img.png)

* Check Health: To check that the endpoint is running and accessible
* Inference:
    * Run inference on video: Runs inference on an uploaded video
    * Fetch run inference: Runs inference on a video on the cloud (downloaded through a link)
    * Fetch run inference send mail: Runs inference on a video on the cloud (downloaded through a link), uploads the processed video, and sends its
      link to the user by mail
