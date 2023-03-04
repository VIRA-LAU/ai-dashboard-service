# AI Dashboard Service

## Description

An AI dashboard service that enables the inference of videos from the cloud.

## Install

To get started, install the proper dependencies

### Pip

```bash
pip install -r requirements.txt
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
