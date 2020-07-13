# Facial Recognition System

This is a simple facial recognition system using Python, PostgreSQL and JavaScript, running on Docker Compose.

## Disclaimer ⚠️

This project was originally developed as part of a surveillance system in the retail industry in Mexico. Most of the content is in Spanish and has retail-related elements.

## Features

The system is capable of:

* Registering new faces in the database
* Deleting faces from the database
* Extracting age, gender and emotion from the faces

When a face is added, the images provided are used to perform data augmentation, in order to keep a minimum of 10 embeddings per entry.

Two different classifiers can be used:
* Support Vector Classifier
* Ball Tree Nearest-Neighbor

The model is trained upon addition and/or deletion of an entry.

## Requirements

* Docker & Docker Compose

## Installation

1. Clone the project and _cd_ into it.
2. Create a file called _.env_ with the following content:
```
PG_USER=example_user
PG_PASS=example_password
PG_DB=example_database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
APP_PORT=8000
```
3. Start the project with ```docker-compose up```

## Register

In order to register new faces, run the project and go to **localhost:8000/agregar**.

![Register 1](/utils/images/registrar1.PNG)

You can add multiple images from your computer and provide additional information about the subject.

Click on "Agregar" in order to get a visualization of the facial landmarks of the images provided. You can then discard the images that could not be useful or that have the landmarks misplaced.

![Register 2](/utils/images/registrar2.PNG)

Then click on "Confirmar" to add the new faces to the database.

## Delete

In order to register new faces, go to **localhost:8000/eliminar**.

You will be prompted with the entries that you had previously registered, as well as their details. You can click on the buttons under "Eliminar" to delete them.

![Delete](/utils/images/eliminar.PNG)

## Predict

In order to predict a new image, go to **localhost:8000/predecir**.

Upload an image from your computer and click on "Predecir". The results of the prediction (name, similarity, age, gender and emotion) will be shown on the right.

![Predict](/utils/images/predecir.PNG)
