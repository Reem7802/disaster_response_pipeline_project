# disaster_response_pipeline_project

this repository show you a machine learning pipeline to categorize events.

.

# Libraries used:

The following python libraries were used in this project.
json, plotly, pandas, joblib, re, nltk, flask, sqlalchemy, sys, pickle, sklearn

.

# Running the code:

From the project folder run run.py to start the dash application

.


# Project Motivation:

For this project, i Built a machine learning pipeline to categorize  events so that you can send the messages to an appropriate disaster relief agency including a web app where an emergency worker can input a new message and get classification results in several categories.

.

# File Description:


- app

     | - template

     |    |- master.html # main page of web app

     |    |- go.html # classification result page of web app

     | - run.py # Flask file that runs app 

- data

     | - disaster_categories.csv # data to process

     | - disaster_messages.csv # data to process

     | - process_data.py

     | - InsertDatabaseName.db # database to save clean data to

- models

     | - train_classifier.py

     | - classifier.pkl # saved model


- README.md


.

# Summary:

in this project we Built a machine learning pipeline to categorize  events so that you can send the messages to an appropriate disaster relief agency, provided by a data set containing real messages that were sent during disaster events.

.
# Licensing, Authors, Acknowledgements:

the datasets was taken from Figure Eight 

