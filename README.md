# DisasterResponsePipeline
Project of  Udacity Data Scientist Nanodegree Program

### Table of Contents
- [Table of Contents](#table-of-contents)
  - [Installation <a name="installation"></a>](#installation-)
  - [Project Motivation<a name="motivation"></a>](#project-motivation)
  - [File Descriptions <a name="files"></a>](#file-descriptions-)
  - [Results and Discussion<a name="results"></a>](#results-and-discussion)
  - [Licensing, Authors, Acknowledgements<a name="licensing"></a>](#licensing-authors-acknowledgements)

## Installation <a name="installation"></a>

Required packeages are listed in `requirement.txt`. 

Follow follows steps to run the app: 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

In this project, the process of a comprehensive implementation of Machine Learning in realworld project is demonstrated, which includes following steps: 

1. ETL process includes extracting data, cleanning data and storing the clean data into a SQLite database. 
2. Using NLP, Pipeline and GridSearchCV to classificate data. 
3. Deployment the model as a web app


## File Descriptions <a name="files"></a>

There are 3 directories here. 

1. Directory `app` contains the script to start the web app `run.py` and the webpage templates in subdirectory `templates`
2. Directory `data` contains the origin data `disaster_categories.csv` and `disaster_messages.csv`, the ETL script `process_data.py` and the database 
`DisasterResponse.db` which saves the cleaned data. 
3. Directory `models` saves the ML script `train_classifier.py` and the saved trained ML model `final_model.py`.

## Results and Discussion<a name="results"></a>

It should be pointed out, there is still much room for imporvement. An obvious problem is the data is imbalanced, which has stongly influenced the accuracy and precison of the trained model. Another improvement is to employ the model in a cloud server rather than locally. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the project. Otherwise, feel free to use the code here as you would like!



