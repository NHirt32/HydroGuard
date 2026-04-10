# HydroGuard

This project implements an end-to-end Big Data pipeline to predict flood risks using diverse environmental data sources. Developed as part of the Master of Science in Big Data program at SFU, the system leverages AWS infrastructure and machine learning to forecast potential flooding events based on historical and real-time metrics.
## Project Overview

The goal of this project is to provide a predictive model that assesses flood probability by analyzing the relationship between meteorological data and hydrological basin characteristics.

Key Data Sources:

    Weather APIs: Historical and forecasted precipitation and temperature.
    https://open-meteo.com/
    
    Reservoir Levels: Current and historical water levels in local basins.
    https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/UnitValueData/Stage/corrected/ 
    https://dd.weather.gc.ca/today/hydrometric/csv/BC/daily/

    Climate Levels: Historical climate metrics
    https://dd.weather.gc.ca/today/climate/observations/daily/csv/ 

    Geospatial Data: Basin size, elevation changes, and soil saturation metrics.
    https://essd.copernicus.org/articles/17/259/2025/

## System Architecture

The project utilizes a cloud-native architecture to handle large-scale data ingestion and processing.

    Ingestion: Automated pipelines using AWS Lambda and EventBridge.

    Storage: Raw and processed data staged in Amazon S3.

    Processing: Feature engineering and data cleaning using Python.

    Modeling: Use a Histogram-Based Gradient Boosting model from scikit

## Repository Structure

The repository is organized into Jupyter Notebooks that follow the data science lifecycle:

    Flood Model + Dataset.ipynb: Contains the full script to process the data files into the training dataset and training the model (combines scripts below into one + model)
    Flood Webscrape.ipynb: Webscraping the Hydro and Climate historical data  from the Canada Government website
    dashboard-etl.ipynb: Processes JSON files from our API calls for use with our visualization dashboard.
    drainage-area-etl.ipynb: Augments our dataset with the corresponding drainage area for each station.
    model-etl.ipynb: Augements our dataset for model training imputing missing features and creating new features.
    unique-coordinates-identifier.ipynb: The script responsible for identifying all unique coordinate combinations for the purpose of weather api polling.
    

## Technical Stack

    Language: Python 3.x

    Big Data: Pandas, Numpy, GeoPandas, Pickle, BeautifulSoup

    Machine Learning: Scikit-learn, SHAP

    Cloud: AWS (Lambda, EventBridge, S3)

    Visualization: Power BI, Matplotlib, Shapely


## Contributors

    Nicholas Hirt – M.Sc. Big Data Candidate, SFU
    Cole Hanniwell – M.Sc. Big Data Candidate, SFU
    TODO (ADD YOUR NAMES)
