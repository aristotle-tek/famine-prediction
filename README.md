# Famine Prediction open source models

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python: 3.x](https://img.shields.io/badge/Python-3.x-blue)

An open-source repository for modeling famine.

Warning! - for now there are no experts involved, just some curiosity and a desire to make the data and models more transparent. The current focus is to model the famine in Sudan.


## Objectives

## 1. Resource Scarcity Model

We replicate (with some minor changes) the famine model described in the following Clingendael Institute reports (but we are of course responsible for any errors!):

- [Sudan: From hunger to death](https://www.clingendael.org/publication/sudan-hunger-death) 
2024-05-24, Dr. Timmo Gaasbeek. 

- [From Catastrophe to Famine: Immediate action needed in Sudan to contain mass starvation](https://www.clingendael.org/publication/catastrophe-famine-immediate-action-needed-sudan-contain-mass-starvation) 2024-02-08. Anette Hoffmann. 

## 2. Price Models

We also provide code for estimating famine alerts based on food price changes using the methodology described in 

Baquedano, F. G. (2015). [Developing an indicator of price anomalies as an early warning tool: A compound growth approach](https://www.fao.org/3/i7550e/i7550e.pdf). Food and Agricultural Organization (FAO).





## Resource Scarcity Model - Start here

We provide several Jupyter notebooks to walk through the main calculations:

1. [Walkthrough of calculating available calories (2023-2024)](notebooks/est_kcal_walkthrough.ipynb)  
    Or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aristotle-tek/famine-prediction/blob/main/notebooks/est_kcal_walkthrough.ipynb)
    For a more narrative overview, see [here](https://theoryandaction.substack.com/p/transparent-famine-models-i-counting).

2. [Fundamentals of caloric deficit, BMI change, and excess mortality](notebooks/fundamentals_walkthrough.ipynb)  
    Or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aristotle-tek/famine-prediction/blob/main/notebooks/fundamentals_walkthrough.ipynb)
    Offers an overview of basic calculations related to caloric deficits, changes in BMI, and corresponding effects on excess mortality.
    For a more narrative overview, see a discussion [ here](https://theoryandaction.substack.com/p/transparent-famine-models-ii).

3. [Resource scarcity model elements](notebooks/model_elements.ipynb)
    Or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aristotle-tek/famine-prediction/blob/main/notebooks/model_elements.ipynb)
    Explains the assumptions on initial BMI distrubtion and how calories are distributed across the population, and the list of updates calculated for each month, and 
    For a more narrative overview, see a discussion [here](https://theoryandaction.substack.com/p/transparent-famine-models-iii).

4. [Run the resource scarcity model and generate plots](notebooks/scarcity_model_run.ipynb)  
    Or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aristotle-tek/famine-prediction/blob/main/notebooks/scarcity_model_run.ipynb)
    Executes a single run of the resource scarcity model and generates output plots.

5. [Simple sensitivity analysis of the model](notebooks/sensitivity.ipynb)  
    Or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aristotle-tek/famine-prediction/blob/main/notebooks/sensitivity.ipynb)
    Evaluate the relative importance of five inputs using Sobol indices.
    For a narrative overview, see [here](https://theoryandaction.substack.com/p/famine-what-i-learned-from-coding).

## Price Anomaly Early Warning Model - Start here


[Walkthrough of the price anomaly model based on Baquedano](notebooks/price_model_walkthrough.ipynb)  
    Or [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aristotle-tek/famine-prediction/blob/main/notebooks/price_model_walkthrough.ipynb)



## Modeling Literature

In addition to the notebooks, there are references and resources on approaches to modeling in the [literature folder](modeling_literature/).


## Contributing

Please reach out with suggestions for improvement, or submit a pull request!


## License

This project is licensed under the MIT License.
