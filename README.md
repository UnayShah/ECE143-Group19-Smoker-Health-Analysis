# Smoker Status Analysis using Bio-Signals

## ECE 143 Group 19 Final Project
Team member: Unay Shah, Pin-Ying Wu, Zhiyang Zhang, Zoom Chow
### Task
Smoking ranks as a primary factor in preventable diseases and fatalities globally, with negative impacts across multiple health aspects.

Despite evidence-based treatments, smoking cessation rates remain low, partly due to perceived inefficacy and time constraints found in physician counseling.

There is a critical need for physicians to effectively identify smokers who are more likely to quit. The proposed solution involves mathematical analysis of datasets to pinpoint influential bio-signals for smoker identification.

### Dataset
We use [ML Olympiad Dataset](https://www.kaggle.com/competitions/ml-olympiad-smoking/data) as our dataset, which contains training and testing CSV files with 23 bio-signal features per patient. We focus on the data analysis and visualization, so we only use the training data.
* Numerical features: age, height, weight, waist, eyesight (left & right), systolic, relaxation, fasting blood sugar, cholesterol, triglyceride, HDL, LDL, hemoglobin, serum creatinine, AST, ALT, Gtp
* Categorical features: hearing (left & right), urine protein, dental caries
* Target prediction: smoking status

## Code
We include two Python files and one Jupyter Notebook file in our project repository.

* [data_processing.py](https://github.com/UnayShah/ECE143/blob/master/project/data_processing.py) includes data preprocessing methods. For example, adding/removing dataframe columns, removing outliers, and calculating BMI, eyesight, total_cholestrol, and health score of patients.
* [build_graphs.py](https://github.com/UnayShah/ECE143/blob/master/project/build_graphs.py) includes the parameters and details for the plots, including pie charts, violin plots, bar charts, KDE plots, and also correlation plots.
* [Group_19_Smoking_Health_Effects.ipynb](https://github.com/UnayShah/ECE143/blob/master/project/Group_19_Smoking_Health_Effects.ipynb) shows all the visualizations in the slides.


## Modules/Packages Used in the Project
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```
