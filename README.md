# Jump2Digital 2022
NUWE DataScience Hackathon

## Background

The [Paris Agreement](https://en.wikipedia.org/wiki/Paris_Agreement) is an international treaty on climate change that was adopted by 196 Parties at COP21 in Paris. Their goal is to limit global warming below 2, preferably 1.5 degrees Celsius, compared to pre-industrial levels.

In order to reach the goal, countries are trying to maximize their greenhouse gas emissions and archive a climate-neutral planet.

That is why EU is investing in the development of new technologies that allow the improvement of the fight againts pollution. One of these is a new type of sensor based on laser technology that allows measure of air quality.

## Problem

The goal of the challenge is to develop and implement and predictive model based on [*RandomForest*](https://en.wikipedia.org/wiki/Random_forest) algorithms that allows to know the air quality using measurements of the sensors.

To perform this task, we are provided with a [training dataset](https://github.com/Visn0/jump2digital2022/blob/main/data/train.csv), a [dataset to predict](https://github.com/Visn0/jump2digital2022/blob/main/data/test.csv), and an [example of the expected format](https://github.com/Visn0/jump2digital2022/blob/main/data/ejemplo_predicciones.csv).

To measure the score of the results, the measure of **f1-score(macro)** will be used together with the quality of the code and the documentation

## Results

For the experiments, 80% of the data was used for training and another 20% for validation.
The model using the parameters obtained from GridSearchCV show the following results:

|                | **Accuracy** | **f1 score** |
| -------------- | ------------ | ------------ |
| **Train**      | 0.9964       | 0.9964       |
| **Validation** | 0.9048       | 0.9044       |

## Analysis

The analysis of the data has been shorter than expected. The analysis has indicated that no data was missing nor were there duplicate rows. In addition, the mean and median of the features were very similar with an optimal standard deviation.

For this particular problem, it was very difficult to obtain any other data external to the provided dataset, that is, *feature extraction* could not be done. So I have limited myself to doing feature reduction using **PCA**, **LinearDiscriminantAnalysis** and **NeighborhoodComponentsAnalysis** to better understand the data. On the other hand, I have also tried to scale the data in different ways (although the data does not need it since they have an optimal std as we said).

Finally, in the exploration file you can see some graphs that helped me understand the composition of the data, such as the correlation graph or different histograms and, in the case of the results, the confusion matrix.

## Solution

The solution is presented in different formats [predictions.csv](https://github.com/Visn0/caixabank_nuwe_reto_data/blob/main/predictions.csv) and [predictions.json](https://github.com/Visn0/caixabank_nuwe_reto_data/blob/main/predictions.json) as requested. These files are located in the root directory of the project.

## License

[MIT licence](https://choosealicense.com/licenses/mit/)