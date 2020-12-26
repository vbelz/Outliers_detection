# A production ready approach for outlier detection and monitoring


## Introduction

<img src="images/will-myers-ku_ttDpqIVc-unsplash.jpg" alt="outliers intro" title="Outliers"/>

Production data in ML projects are often polluted by anomalies also called outliers.
These outliers might be due to an error in a process or simply be part of your dataset. It is important
to identify and correct for them as they can disturb your modelling/interpretation of your data.
In this article, we are going to present a production ready approach in order to detect and monitor outliers,
enabling you to save some precious time to build your ML pipeline.

In a first step, we are going to show an easy and effective manner to detect outliers column-wise using the interquartile range (IQR).

In a second, we are going to treat outliers from a multidimension perspective (multi-columns),
and detect them automatically using isolation forest.

Finally, we are going to show an effective way to monitor your database with pandera, a statistical data validation toolkit.

The dataset being used in this project is available at the github repository. This is a fictitious dataset representing real estate data.
It contains 6 columns representing `rooms` (number of rooms), `garages` (number of garages), `useful_area` (area of the flat), `value` (price of the flat), `interior_quality` (interior quality of the flat), `time_on_market` (time needed to sell the flat). I introduced several anomalies in the 6 columns
to demonstrate the effectiveness of the metodologies presented below.


## Easily detect outliers column-wise with IQR

<img src="images/IQR.png" alt="IQR" title="IQR"/>

The interquartile range (IQR) is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles,
or equivalently between upper and lower quartiles, IQR = Q3 − Q1.

It can be used to extract the most anomalous values for each of the column from our data.

One way to do so is to consider as potential outliers, data of each column outside the range Q1 - 1.5 * IQR, Q3 + 1.5 * IQR.

Per column, we can look at the distribution with all values vs distribution excluding these outliers.

When the distribution have a very different shape, it highlights the presence of outliers within the column as illustrated below.

<img src="images/distributions.png" alt="distributions" title="distributions"/>

Here we can see on the first row the full distribution for price, area and time on market. On the second row, we applied the IQR filtering before
plotting the distribution:

[https://gist.github.com/vbelz/8a09ff54ed740b71bf8aafeaa009d193]
<img src="images/IQR_example.py.png" alt="IQR" title="IQR"/>

We can see that without the filtering the distributions are highly peaked, highlighting the presence of outliers.
The plots after filtering are more representative of the true data distribution. Based on these plots and our knowledge of
the data, we can decide what range of data would be acceptable for each column.

For this example, we would not expect price outside of range 50 000 - 10 000 000 Reais, area oustide range 20-850 m2, time on market outside of range
0-400 days. We can perform this type of outlier EDA for each of our column. This will be useful to set up our business rules for each column in the
section for outliers monitoring using pandera.


## Isolation forest to detect outliers from multi-columns

<img src="images/plot_isolation_forest_varying_threshold.gif" alt="isolation forest" title="isolation forest"/>

<img src="images/boxplot_score.png" alt="boxplot score isolation" title="boxplot score isolation"/>


## Monitor your data with pandera: a statistical data validation toolkit for pandas


<img src="images/example_pandera.png" alt="pandera output" title="pandera output"/>

## References

>Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation-based anomaly detection.” ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
>
>[https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf]

>Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008. ICDM’08. Eighth IEEE International Conference on.
>
>[https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest]

>
>
>[https://www.pyopensci.org/blog/pandera-python-pandas-dataframe-validation]

>
>
>[https://towardsdatascience.com/how-automated-data-validation-made-me-more-productive-7d6b396776]

> Pandera documentation:
>
> [https://pandera.readthedocs.io/en/stable/]
