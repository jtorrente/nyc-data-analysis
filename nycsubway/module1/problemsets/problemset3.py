__author__ = 'Udacity+jtorrente'

import numpy as np
import pandas
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
import sys

'''
' 1) Exploratory Data Analysis
'''
def entries_histogram(turnstile_weather):
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.

    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()

    Your histograph may look similar to bar graph in the instructor notes below.

    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms

    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''

    plt.figure()
    turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly'].hist(color='green') # your code here to plot a historgram for hourly entries when it is raining
    turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly'].hist(color='blue') # your code here to plot a historgram for hourly entries when it is not raining
    return plt

'''
' 2) Cannot run Welch's T test because data is not normally distributed
'''
'''
' 3) Mann-Whitney U-Test
'''
def mann_whitney_plus_means(turnstile_weather):
    '''
    This function will consume the turnstile_weather dataframe containing
    our final turnstile weather data.

    You will want to take the means and run the Mann Whitney U-test on the
    ENTRIESn_hourly column in the turnstile_weather dataframe.

    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain

    You should feel free to use scipy's Mann-Whitney implementation, and you
    might also find it useful to use numpy's mean function.

    Here are the functions' documentation:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html

    You can look at the final turnstile weather data at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    entries_with_rain = turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly']
    entries_without_rain = turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly']
    with_rain_mean = np.mean(entries_with_rain)
    without_rain_mean = np.mean(entries_without_rain)
    U,p = scipy.stats.mannwhitneyu(entries_with_rain,
                                   entries_without_rain)
    return with_rain_mean, without_rain_mean, U, p # leave this line for the grader

'''
' 4) Ridership on Rainy vs. Nonrainy Days

We have used the Mann-Whitney U-test.
This non-parametric test evaluates the null hypothesis that two independent samples come from the same population.
The resulting p-value is roughly 0.03, <0.05 so we reject the null hypothesis (less than 5% of chances to observe data like this if samples 'rain' and 'not rain' come from same population).
Therefore the difference between entries when rain and not rain is statistically significant
'''

'''
' 5) Linear Regression
'''
"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.

    This can be the same code as in the lesson #3 exercise.
    """

    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]

    return intercept, params

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    # Select Features (try different features!)
    features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    # Perform linear regression
    intercept, params = linear_regression(features_array, values_array)

    predictions = intercept + np.dot(features_array, params)
    return predictions

'''
' 6) Plotting Residuals
'''
def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''

    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist()
    return plt

'''
' 7)  Compute R^2
'''
def compute_r_squared(data, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.

    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''

    # your code here
    SSR = np.sum((data-predictions)**2)
    mean = np.mean(data)
    SST = np.sum((data-mean)**2)
    r_squared=1-SSR/SST
    return r_squared

'''
' 8) Gradient Descent (optional)
'''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
INSTRUCTIONS:

Gradient descent in Python
--------------------------

Since ordinary least squares can be very slow for a large number of features,
you may wish to use gradient descent to predict subway ridership. Scikit Learn uses
gradient descent in its SGDRegressor (http://scikit-learn.org/0.14/modules/generated/sklearn.linear_model.SGDRegressor.html).
In the following exercise, you will use this library to perform linear regression on
the subway dataset, which will allow you to include a larger subset of the data without timing out.

Feature normalization
--------------------

In order for gradient descent to perform efficiently, each feature needs to be normalized -
that is, it must be rescaled so the mean is 0 and the standard deviation is 1.
Normalizing the features will result in different parameters in the model, but the correct parameters
can be recovered given the means and standard deviations of the original features. The provided
functions normalize_features and recover_params perform this operation for you in the exercise code.

When making predictions, you can use either the normalized features with the normalized parameters,
or the original features with the recovered parameters. When interpreting your parameters, however,
it is important to use the recovered versions.

Improving r-squared
-------------------

You may get a worse r-squared when using gradient descent instead of ordinary least squares.
This can happen for a variety of reasons, but a very common one is that gradient descent stopped
before reaching the minimum of the cost function. You can increase the number of iterations
SGDRegressor completes before stopping using the n_iter parameter.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(features):
    '''
    Returns the means and standard deviations of the given features, along with a normalized feature
    matrix.
    '''
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    '''
    Recovers the weights for a linear model given parameters that were fitted using
    normalized features. Takes the means and standard deviations of the original
    features, along with the intercept and parameters computed using the normalized
    features, and returns the intercept and parameters that correspond to the original
    features.
    '''
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def linear_regression_gradient_descent(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    """

    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################
    clf = SGDRegressor(n_iter=15)
    results = clf.fit(features, values)
    intercept= results.intercept_
    params = results.coef_

    return intercept, params

def predictions_gradient_descent(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~50%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features or fewer iterations.
    '''
    # Select Features (try different features!)
    features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    means, std_devs, normalized_features_array = normalize_features(features_array)

    # Perform gradient descent
    norm_intercept, norm_params = linear_regression_gradient_descent(normalized_features_array, values_array)

    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)

    predictions = intercept + np.dot(features_array, params)
    # The following line would be equivalent:
    # predictions = norm_intercept + np.dot(normalized_features_array, norm_params)

    return predictions