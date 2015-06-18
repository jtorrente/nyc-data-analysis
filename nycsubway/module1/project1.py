"""
    Copyright Javier Torrente (contact@jtorrente.info), and Udacity, 2015.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

    ************************************************************************

    NOTE: I developed the program contained in this file 'project1.py' as
    part of Udacity's nano-degree in data science. This file contains the
    code for the first project that I submitted after module 1. The code
    is publicly available in case it is useful, but if you happen to reuse
    it in your own nanodegree projects, do not forget to indicate that -
    otherwise it may be considered as cheating!

    ************************************************************************

    The source code in this file is mostly my own creation, although some methods
    (especially those used to calculate the linear regression model) were
    provided by the Udacity team. Source code contained in other parts of this
    project may also contain contributions from Udacity's team.
    This is the case of module "problemsets", as in there I've put all the code
    corresponding to problem sets 1-4 in the "Introduction to Data Science"
    course. Udacity provides the code half completed - the
    student (me in this case) is then instructed to complete it. Therefore
    the code contained in problemsets cannot be considered my sole contribution.
"""
import pandasql

__author__ = 'jtorrente+Udacity'

import datetime
import pandas
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import SGDRegressor
from ggplot import *
from nycsubway.module1.simple_log import SimpleLog

def data_cleanup(turnstile_weather, max_dif_entries_exits):
    """
    Makes a quick consistency check - for any given entry, the expected number of entries
    should be more or less similar to the number of exits. This cleanup procedure will
    calculate the difference between entries and exits and remove any elements where the
    difference exceeds the given threshold
    :param turnstile_weather: The data set (pandas data frame). Should have
            'ENTRIESn_hourly' and 'EXITSn_hourly' columns
    :param max_dif_entries_exits: The maximum tolerable difference between entries and
            exits
    :return: The number of rows deleted
             A clean version of turnstile_weather
    """
    turnstile_weather['ENTRIES_EXITSn_hourly_dif'] = turnstile_weather['ENTRIESn_hourly'] - \
                                                     turnstile_weather['EXITSn_hourly']

    entries_to_remove = turnstile_weather[abs(turnstile_weather.ENTRIES_EXITSn_hourly_dif) >= max_dif_entries_exits]
    nentries_to_remove = len(entries_to_remove)
    print "  NO ENTRIES TO REMOVE = " + str(nentries_to_remove)
    print "  MAX BEFORE CLEANUP = " + str(np.max(turnstile_weather['ENTRIESn_hourly']))
    turnstile_weather = turnstile_weather[abs(turnstile_weather.ENTRIES_EXITSn_hourly_dif) < max_dif_entries_exits]
    print " MAX AFTER CLEANUP = " + str(np.max(turnstile_weather['ENTRIESn_hourly']))
    return nentries_to_remove, turnstile_weather

#################################################################
#                        HISTOGRAMS                             #
#################################################################

def entries_histogram(turnstile_weather, group_variable, label_if_0, label_if_1, color_if_0, color_if_1,
                      n_bins, max_range):
    """
    Plots an histogram, using matplotlib.pyplot, of the number of hourly entries from the 'turnstile_weather'
    data set with two series:  one for rows where group_variable is 0, another one when group_variable is 1.
    Data is normalized to account for the differences in sizes of the groups
    :param turnstile_weather: The data set
    :param group_variable: Dichotomic variable (values 0|1) used to determine series
    :param label_if_0:  Label for the first series (group_variable == 0)
    :param label_if_1:  Label for the second series (group_variable == 1)
    :param color_if_0:  Color for the first series (group_variable == 0)
    :param color_if_1:  Color for the second series (group_variable == 1)
    :param n_bins:  Number of bins (columns) for each series
    :param max_range:   Cut-off value of ENTRIESn_hourly (any value beyond this upper limit will not be
                        rendered
    :return: The plot
    """
    # Labels for the chart
    x_label = "Number of entries per hour"
    y_label = "Frequency (Normed)"

    # Make it normed so different in sample sizes are not displayed
    normed = True

    fig_height = 15.0 # inches
    fig_width = 17.0 # inches
    plt.figure(figsize=(fig_width,fig_height))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    turnstile_weather[turnstile_weather[group_variable] == 0]['ENTRIESn_hourly'].hist(color=color_if_0, bins=n_bins,
                                                                           range=(0,max_range),
                                                                           label=label_if_0,
                                                                           normed=normed)

    turnstile_weather[turnstile_weather[group_variable] == 1]['ENTRIESn_hourly'].hist(color=color_if_1, bins=n_bins,
                                                                           range=(0,max_range),
                                                                           label=label_if_1,
                                                                           normed=normed, alpha = 0.5)
    plt.legend(loc='upper right')
    return plt

def entries_histogram_rain(turnstile_weather):
    """
    Draws an histogram using variable rain to create the groups (see entries_histogram for more details)
    :param turnstile_weather:  The data set
    :return:    The plot
    """
    # Number of columns in the histogram
    n_bins = 30
    # Max range of the histogram (x-axis) => Truncated at this point
    max_range = 9000

    # Variable Taking values 1/0 used to make the groups
    group_variable = "rain"
    label_if_0="Not rainy days"
    label_if_1="Rainy days"
    color_if_0='#FFCA28'
    color_if_1='#90CAF9'

    return entries_histogram(turnstile_weather,group_variable, label_if_0, label_if_1,
                             color_if_0, color_if_1, n_bins, max_range)

def entries_histogram_fog(turnstile_weather):
    """
    Draws an histogram using variable fog to create the groups (see entries_histogram for more details)
    :param turnstile_weather:  The data set
    :return:    The plot
    """
    # Number of columns in the histogram
    n_bins = 30
    # Max range of the histogram (x-axis) => Truncated at this point
    max_range = 9000

    # Variable Taking values 1/0 used to make the groups
    group_variable = "fog"
    label_if_0="Not foggy days"
    label_if_1="Foggy days"
    color_if_0='#FF7043'
    color_if_1='#9E9E9E'

    return entries_histogram(turnstile_weather,group_variable, label_if_0, label_if_1,
                             color_if_0, color_if_1, n_bins, max_range)

def entries_histogram_weekday(turnstile_weather):
    """
    Draws an histogram using variable 'weekday' to create the groups (see entries_histogram for more details)
    :param turnstile_weather:  The data set
    :return:    The plot
    """
    # Number of columns in the histogram
    n_bins = 30
    # Max range of the histogram (x-axis) => Truncated at this point
    max_range = 9000

    # Variable Taking values 1/0 used to make the groups
    group_variable = "weekday"
    label_if_0="Weekend"
    label_if_1="Weekday"
    color_if_0='white'
    color_if_1='black'

    return entries_histogram(turnstile_weather,group_variable, label_if_0, label_if_1,
                             color_if_0, color_if_1, n_bins, max_range)


#################################################################
#                       MANN-WHITNEY                            #
#################################################################

def rank_biserial_correlation(U, n1, n2):
    """
    Calculates the rank-biserial correlation of a Mann-Whitney test,
    using Wendt's formula r = 1-2*U/(n1*n2), to be used as an estimator of the effect size.
    Interpretation guide:

    r     |   Effect size   |
    ------ -----------------
    0.1      Small
    0.3      Medium size
    0.5      Large size

    Links:
    http://yatani.jp/teaching/doku.php?id=hcistats:mannwhitney
    https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Rank-biserial_correlation

    Reference:
    "Wendt, H. W. (1972). Dealing with a common problem in Social science: A simplified
    rank-biserial coefficient of correlation based on the U statistic. European Journal
    of Social Psychology, 2(4), 463-465. http://doi.org/10.1002/ejsp.2420020412"

    :param U: The statistic, as returned by mann_whitney_plus_means
    :param n1: Number of samples in first group
    :param n2:  Number of samples in second group
    :return: A value between 0 and 1 (although it can be larger)
    """
    return 1-(2*U)/(n1*n2)

def mann_whitney_plus_means(turnstile_weather, group_variable):
    """
    Runs a Mann-Whitney U test on the turnstile_weather data using ENTRIESn_hourly as dependant variable
    and the group_variable provided as independent variable.
    Results are also printed on screen.

    :param turnstile_weather: The data set (data frame object). Must contain a column ENTRIESn_hourly
                                and a column 'group_variable' with values 0,1
    :param group_variable: The name of the independent variable
    :return: The mean of ENTRIESn_hourly for entries with the condition
             The mean of ENTRIESn_hourly for entries without the condition
             The standard deviation of ENTRIESn_hourly for entries with the condition
             The standard deviation of ENTRIESn_hourly for entries without the condition
             The U statistic calculated
             The p-value
    """
    # Calculate difference between hourly entries and exits
    entries_with_condition = turnstile_weather[turnstile_weather[group_variable] == 1]['ENTRIESn_hourly']
    entries_without_condition = turnstile_weather[turnstile_weather[group_variable] == 0]['ENTRIESn_hourly']
    with_condition_mean = np.mean(entries_with_condition)
    without_condition_mean = np.mean(entries_without_condition)
    with_condition_sd = np.std(entries_with_condition)
    without_condition_sd = np.std(entries_without_condition)

    U, p = scipy.stats.mannwhitneyu(entries_with_condition,
                                   entries_without_condition)
    p *= 2

    # Calculate Rank-biserial correlation as an effect-size measure
    r = rank_biserial_correlation(U, len(entries_with_condition), len(entries_without_condition))

    # Print results before returning
    print "----------------------------------------------------------"
    print "Mann-Whitney U test results for condition: "+group_variable
    print "----------------------------------------------------------"
    print "  Mean with condition = "+str(with_condition_mean)
    print "  SD with condition = "+str(with_condition_sd)
    print "  Mean without condition = "+str(without_condition_mean)
    print "  SD without condition = "+str(without_condition_sd)
    print "  U statistic = "+str(U)
    print "  Two-tail p-value = "+str(p)
    print "  Rank biserial correlation (effect size) = "+str(r)
    print "----------------------------------------------------------"

    return with_condition_mean, without_condition_mean, with_condition_sd, without_condition_sd, U, p

#################################################################
#                     LINEAR REGRESSION                         #
#################################################################

def predictions_gradient_descent(turnstile_weather, predictors):
    """
    Estimates 'ENTRIESn_hourly' based on the predictors passed as argument,
    using gradient descent linear regression.

    This method is virtually the same used in problem set 3.
    :param turnstile_weather: The data frame object containing 'ENTRIESn_hourly'
                              and predictor variables
    :param predictors: A list with the name of the variables to be used
                       as predictors
    :return: A list with the estimations of 'ENTRIESn_hourly'
             The value of the intercept for the linear model
             The etha values of the linear model
    """
    # Features are selected from a list of predictors passed as argument
    features = turnstile_weather[predictors]
    print "***********************************************************"
    print "CORRELATIONS IN FEATURES - TO CHECK FOR MULTICOLINEARITY"
    print "***********************************************************"
    # Values
    values = turnstile_weather['ENTRIESn_hourly']
    features_and_values = features.join(values)
    print features_and_values.corr()
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(turnstile_weather['UNIT'], prefix='unit')
    features = features.join(dummy_units)

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

    return predictions, intercept, params


def normalize_features(features):
    """
    Normalizes the given list of features by adjusting for the mean and
    standard deviation as to produce a new list of normalized features
    N(0,1)
    :param features: Data-frame with the data to be normalized
    :return:    Means of the features
                Standard deviations of the features
                A list with the normalized features
    """
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def linear_regression_gradient_descent(features, values):
    """
    Calculates linear regression on the given features with the given values
    :param features: Array of numpy arrays with the features
    :param values: The values used for the linear regression
    :return:    The intercept
                The etha parameters
    """
    clf = SGDRegressor(n_iter=15)
    results = clf.fit(features, values)
    print results
    intercept = results.intercept_
    params = results.coef_

    return intercept, params

def recover_params(means, std_devs, norm_intercept, norm_params):
    """
    Undoes the normalization of normalize_features
    :param means: Array or list with means of the features
    :param std_devs: Array or list with the std. deviations of the features
    :param norm_intercept: The normalized intercept of the linear regression
    :param norm_params: Normalized parameters of the L.R.
    :return:    De-normalized intercept
                De-normalized parameters
    """
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def compute_r_squared(data, predictions):
    """
    Calculates the R^2 as an estimation of the error introduced by
    linear regression.

    :param data: Array-like with the real values of a variable
    :param predictions: Array-like with the estimated values of the variable
                        as generated by the L.R. procedure
    :return: The r^2 value, being a float number from 0 (0% accuracy)
             to 1 (100% accuracy)
    """
    SSR = np.sum((data-predictions)**2)
    mean = np.mean(data)
    SST = np.sum((data-mean)**2)
    r_squared = 1 - SSR / SST
    return r_squared

#################################################################
#                         RESIDUALS                             #
#################################################################

def analyse_residuals(show_plots, turnstile_weather):
    """
    Analyses residuals through three plots
    :param show_plots:
    :param turnstile_weather:
    :return:
    """
    if show_plots:
        plot_residuals_histogram(turnstile_weather).show()
        plot_residuals_probability_plot(turnstile_weather).show()
        print plot_residuals_by_datapoint(turnstile_weather)

    compare_large_small_residuals(turnstile_weather)

    # Get problematic units and analyze them
    #list_units_large_residuals(turnstile_weather, 5000, 20)
    print_residuals_by_unit(turnstile_weather)

    if show_plots:
        plot = ggplot(turnstile_weather,
                      aes(x='ENTRIES_EXITSn_hourly_dif', y='residuals')) + \
                      geom_point() + ggtitle('Residuals compared to ENTRIES_EXITSn_hourly_dif')
        print plot
        plot_dif_entries_exits_by_datapoint(turnstile_weather)

def compare_large_small_residuals(turnstile_weather):
        large_residuals = turnstile_weather[abs(turnstile_weather.residuals) > 5000]
        small_residuals = turnstile_weather[abs(turnstile_weather.residuals) <= 5000]
        print "************************************************"
        print "* DESCRIBING ENTRIES WITH LARGE RESIDUALS "
        print "************************************************"
        large_residuals.describe()
        print "************************************************"
        print "* DESCRIBING ENTRIES WITH SMALL RESIDUALS "
        print "************************************************"
        small_residuals.describe()

def list_units_large_residuals(turnstile_weather, residual_threshold=5000, occurrence_threshold=10):
    turnstile_weather['residual_type'] = turnstile_weather['residuals'].apply(lambda x: 1 if x>residual_threshold else 0)
    counts_large_residuals = turnstile_weather[turnstile_weather.residual_type==1]['UNIT'].value_counts()
    counts_large_residuals = counts_large_residuals[counts_large_residuals>occurrence_threshold]
    units_large_residuals = counts_large_residuals.keys().tolist()
    bad_predictions = turnstile_weather[~turnstile_weather.UNIT.isin(units_large_residuals)]
    print "******************************************"
    print "* DESCRIBING UNITS WITH LARGE RESIDUALS"
    print "******************************************"
    print bad_predictions.describe()

def print_residuals_by_unit(turnstile_weather):
    """
    Prints the avg residuals for each unit
    :param turnstile_weather: The data frame
    """
    grouped = turnstile_weather.groupby(turnstile_weather['UNIT'])
    unit_residuals_means = pandas.DataFrame()
    names = []
    residual_means = []
    for name, group in grouped:
        names.append(name)
        residual_means.append(np.mean(group['residuals']))
    unit_residuals_means['unit'] = names
    unit_residuals_means['mean_residuals'] = residual_means
    print "*******************************************"
    print " MEAN OF RESIDUALS FOR EACH UNIT "
    print "*******************************************"
    print unit_residuals_means


def plot_residuals_histogram(turnstile_weather):
    """
    Plots the error (residuals) introduced by the linear model
    as an histogram using matplotlib.pyplot
    :param turnstile_weather:   The data-frame containing the data. Must contain a column 'residuals'
    :param predictions: The array-like structure with the predictions
    (values) generated
    :return:    The plot generated
    """
    plt.figure()
    turnstile_weather['residuals'].hist()
    return plt

def plot_residuals_probability_plot(turnstile_weather):
    """
    Plots a probability plot of the residuals against the normal
    :param turnstile_weather Must contain a column 'residuals'
    """
    plt.figure()
    scipy.stats.probplot(turnstile_weather['residuals'], dist='norm', plot=plt)
    return plt

def plot_residuals_by_datapoint(turnstile_weather, min_limit=0, limit=50000):
    """
    Plots absolute value of residuals for all datapoints.
    :param turnstile_weather: The dataframe, with a residuals column
    :param min_limit: Index of first data entry to be shown
    :param limit:  Index of last data entry
    :return: The plot
    """
    data = pandas.DataFrame()
    data['residuals'] = abs(turnstile_weather['residuals'])
    index =[]
    for i in range(0, len(data)):
        index.append(i)
    data['index'] = index
    plot = ggplot(data,
                  # y='ENTRIESn_hourly', x='date_number', color='weekday')
                  aes(x='index', y='residuals')) + \
                  geom_point() + ggtitle('All residuals (without sign)') + xlim(low=min_limit, high=limit) +\
                  xlab('Different entries') + ylab('Residuals = |ENTRIESn_hourly-predictions|')

    print plot


def plot_dif_entries_exits_by_datapoint(turnstile_weather, min_limit=0, limit=50000):
    """
    Plots absolute value of difference between hourly entries and exits for all datapoints.
    :param turnstile_weather: The dataframe, with a residuals column
    :param min_limit: Index of first data entry to be shown
    :param limit:  Index of last data entry
    :return: The plot
    """
    data = pandas.DataFrame()
    data['ENTRIES_EXITSn_hourly_dif'] = abs(turnstile_weather['ENTRIES_EXITSn_hourly_dif'])
    index =[]
    for i in range(0, len(data)):
        index.append(i)
    data['index'] = index
    plot = ggplot(data,
                  # y='ENTRIESn_hourly', x='date_number', color='weekday')
                  aes(x='index', y='ENTRIES_EXITSn_hourly_dif')) + \
                  geom_point() + ggtitle('Differences between hourly entries and exits (without sign)') + xlim(low=min_limit, high=limit) +\
                  xlab('Different entries') + ylab('|ENTRIESn_hourly - EXITSn_hourly|')

    print plot


#################################################################
#                     ADDITIONAL PLOTS                          #
#################################################################

def add_date_column_for_plotting(turnstile_weather):
    """
    Adds a new column 'date_number' as a python date, calculated from column DATEn,
    that is useful for exploiting ggplot's capabilities to display dates in charts
    :param turnstile_weather: The data frame
    :return: The new version of the data
    """

    # turnstile_weather = turnstile_weather.loc[:1000,:]
    turnstile_weather.is_copy = False
    d = turnstile_weather.loc[:, 'DATEn']

    d = d.apply(lambda x: datetime.date(int(x[6:] if len(x) > 8 else "20"+x[6:]), int(x[:2]), int(x[3:5])).toordinal())

    turnstile_weather['date_number'] = d
    return turnstile_weather

def plot_histogram_ridership_by_rain(turnstile_weather):
    p = ggplot(turnstile_weather,
                  aes(x='ENTRIESn_hourly', fill='rain', color='rain')) + \
           geom_histogram(binwidth=100, alpha=0.6) +\
           xlim(0, 7500) +\
           ggtitle("ENTRIESn_hourly histogram (rainy days in blue, not rainy in red)") + \
           xlab('ENTRIESn_hourly') + ylab('Frequency')
    return p

def plot_readership_by_date_weekday(turnstile_weather):
    """
    Plots a line chart showing in different colours ridership for each
    date
    :param turnstile_weather: The data
    :return: The plot
    """
    grouped = turnstile_weather.groupby('date_number')
    data = pandas.DataFrame()
    date_numbers = []
    total_hourly_entries = []
    weekdays = []
    for date_number, group in grouped:
        date_numbers.append(date_number)
        total_hourly_entries.append(np.sum(group['ENTRIESn_hourly']))
        weekday = group.iloc[0]['weekday']
        weekdays.append(weekday)

    data['date_number'] = date_numbers
    data['total_hourly_entries'] = total_hourly_entries
    data['weekday'] = weekdays
    print data

    p = ggplot(data,
                  aes(y='total_hourly_entries', x='date_number', color='weekday')) + \
                  geom_point() + geom_line() + \
                  scale_x_date(labels=date_format("%Y-%m-%d"), breaks="1 day") + \
                  ggtitle('Ridership by date') + \
                  xlab('Date (Weekdays in blue, Week-ends in red)') + ylab('Total number of entries in the day')
    return p

def plot_meanprecepi_meantempi(turnstile_weather):
    """
    Plots a heat map chart showing how ridership changes (using intensity of colours to convey
    information) depending on mean precipitations and temperature
    :param turnstile_weather: The data
    :return: The plot
    """
    p = ggplot(data=turnstile_weather,aesthetics=aes(x='meanprecipi', y='meantempi', color='ENTRIESn_hourly', size='ENTRIESn_hourly')) + \
                  geom_point() + scale_color_gradient(low='#05D9F6', high='#5011D1') +\
                  ggtitle('Ridership by mean precipitations and temperature') + \
                  xlab('Mean precipitations') + ylab('Mean temperature') + \
                  scale_x_continuous()

    return p

#################################################################
#                           MAIN                                #
#################################################################

def analyse_weather_turnstile_data(datafile, show_plots):
    simple_log = SimpleLog(7)
    # 0 Read datafile, cleanup and preparation: Add ordinal date value for plotting
    turnstile_weather = pandas.read_csv(datafile)
    turnstile_weather = add_date_column_for_plotting(turnstile_weather)
    turnstile_weather['ENTRIES_EXITSn_hourly_dif'] = turnstile_weather['ENTRIESn_hourly'] - \
                                                     turnstile_weather['EXITSn_hourly']
    simple_log.log_object(turnstile_weather[turnstile_weather.rain == 0],
                          "DATA FOR NOT RAINY DAYS")
    simple_log.log_object(turnstile_weather[turnstile_weather.rain == 1],
                          "DATA FOR RAINY DAYS")

    #1 Do exploratory analysis
    simple_log.log("Printing three histograms that show\ndistribution of rain, fog and weekday\n"
                   "Close all three pop up windows to continue.\n")
    if show_plots:
        plt = entries_histogram_rain(turnstile_weather)
        plt2 = entries_histogram_fog(turnstile_weather)
        plt3 = entries_histogram_weekday(turnstile_weather)
        plt.show()
        plt2.show()
        plt3.show()

    # 2) Mann-Whitney U tests
    simple_log.log("Results from three Mann-Whitney U-tests\nto check if rain, fog and weekday have\n"
                   "effect on ENTRIESn_hourly\n")
    mann_whitney_plus_means(turnstile_weather, 'rain')
    mann_whitney_plus_means(turnstile_weather, 'fog')
    mann_whitney_plus_means(turnstile_weather, 'weekday')


    # 3) Calculate correlations for predictors
    predictors = ['meanprecipi', 'meantempi', 'hour', 'weekday']
    rho_meanprecipi, p_meanprecepi = scipy.stats.spearmanr(turnstile_weather['ENTRIESn_hourly'],
                          turnstile_weather['meanprecipi'])
    rho_meantempi, p_meantempi = scipy.stats.spearmanr(turnstile_weather['ENTRIESn_hourly'],
                          turnstile_weather['meantempi'])
    simple_log.log("Calculating Spearman's rho correlations between meanprecipi and meantempi and\n"
                   "ENTRIESn_hourly respectively\n"
                   "MEANPRECIPI => rho = "+str(rho_meanprecipi)+"    p = "+str(p_meanprecepi)+"\n"
                   "MEANTEMPI => rho = "+str(rho_meantempi)+"    p = "+str(p_meantempi)+"\n")

    # 4) Linear regression
    predictions, intercept, ethas = predictions_gradient_descent(turnstile_weather, predictors)
    turnstile_weather['residuals'] = turnstile_weather['ENTRIESn_hourly'] - predictions
    r_squared = compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predictions)
    simple_log.log("Linear model calculated using gradient_descent\n"
                   "If show_plots is true, the histogram of residuals will\n"
                   "be shown\n."
                   "Features used: "+str(predictors)+"\n"
                   "R_SQUARED = " + str(r_squared)+"\n"
                   "INTERCEPT = "+str(intercept)+"\n"
                   "ETHAS = " + str(ethas)+"\n")
    analyse_residuals(show_plots, turnstile_weather)

    # 5) Final plots
    if show_plots:
        simple_log.log("Three more plots will be shown if show_plots is true.\n"
                       "The last one may take up to one minute to complete.\n")
        print plot_histogram_ridership_by_rain(turnstile_weather)
        print plot_readership_by_date_weekday(turnstile_weather)
        # The last figure can take up to one minute to calculate
        # print plot_meanprecepi_meantempi(turnstile_weather)
    else:
        simple_log.log("Procedure complete.\n"
                       "Run 'analyse_weather_turnstile_data' again with show_plots=True\n"
                       "to get charts and plots.\n")


analyse_weather_turnstile_data(r"../../data/turnstile_weather_v2.csv", False)