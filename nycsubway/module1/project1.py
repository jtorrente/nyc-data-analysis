"""
    Copyright Javier Torrente (contact@jtorrente.info), 2015.

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

    The source code in this file is entirely my own creation. However, source
    code contained in other parts of this project may contain contributions
    from Udacity's team. This is the case of module "problemsets", as in there
    I've put all the code corresponding to problem sets 1-4 in the "Introduction
    to Data Science" course. Udacity provides the code half completed - the
    student (me in this case) is then instructed to complete it. Therefore
    the code contained in problemsets cannot be considered my sole contribution.
"""
__author__ = 'jtorrente'

import datetime
import pandas
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.linear_model import SGDRegressor
from ggplot import *
from nycsubway.module1.simple_log import SimpleLog

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
    Calculates the rank biserial correlation of a Mann-Whitney test,
    To be used as an estimator of the effect size. 
    :param U:
    :param n1:
    :param n2:
    :return:
    """
    return 1-(2*U)/(n1*n2)

def mann_whitney_plus_means(turnstile_weather, group_variable):
    entries_with_condition = turnstile_weather[turnstile_weather[group_variable] == 1]['ENTRIESn_hourly']
    entries_without_condition = turnstile_weather[turnstile_weather[group_variable] == 0]['ENTRIESn_hourly']
    with_condition_mean = np.mean(entries_with_condition)
    without_condition_mean = np.mean(entries_without_condition)

    U,p = scipy.stats.mannwhitneyu(entries_with_condition,
                                   entries_without_condition)

    # Calculate Rank-biserial correlation as an effect-size measure
    r = rank_biserial_correlation(U, len(entries_with_condition), len(entries_without_condition))

    # Print results before returning
    print "----------------------------------------------------------"
    print "Mann-Whitney U test results for condition: "+group_variable
    print "----------------------------------------------------------"
    print "  Mean with condition = "+str(with_condition_mean)
    print "  Mean without condition = "+str(without_condition_mean)
    print "  U statistic = "+str(U)
    print "  p-value = "+str(p)
    print "  Rank biserial correlation (effect size) = "+str(r)
    print "----------------------------------------------------------"

    return with_condition_mean, without_condition_mean, U, p

def plot_residuals(turnstile_weather, predictions):
    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist()
    return plt

def compute_r_squared(data, predictions):
    SSR = np.sum((data-predictions)**2)
    mean = np.mean(data)
    SST = np.sum((data-mean)**2)
    r_squared=1-SSR/SST
    return r_squared

def normalize_features(features):
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def linear_regression_gradient_descent(features, values):
    clf = SGDRegressor(n_iter=15)
    results = clf.fit(features, values)
    intercept = results.intercept_
    params = results.coef_

    return intercept, params

def predictions_gradient_descent(dataframe, predictors):
    # Select Features (try different features!)
    # features = dataframe[['rain', 'precipi', 'hour', 'meantempi']]
    features = dataframe[predictors]
    #
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

def add_date_column_for_plotting(turnstile_weather):
    # turnstile_weather = turnstile_weather.loc[:1000,:]
    turnstile_weather.is_copy = False
    d = turnstile_weather.loc[:, 'DATEn']

    d = d.apply(lambda x: datetime.date(int(x[6:] if len(x) > 8 else "20"+x[6:]), int(x[:2]), int(x[3:5])).toordinal())

    turnstile_weather['date_number'] = d
    return turnstile_weather

def plot_readership_by_date_weekday(turnstile_weather):
    plot = ggplot(turnstile_weather,
                  aes(y='ENTRIESn_hourly', x='date_number', color='weekday')) + \
                  geom_point() + stat_smooth(colour='blue', span=0.2) + \
                  scale_x_date(labels=date_format("%Y-%m-%d"), breaks="1 day") + \
                  ggtitle('Ridership by date') + \
                  xlab('Date (Weekdays in blue, Week-ends in red)') + ylab('Ridership')
    return plot


def plot_meanprecepi_meantempi(turnstile_weather):
    plot = ggplot(turnstile_weather,
                  aes(x='meanprecipi', y='meantempi', color='ENTRIESn_hourly', size='ENTRIESn_hourly')) + \
                  geom_point() + scale_color_gradient(low='#05D9F6', high='#5011D1') +\
                  ggtitle('Ridership by mean precipitations and temperature') + \
                  xlab('Mean precipitations') + ylab('Mean temperature') + \
                  scale_x_continuous()
    return plot


def analyse_weather_turnstile_data(datafile, show_plots):
    simple_log = SimpleLog(7)
    #0 Read datafile and add ordinal date value for plotting
    turnstile_weather = pandas.read_csv(datafile)
    turnstile_weather = add_date_column_for_plotting(turnstile_weather)
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
    predictions = predictions_gradient_descent(turnstile_weather, predictors)
    r_squared = compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predictions)
    simple_log.log("Linear model calculated using gradient_descent\n"
                   "If show_plots is true, the histogram of residuals will\n"
                   "be shown\n."
                   "Features used: "+str(predictors)+"\n"
                   "R_SQUARED = " + str(r_squared)+"\n")
    if show_plots:
        plot_residuals(turnstile_weather, predictions).show()

    # 5) Final plots
    if show_plots:
        simple_log.log("Two more plots will be shown if show_plots is true.\n"
                       "The last one may take up to one minute to complete.\n")
        print plot_readership_by_date_weekday(turnstile_weather)
        # The last figure can take up to one minute to calculate
        print plot_meanprecepi_meantempi(turnstile_weather)

analyse_weather_turnstile_data(r"../../data/turnstile_weather_v2.csv", True)