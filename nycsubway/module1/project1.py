import datetime

__author__ = 'jtorrente'
import pandas
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.linear_model import SGDRegressor
from ggplot import *

def entries_histogram(turnstile_weather, group_variable, label_if_0, label_if_1, color_if_0, color_if_1,
                      n_bins, max_range):

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
    # Number of columns in the histogram
    n_bins = 30
    # Max range of the histogram (x-axis) => Trucated at this point
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
    # Number of columns in the histogram
    n_bins = 30
    # Max range of the histogram (x-axis) => Trucated at this point
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
    # Number of columns in the histogram
    n_bins = 30
    # Max range of the histogram (x-axis) => Trucated at this point
    max_range = 9000

    # Variable Taking values 1/0 used to make the groups
    group_variable = "weekday"
    label_if_0="Weekend"
    label_if_1="Weekday"
    color_if_0='white'
    color_if_1='black'

    return entries_histogram(turnstile_weather,group_variable, label_if_0, label_if_1,
                             color_if_0, color_if_1, n_bins, max_range)

def rank_biserial_correlation(U, n1, n2):
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
                  geom_point() + stat_smooth(colour='blue', span=0.2) + scale_x_date(labels=date_format("%Y-%m-%d"), breaks="1 day") + \
                  ggtitle('Ridership by date') + \
                  xlab('Date (Weekdays in blue, Week-ends in red)') + ylab('Ridership')
    return plot

def plot_readership_by_date_hour(turnstile_weather):
    plot = ggplot(turnstile_weather,
                  aes(x='weekday', y='hour')) + \
                  geom_boxplot() + \
                  ggtitle('Ridership grouped by hour') + \
                  xlab('Hour of day') + ylab('Ridership')
    return plot


def plot_weather_data2(turnstile_weather):
    plot = ggplot(turnstile_weather,
                  aes(x='meanprecipi', y='meantempi', color='ENTRIESn_hourly', size='ENTRIESn_hourly')) + \
                  geom_point(position='jitter') + scale_color_gradient(low='#05D9F6', high='#5011D1') +\
                  ggtitle('Ridership by date and precipitations') + \
                  xlab('Mean precipitations') + ylab('Mean temperature') + \
                  scale_x_continuous()
    return plot


def analyse_weather_turnstile_data(datafile, show_plots):
    #0 Read datafile and add ordinal date value for plotting
    turnstile_weather = pandas.read_csv(datafile)
    turnstile_weather = add_date_column_for_plotting(turnstile_weather)
    print "-----------------------"
    print "DATA FOR NOT RAINY DAYS"
    print "-----------------------"
    print turnstile_weather[turnstile_weather.rain==0]
    print ""
    print "-------------------"
    print "DATA FOR RAINY DAYS"
    print "-------------------"
    print turnstile_weather[turnstile_weather.rain==1]
    #1 Do exploratory analysis

    if show_plots:
        plt = entries_histogram_rain(turnstile_weather)
        plt2 = entries_histogram_fog(turnstile_weather)
        plt.show()
        plt2.show()
        plt3 = entries_histogram_weekday(turnstile_weather)
        plt3.show()

    # Mann-Whitney U tests
    mann_whitney_plus_means(turnstile_weather, 'rain')
    mann_whitney_plus_means(turnstile_weather, 'fog')
    mann_whitney_plus_means(turnstile_weather, 'weekday')

    # Linear regression
    predictors = ['meanprecipi', 'meantempi', 'hour', 'weekday']
    predictions = predictions_gradient_descent(turnstile_weather, predictors)
    r_squared = compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predictions)
    if show_plots:
        plot_residuals(turnstile_weather, predictions).show()
    print "-----------------------------------------------"
    print "Linear model calculated using gradient_descent"
    print "Features used: "+str(predictors)
    print "R_SQUARED = " + str(r_squared)
    print "-----------------------------------------------"

    # Final plots
    if show_plots:
        print plot_readership_by_date_weekday(turnstile_weather)
        print plot_readership_by_date_hour(turnstile_weather)
        print plot_weather_data2(turnstile_weather)

analyse_weather_turnstile_data(r"../../data/turnstile_weather_v2.csv", True)