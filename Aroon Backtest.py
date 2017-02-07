## This function will accept a ticker (string),
## start date and end date (as strings in the form %Y-%m-%d),
## and a performance threshold, and give back a table of summary stats

def aroon_backtest(ticker, start, end, threshold):
    import datetime as dt
    import pandas as pd
    import pandas_datareader.data as web
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import style
    from matplotlib.finance import candlestick_ohlc
    import matplotlib.dates as mdates
    style.use('ggplot')

    # Store start/end dates
    start = dt.datetime.strptime(start, "%Y-%m-%d")
    end = dt.datetime.strptime(end, "%Y-%m-%d")

    # Download Data
    df = web.DataReader(ticker, 'yahoo', start, end)

    # Create Aroon Indicators
    df['AroonHigh'] = pd.rolling_apply(df['High'], 15, lambda x:  x.tolist().index(max(x)) / float(14) * 100)
    df['AroonLow'] = pd.rolling_apply(df['Low'], 15, lambda x:  x.tolist().index(min(x)) / float(14) * 100)

    # Identify Crossover
    df['Cross'] = df['AroonHigh'] - df['AroonLow']

    # Drop NA
    df.dropna(inplace=True)

    # Store Cross Periods in list
    Holds = []
    HoldsDates = []
    hold = []
    holdDates = []
    for x in range(0, len(df)-1):
        if df['Cross'][x] > 0 and (df['Cross'][x - 1] > 0 or df['Cross'][x + 1] > 0):
            hold.append(df['Close'][x])
            holdDates.append(df.index[x])
            if df['Cross'][x + 1] <= 0:
                Holds.append(hold)
                HoldsDates.append(holdDates)
                hold = []
                holdDates = []

    # Summary Statistics
    Lengths = []
    Performances = []
    Performances3day = []
    PerformanceTotal = 1

    for hold in Holds:
        # Lengths
        Lengths.append(len(hold))

        # Performance
        performance = hold[-1] / hold[0] - 1
        Performances.append(performance)

        # Overall Performance
        PerformanceTotal = PerformanceTotal * (performance + 1)

        if len(hold) >= 3:
            # Performance 3 day
            performance3day = hold[2] / hold[0] - 1
            Performances3day.append(performance3day)

    PerformanceTotal = PerformanceTotal - 1

    # Threshold Probability
    ThresholdProbability = sum(1 for i in Performances if i > threshold) / len(Performances3day)
    PositiveProbability = sum(1 for i in Performances if i > 0) / len(Performances3day)

    # 3 day return threshold probability
    ThresholdProbability3day = sum(1 for i in Performances3day if i > threshold) / len(Performances3day)
    PositiveProbability3day = sum(1 for i in Performances3day if i > 0) / len(Performances3day)

    # Create Sum-Up Statistics
    AvgLength = np.array(Lengths).mean()
    AvgPerformance = np.array(Performances).mean()
    AvgPerformance3day = np.array(Performances3day).mean()

    # Create Benchmark
    PerformanceBenchmark = df['Close'][-1] / df['Close'][0] - 1


    Summary_Matrix = [['Metric', 'Figure'],
                      ['Benchmark Performance', '{0:.0%}'.format(PerformanceBenchmark)],
                      ['Total Performance', '{0:.0%}'.format(PerformanceTotal)],
                      ['Relative Performance', '{0:.0%}'.format(PerformanceTotal - PerformanceBenchmark)],
                      ['Average Performance', '{0:.0%}'.format(AvgPerformance)],
                      ['Average 3 Day Performance', '{0:.0%}'.format(AvgPerformance3day)],
                      ['Average Length', str(round(AvgLength, 1)) + " Days"],
                      ['3D Performance Threshold ' + str('{0:.0%}'.format(threshold)) + ' Probability', '{0:.0%}'.format(ThresholdProbability3day)],
                      ['3D Performance Positive Probability', '{0:.0%}'.format(PositiveProbability3day)],
                      ['Performance Threshold ' + str('{0:.0%}'.format(threshold)) + ' Probability', '{0:.0%}'.format(ThresholdProbability)],
                      ['Performance Positive Probability', '{0:.0%}'.format(PositiveProbability)]]

    fig, ax = plt.subplots()
    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.table(cellText=Summary_Matrix[1:11], colLabels=Summary_Matrix[0], loc='center')

    # Reset index and convert to MDate, for charting purposes
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)

    # Create subplot grid
    ax1 = plt.subplot2grid((15, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((15, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((15, 1), (10, 0), rowspan=2, colspan=1)
    # Set Title
    plt.suptitle(ticker, fontsize=28, fontweight='bold')
    # Format axes
    # plot 1
    ax1.xaxis_date()
    ax1.xaxis.set_visible(False)
    # plot 2
    ax2.xaxis_date()
    # plot3
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.axis('off')

    # Shade "hold" areas on candlestick chart
    for dates in HoldsDates:
        ax1.axvspan(dates[0], dates[-1], alpha=0.2, color='green')

    # Create subplots
    candlestick_ohlc(ax1, df[['Date', 'Open', 'High', 'Low', 'Close']].values, width=1, colorup='g', colordown='r')
    ax2.plot(df['Date'], df['AroonHigh'], color='g')
    ax2.plot(df['Date'], df['AroonLow'], color='r')
    ax3.table(cellText=Summary_Matrix[1:11], colLabels=Summary_Matrix[0], loc='center')
