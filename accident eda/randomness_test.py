from statsmodels.sandbox.stats.runs import runstest_1samp 


def test_runs_randomness(data):
    #Perform Runs test
    z, alpha = runstest_1samp(data, correction=False)
    if alpha > 0.05:
        return f'Alpha is {alpha}, data points are random'
    else:
        return f'Alpha is {alpha}, data points are not random'
    