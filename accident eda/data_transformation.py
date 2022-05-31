import pandas as pd


def group_time_to_noon(data, time_column:str) -> pd.DataFrame:
    """
    Hours is segmented by
    - Night (from 12 am to 5.59 am)
    - Morning (from 6 am to 11.59 am)
    - Afternoon (from 12 pm to 5.59 pm)
    - Evening (from 6 pm to 11.59 pm)
    """
    new_data = data.copy(deep=True)
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    new_data['timebin'] = pd.cut(new_data[time_column].dt.hour, bins, labels = labels, right = False)
    return new_data

def group_time_to_hour(data, time_column):
    """
    Return bin_hours column có dạng [h0,h1)
    vd: 9h -> 9h59 => [9, 10)
    """
    new_data = data.copy(deep=True)
    bins = list(range(0, 25))
    new_data['bin_hours'] = pd.cut(new_data[time_column].dt.hour, bins, right = False)
    return new_data

def split_df_to_weekend(df, day_of_week_column:str, weekend_label:list, return_weekday=True):
    weeken_boolean = df[day_of_week_column].isin(weekend_label)
    time_weekend = df[weeken_boolean]
    if return_weekday:
        time_weekdays = df[~weeken_boolean]
        return time_weekdays, time_weekend
    return time_weekend

def group_date_to_month(data, date_column:str):
    """
    Trả về 1 cột 'month' group các ngày trong tháng thành 1 month
    """
    new_data = data.copy(deep=True)
    new_data['date'] = pd.to_datetime(new_data[date_column], infer_datetime_format=True)
    new_data['month'] = new_data['date'].dt.month
    return new_data