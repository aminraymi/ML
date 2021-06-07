import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

new_york_temperature_data = pd.read_csv('resources/nyc.csv')
new_york_temperature_data.columns = ['Date', 'Temperature', 'Anomaly']
new_york_temperature_data.Date = new_york_temperature_data.Date.floordiv(100)
pd.set_option('precision', 2)
linear_regression = stats.linregress(x=new_york_temperature_data.Date, y=new_york_temperature_data.Temperature)
predict_2022 = linear_regression.slope * 2022 + linear_regression.intercept
print('2022 prediction: ', predict_2022)
sns.set_style('whitegrid')
axes = sns.regplot(x=new_york_temperature_data.Date, y=new_york_temperature_data.Temperature)
axes.set_ylim(10, 50)
plt.axes = axes
plt.show()
