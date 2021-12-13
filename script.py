import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns',None)
# load rankings data
wood_rankings = pd.read_csv('Golden_Ticket_Award_Winners_Wood.csv')
#print(wood_rankings.head())
steel_rankings = pd.read_csv('Golden_Ticket_Award_Winners_Steel.csv')
#print(steel_rankings.head())

# function to plot rankings over time for 1 roller coaster
def plot_coaster_ranking(coaster_name, park_name, rankings_df):
  plt.clf()
  coaster_rankings = rankings_df[(rankings_df['Name'] == coaster_name) & (rankings_df['Park'] == park_name)]
  fig, ax = plt.subplots()
  ax.plot(coaster_rankings['Year of Rank'],coaster_rankings['Rank'])
  ax.set_yticks(coaster_rankings['Rank'].values)
  ax.set_xticks(coaster_rankings['Year of Rank'].values)
  ax.invert_yaxis()
  plt.title("{} Rankings".format(coaster_name))
  plt.xlabel('Year')
  plt.ylabel('Ranking')
  plt.show()

#plot_coaster_ranking('El Toro', 'Six Flags Great Adventure', wood_rankings)


# write function to plot rankings over time for 2 roller coasters here:

def plot_2coaster_ranking(coaster_name_1, coaster_name_2, park_name_1,park_name_2, rankings_df):
  plt.clf
  coaster_rankings_1 = rankings_df[(rankings_df['Name'] == coaster_name_1) & (rankings_df['Park'] == park_name_1)]
  coaster_rankings_2 = rankings_df[(rankings_df['Name'] == coaster_name_2) & (rankings_df['Park'] == park_name_2)]
  ax = plt.subplot()
  plt.plot(coaster_rankings_1['Year of Rank'],coaster_rankings_1['Rank'], color= 'blue')
  plt.plot(coaster_rankings_2['Year of Rank'],coaster_rankings_2['Rank'], color='red')
  ax.set_yticks(coaster_rankings_1['Rank'].values)
  ax.set_xticks(coaster_rankings_1['Year of Rank'].values)
  ax.invert_yaxis()
  plt.title("{} and {} Rankings".format(coaster_name_1, coaster_name_2))
  plt.xlabel('Year')
  plt.ylabel('Ranking')
  plt.legend([coaster_name_1, coaster_name_2])
  plt.show()

#plot_2coaster_ranking('El Toro', 'Boulder Dash', 'Six Flags Great Adventure', 'Lake Compounce', wood_rankings)
plt.clf()

# write function to plot top n rankings over time here:

def plot_n_rank(n, df):
 coaster_rank = df[df['Rank'] <= n]
 plt.clf()
 plt.figure(figsize = (10,10))
 ax= plt.subplot()
 for coaster in set(coaster_rank['Name']):
      ax.plot(coaster_rank['Year of Rank'][coaster_rank['Name'] == coaster],  coaster_rank        ['Rank'][coaster_rank['Name'] == coaster], label = coaster)
 ax.set_yticks(range(n+1))
 plt.xlabel('Year')
 plt.ylabel('Ranking')
 plt.title(f"Top {n} Rankings")
 plt.legend(loc = 4)
 plt.show()

#plot_n_rank( 5, steel_rankings)

# load roller coaster data here:
roller_coaster = pd.read_csv('roller_coasters.csv')
#print(roller_coaster.head())


# write function to plot histogram of column values here:
def plot_histogram(df, col_name):
 plt.clf()
 df_new = df[col_name].dropna()
 minimum_value = np.quantile(df_new, 0.01)
 maximum_value = np.quantile(df_new, 0.99)
 #print(maximum_value)
 plt.hist(df_new, range =(minimum_value ,maximum_value ), bins = 50, color= 'gold')
 plt.title(f'Histogram of {col_name}')
 plt.xlabel(f'{col_name}')
 plt.ylabel(f'Frequency of {col_name}')
 plt.legend(col_name, loc =4)
 plt.show()


#plot_histogram(roller_coaster, 'height')

plt.clf()

# write function to plot inversions by coaster at a park here:
def plot_inversions(df, park_name):
 #fig = plt.figure(figsize = (10,10))
 df_new = df[df['park'] == park_name].sort_values('num_inversions', ascending = False)
 ax = plt.subplot()
 plt.bar(df_new['name'], df_new['num_inversions'], color='magenta')
 ax.set_xticks(range(len(df_new['name'])))
 ax.set_xticklabels(df_new['name'], rotation= 30)
 plt.subplots_adjust( bottom =.3)
 plt.xlabel('name of roller coaster')
 plt.ylabel('num_inversions')
 plt.title(f'Num_inversions of roller coaster at {park_name}')
 #fig.tight_layout()
 plt.show()


#plot_inversions(roller_coaster, 'Parc Asterix')

plt.clf()

# write function to plot pie chart of operating status here:
def plot_pie(df):
 df_new =  df['status'][(df['status'] == 'status.operating') | (df['status'] == 'status.closed.definitely')].value_counts()
 print(df_new)
 plt.pie(df_new.values, autopct = '%d%%', labels =  df_new.index)
 plt.title('% operational status of roller coasters')
 plt.show()

#plot_pie(roller_coaster)
plt.clf()

# write function to create scatter plot of any two numeric columns here:

def plot_scatter(df, col_1, col_2):
  plt.scatter(df[col_1], df[col_2], marker = 'o', color='pink')
  plt.title(f'Comparison between {col_1} and {col_2} of roller coasters')
  plt.xlabel(f'{col_1}')
  plt.ylabel(f'{col_2}')
  plt.xlim((0,50))
  plt.show()


plot_scatter(roller_coaster,'height', 'num_inversions')
plt.clf()
