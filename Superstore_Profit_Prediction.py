#!/usr/bin/env python
# coding: utf-8

# # Superstore Profit Prediction (Machine Learning Project)
# 
# End-to-end ML project analyzing retail sales data and predicting profit using regression models.
# 

# In[1]:


#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


# # **READING THE DATA**

# In[2]:


df=pd.read_csv('Supermart Grocery Sales - Retail Analytics Dataset.csv')


# In[3]:


#display the first five rows of the data
df.head().\
style.background_gradient(subset='Sales',cmap='Blues').\
background_gradient(subset='Profit',cmap='Blues').\
background_gradient(subset='Discount',cmap='Greens')


# In[4]:


#display the rows and columns
df.shape


# In[5]:


df.info()


# In[6]:


# Dataframe has no NaN values. Lets look at its statistical view . using describe()
df.describe()


# In[7]:


df.describe(include='object')


# In[8]:


#column names of table
df.columns


# In[9]:


# lets check data type of columns/attributes
df.dtypes


# There are categorical and numerical data in this dataset. 

# Now, we will take a look at the columns into two categories: Categorical and numerical data. For this section, we will have a general understanding of the most important columns. Next section we will explore correlation among columns.
# 
# Categorical columns:
# 
# Order ID          
# Customer Name     
# Category          
# Sub Category      
# City              
# Order Date       
# Region   
# State
# 
# Numerical columns:
# 
# Sales
# Profit
# Discount

# # Data Cleaning

# In[24]:


df['Order Date'].head(20)


# In[27]:


df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)


# In[28]:


#changed to date data type
df.info()


# In[30]:


#Extract month from the order date
#Extract month from the order date
df['month_no'] = df['Order Date'].dt.month
df['Month'] = pd.to_datetime(df['Order Date']).dt.strftime('%B')
df['year'] = df['Order Date'].dt.year


# In[31]:


#check the data to view the added columns
df.head()


# In[32]:


# Let"s count the number of missing (NaN) values in each column 
df.isnull().sum()


# There is no missing values
# 

# In[33]:


#count the number of duplicate rows 
df.duplicated().sum()


# # Exploratory Data Analysis

# In[37]:


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='RdYlGn', fmt='.2f')


# Analysis:
# 
# 1. **Sales and Profit Correlation (0.61)**:
#    - A positive correlation of 0.61 between sales and profit suggests a moderately strong positive relationship between these two variables.
#    - As sales increase, profit tends to increase as well, and as sales decrease, profit tends to decrease. The relationship is not perfect but is still considered positive.
# 
# 2. **Discount and Profit Correlation (0.00)**:
#    - There is no significant trend or pattern suggesting that changes in discount lead to predictable changes in profit. 
#    - In practical terms, this correlation suggests that there is  no linear relationship between the amount of discount applied and the resulting profit. Changes in discount have minimal impact on profit.
#    
#  
# 3. **Discount and Sales Correlation (-0.01)**:
#    - A very low negative correlation of -0.01 between discount and sales suggests a very weak negative relationship between these two variables.
#    - In practical terms, this correlation indicates that there is almost no linear relationship between the amount of discount applied and the level of sales. Changing the discount has little impact on sales.

# # **## Bivariate Analysis**

# We will be comparing other features to profit, and sales to get a visual idea about what affects the profit and sales most.

# # 1. Category

# In[38]:


#we want to find the total sales and profit by category
# firstly, we group by Category and get the total number of sales and profit for each category
category_df= pd.DataFrame(df.groupby(["Category"])[["Sales","Profit"]].sum().sort_values('Sales', ascending=False))
category_df


# In[39]:


fig, axes = plt.subplots(2,1, figsize=(8,14))
sns.set_theme(style="darkgrid")
axes[0].set_title("Category by Profit")
axes[1].set_title("Category by Sales")

sns.barplot(x=category_df.index,
           y=category_df['Profit'],
           data= category_df,
             palette='viridis',
           ax = axes[0]);

sns.barplot(x=category_df.index,
           y=category_df['Sales'],
           data= category_df,
             palette='viridis',
           ax = axes[1]);

axes[0].set_xticklabels(category_df.index, rotation=45)
axes[1].set_xticklabels(category_df.index, rotation=45)

plt.tight_layout(pad=4)


# The categories of Eggs, Meat & Fish, and Snacks make the most significant contributions to the sales, accounting for approximately 30% of the total sales. These categories demonstrate strong sales potential, suggesting that allocating additional resources and investments in them could yield promising returns for the company.

# In[40]:


sub_category_df = pd.DataFrame(df.groupby(['Sub Category'])[['Profit', 'Sales']].sum().sort_values('Sales', ascending=False))
sub_category_df


# In[41]:


fig, axes = plt.subplots(2,1, figsize=(10,18))

sns.set_theme(style="darkgrid")
axes[0].set_title("Sub-Category vs Profit")
axes[1].set_title("Sub-Category vs Sales")

sns.barplot(y=sub_category_df.index,
           x='Profit',
           data=sub_category_df,
           palette='icefire',
           ax=axes[0])

sns.barplot(y=sub_category_df.index,
           x='Sales',
           data=sub_category_df,
           palette='icefire',
           ax=axes[1])

plt.tight_layout(pad=3);


# The bar chart findings indicate that Health Drinks and Soft Drinks are the most frequently purchased products. Customers show a clear preference for these items. The company should Ensure that Health Drinks and Soft Drinks are prominently displayed and easily accessible in stores or on the company's website. 

# ## 2. Region

# In[42]:


region_df = pd.DataFrame(df.groupby(['Region'])[['Profit', 'Sales']].sum())
region_df


# In[43]:


fig, axes = plt.subplots(1,2, figsize=(14,8))

sns.set_theme(style="darkgrid")
axes[0].set_title("Region vs Profit")
axes[1].set_title("Region vs Sales")

sns.barplot(x=region_df.index,
           y='Profit',
           data=region_df,
           palette='Paired',
           ax=axes[0])

sns.barplot(x=region_df.index,
           y='Sales',
           data=region_df,
           palette='Paired',
           ax=axes[1])

plt.tight_layout(pad=1);


# Based on the findings from the bar chart analysis, it is evident that both the East and West regions make the most substantial contributions to the company's financial performance. These regions stand out as key drivers of the company's success, highlighting their pivotal roles in generating revenue and profitability."

# ## 3. Cities

# In[44]:


#Step 1: Extract relevant columns
city_sales = df[['City', 'Sales']]
cities_df = city_sales.groupby('City').sum().sort_values('Sales', ascending=False)
top5_sales = cities_df.head()

city_sales = df[['City', 'Profit']]
cities1_df = city_sales.groupby('City').sum().sort_values('Profit',ascending = False)
top5_profit = cities1_df.head()


# In[45]:


fig, axes = plt.subplots(2,1, figsize=(7, 10))

axes[0].set_title("Profit of top 5 ")
axes[1].set_title("Sales of top 5 ")

sns.barplot(y=top5_sales.index,
           x=top5_sales['Sales'],
           data=top5_sales,
            color='#4169E1',
           ax=axes[0])

sns.barplot(y=top5_profit.index,
           x=top5_profit['Profit'],
           data=top5_profit,
           color='green',
           ax=axes[1])

plt.tight_layout(pad=3);


# The top 5 City with the highest profit between 2015 to 2018 were Kanyakumari,Vellore, Bodi, Tirunelveli and Perambalur. Kanyakumari had  0.71 Sales,Velllore had 0.68 Sales and Bodi had  0.67 Sales or about 4.5% sales respectively.

# ## 4. Month

# In[51]:


# Sum up sales by month
monthly_sales = df.groupby('month_no')['Sales'].sum().reset_index()
monthly_profit= df.groupby('month_no')['Profit'].sum().reset_index()


# In[53]:


# Create a figure with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Month labels
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# First subplot: Sales by Month
axes[0].plot(monthly_sales['month_no'], monthly_sales['Sales'], marker='o', linewidth=2)
axes[0].set_title('Sales by Month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Sales')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_labels)
axes[0].grid(True)

# Second subplot: Profit by Month
axes[1].plot(monthly_profit['month_no'], monthly_profit['Profit'], marker='o', color='green', linewidth=2)
axes[1].set_title('Profit by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Profit')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_labels)
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# September and November are the highest profitability month

# ## 5. Year

# In[55]:


#we want to find the Yearly Sales and profit
# we group by Year and get the total number of sales for each year
Yearly_Sales=df.groupby("year")["Sales"].sum()

# we group by Year and get the total number of sales for each year
Yearly_Profit=df.groupby("year")["Profit"].sum()


# In[57]:


# Create a figure with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figsize as needed

# Define custom colors for the pie slices
colors_sales = ['lightcoral', 'lightblue', 'lightgreen', 'lightsalmon']
colors_profit = ['gold', 'lightpink', 'lightseagreen', 'lightgray']

# First subplot: Sales by Year
sales_pie, sales_text, sales_autotext = axes[0].pie(
    Yearly_Sales, labels=Yearly_Sales.index, autopct='%1.1f%%',
    colors=colors_sales, startangle=90, shadow=True, wedgeprops={'edgecolor': 'black'}
)
axes[0].set_title('Sales by Year')

# Second subplot: Profits by Year
profit_pie, profit_text, profit_autotext = axes[1].pie(
    Yearly_Profit, labels=Yearly_Profit.index, autopct='%1.1f%%',
    colors=colors_profit, startangle=90, shadow=True, wedgeprops={'edgecolor': 'black'}
)
axes[1].set_title('Profits by Year')

# Customize legend labels and position
axes[0].legend(sales_pie, Yearly_Sales.index, title='Years', loc='upper right')
axes[1].legend(profit_pie, Yearly_Profit.index, title='Years', loc='upper right')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()


# The sales and profit increase as the year increases which shows the company devised better and suitable plan to increase sales and profit at each point in time.

# In[ ]:





# ## Machine Learning Models
# 
# now we will build a machine learning model that predicts **Profit** based on **Sales** and **Discount**.
# 
# This helps us understand how business factors influence profitability and allows us to estimate profit from sales activity.

# In[58]:


X = df[['Sales', 'Discount']]
y = df['Profit']


# ## Feature Selection
# 
# We selected two features:
# - Sales: total revenue generated from transactions
# - Discount: discount applied to sales
# 
# The target variable is Profit, which we want to predict.
# 
# These variables were chosen because they have a direct business relationship with profit generation.

# In[59]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ## Train-Test Split
# 
# The dataset is split into:
# - Training set (80%): used to train the model
# - Testing set (20%): used to evaluate performance
# 
# This ensures the model is tested on unseen data, preventing overfitting and ensuring generalization.

# In[60]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


# ## Linear Regression
# 
# This model assumes a linear relationship between inputs and profit.
# 
# It serves as a baseline model for comparison.

# In[61]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


# ## Random Forest
# 
# This model captures non-linear relationships using multiple decision trees.
# 
# It is expected to perform better than Linear Regression.

# In[62]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("Linear Regression Results")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))

print("\nRandom Forest Results")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))


# In[ ]:


## Model Evaluation Results

Two models were tested: Linear Regression and Random Forest.

### Results:
- Linear Regression performed better with R² = 0.36 and lower errors (MAE: 158.78, RMSE: 196.18)
- Random Forest performed worse with R² = 0.22 and higher errors

### Interpretation:
The data shows a mostly linear relationship between Sales, Discount, and Profit. Therefore, the simpler Linear Regression model was more effective than the more complex Random Forest model.

### Conclusion:
Model performance suggests that profit in this dataset is primarily driven by linear relationships rather than complex interactions.


# ## ACTUAL VS PREDICTED (VISUALIZATION)

# In[63]:


import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit (Random Forest)")
plt.show()


# In[64]:


importance = rf.feature_importances_

plt.bar(X.columns, importance)
plt.title("Feature Importance (Random Forest)")
plt.show()


# ## Feature Importance
# 
# This shows which features influence profit most.
# 
# Sales has higher importance than Discount, meaning it drives profit more strongly in this dataset.

# ## Conclusion
# 
# The machine learning models were used to predict Profit based on Sales and Discount.
# 
# Linear Regression performed better than Random Forest, achieving a higher R² score (0.36) and lower error values.
# 
# This suggests that the relationship between the variables is mostly linear rather than complex.
# 
# Sales had a stronger influence on Profit compared to Discount.
# 
# Overall, this project demonstrates data analysis, feature engineering, and machine learning modeling skills applied to a real business dataset.
