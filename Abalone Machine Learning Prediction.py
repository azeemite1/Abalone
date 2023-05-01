#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('abalone.csv')


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.value_counts('Sex')


# In[6]:


print(df.dtypes)


# In[35]:


df.hist(figsize=(10,10))


# In[52]:


sns.pairplot(df)


# In[53]:


plt.scatter(df['Length'], df['Age'])


# In[54]:


plt.scatter(df['Diameter'], df['Age'])


# In[55]:


plt.scatter(df['Shell weight'], df['Age'])


# In[56]:


plt.scatter(df['Shucked weight'], df['Age'])


# In[57]:


sns.regplot(x="Length", y="Age", data=df)


# In[7]:


# Create the fig, ax
fig, ax = plt.subplots()

# Create bar plot of first 1000 data entries 
ax.bar(df["Sex"][:1000], df["Rings"][:1000], color="peru")

#Add legend: x is horizonatal axis, y is vertical
ax.set(title="Sex and No. of Rings", 
       xlabel="Sex",
       ylabel="Number of rings");


# In[8]:


# Create a fig, ax
fig, ax = plt.subplots()

# Create a scatter plot with first 1000 data points
ax.scatter(df["Diameter"][:1000], df["Rings"][:1000], color="blue")

# Create a legend
ax.set(title="Diameter and Nr. of Rings",
       xlabel="Diameter",
       ylabel="No. of Rings");


# In[9]:


# Create fig, ax
fig, ax = plt.subplots()

# Create scatter plot of first 1000 data points
ax.scatter(df["Whole weight"][:1000], df["Rings"][:1000], color="teal")

#Create a legend
ax.set(title="Whole weight and Nr. of Rings",
       xlabel="Whole weight",
       ylabel="Number of rings");


# In[10]:


df['Sex'].value_counts()


# In[11]:


import matplotlib.pyplot as plt

data = {'M': 1528, 'I': 1342, 'F': 1307}
labels = list(data.keys())
values = list(data.values())

plt.bar(labels, values)
plt.title('Dataset')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[12]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["Sex_code"] = lb_make.fit_transform(df["Sex"])
df["Sex_code"].value_counts()


# In[13]:


df = df.drop('Sex', 1)


# In[14]:


df


# In[15]:


df['Age'] = df['Rings'] + 1.5


# In[16]:


df


# In[17]:


df = df.drop('Rings', 1)


# In[18]:


df


# In[19]:


corr_matrix = df.corr()


# In[20]:


plt.figure(figsize=(15,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')


# In[21]:


df.columns


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


# Define the list of features
features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Sex_code']

# Separate out the features and target
X = df.loc[:, features].values
y = df.loc[:,['Age']].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add the target column back to the scaled features
scaled_data = pd.DataFrame(X_scaled, columns=features)
scaled_data['Age'] = df['Age']


# In[ ]:





# In[24]:


scaled_corr = scaled_data.corr()


# In[25]:


plt.figure(figsize=(15,8))
sns.heatmap(scaled_corr, annot=True, cmap='YlGnBu')


# In[26]:


scaled_data.columns


# In[58]:


import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))


# In[ ]:


X = scaled_data['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Sex_code']

y = scaled_data["type_code"]


# In[61]:


df.var()


# In[29]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import numpy as np


# Separate out the features and target
X = scaled_data.drop('Age', axis=1)
y = scaled_data['Age']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a random forest classifier
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Perform permutation importance to rank the features
result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)

# Get the feature importances and sort them in descending order
importances = result.importances_mean
sorted_indices = np.argsort(importances)[::-1]


# In[30]:


# Create a dataframe with feature names and importances
importance_df = pd.DataFrame({'Feature': X.columns[sorted_indices], 'Importance': importances[sorted_indices]})

# Sort the dataframe by importance value in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 8))
heatmap_data = importance_df.pivot_table(index='Feature', values='Importance', aggfunc='mean')
heatmap_data = heatmap_data.sort_values(by='Importance', ascending=False)  # sort values in descending order
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[33]:


from eli5.sklearn import PermutationImportance

features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Sex_code']

target = ['Age']


# Store Randomforest feature importance details in the dataframe for further analysis
rf_feature_importance_df = pd.DataFrame()
rf_feature_importance_df["feature"] = features
rf_feature_importance_df["importance"] = clf.feature_importances_

# Perform permutaion feature importance and store details in the dataframe for further analysis
# Choosing roc_auc scoring metrics from sklearn.metrics.SCORERS.keys() for permutation importance.
perm_imp = PermutationImportance(clf, random_state=10)
perm_imp.fit(X_test, y_test)
# Store Permutaion feature importance details in the dataframe for further analysis
perm_imp_df = pd.DataFrame()
perm_imp_df["feature"] = features
perm_imp_df["importance"] = perm_imp.feature_importances_


# In[34]:


import eli5
eli5.show_weights(perm_imp, feature_names = X_test.columns.tolist())


# In[36]:


scaled_data.columns


# In[37]:


# Separate out the features and target
X = scaled_data.drop('Age', axis=1)
y = scaled_data['Age']


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[39]:


rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)


# In[41]:


from sklearn.metrics import r2_score

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Compute the R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 score: {r2:.2f}")


# In[42]:


# predict on the test set
y_pred = rf_model.predict(X_test)

# calculate the R-squared score
r2_score(y_test, y_pred)


# In[43]:


from sklearn.linear_model import LinearRegression
lr_model=LinearRegression() # initialzing the model
lr_model.fit(X_train, y_train)


# In[44]:


# predict on the test set
y_pred = lr_model.predict(X_test)

# calculate the R-squared score
r2_score(y_test, y_pred)


# In[46]:


from sklearn.metrics import mean_squared_error
# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f"R^2 score: {r2:.2f}")


# In[47]:


y_pred


# In[51]:


# Define input features
input_features = [[0.625, 0.485, 0.150, 1.164, 0.510, 0.252, 0.357, 1]]
scal_inf = scaler.transform(input_features)

# Make prediction using trained model
predicted_age = rf_model.predict(scal_inf)

# Print predicted age
print(predicted_age)


# In[60]:


pickle.dump(rf_model, open('rf_model.pkl', 'wb'))


# In[ ]:




