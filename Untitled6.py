#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv',header=1 )
df.head()


# In[3]:


df.shape


# In[4]:


df.drop([122,123],inplace=True)
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)


# In[5]:


df.shape


# In[6]:


df.tail(124)


# In[7]:


df.loc[:122, 'region'] = 'bejaia'
df.loc[122:, 'region'] = 'Sidi-Bel Abbes'


# In[8]:


df.columns


# In[9]:


df.columns=[i.strip() for i in df.columns]
df.columns


# In[10]:


for feature in [ 'DC']:
    df[feature]=df[feature].str.replace(" ","")    
    


# In[11]:


for feature in [ 'Classes']:
    df[feature]=df[feature].str.replace(" ","")    


# In[12]:


df['Classes']  


# In[13]:


df['FWI'].unique()


# In[14]:


df[df['FWI']=='fire   '].index


# In[15]:


df['FWI' ].mode()


# In[16]:


df.loc[165,'FWI']='0.4'


# In[17]:


df.info() 


# In[18]:


df.isnull().sum()


# In[19]:


df[df['Classes']=='nan'].index


# In[20]:


df.loc[165,'Classes']='fire'


# In[21]:


df['Classes'].unique()


# In[22]:


df['Classes']


# In[23]:


dfs=  { 'notfire':0,'fire':1}
df['Classes']= df['Classes'].map(dfs)


# In[24]:


df['Classes']
df['Classes'].unique()


# In[25]:


df['region'].unique()


# In[26]:


df['region']=df['region'].str.replace('bejaia','0')
df['region']=df['region'].str.replace('Sidi-Bel Abbes','1')


# In[27]:


df['day']=df['day'].astype(int)
df['month']=df['month'].astype(int)
df['year']=df['year'].astype(int)
df['Temperature']=df['Temperature'].astype(int)
df['RH']=df['RH'].astype(int)
df['Ws']=df['Ws'].astype(int)
df['Rain']=df['Rain'].astype(float)
df['FFMC']=df['FFMC'].astype(float)
df['DMC']=df['DMC'].astype(float)
df['ISI']=df['ISI'].astype(float)
df['BUI']=df['BUI'].astype(float)
df['DC']=df['DC'].astype(float)
df['FWI']=df['FWI'].astype(float)
 
df['region']=df['region'].astype(int)


# In[28]:


df['Classes']=df['Classes'].astype(int)


# In[29]:


data=df.copy()
data.head()


# In[30]:


df.info()


# In[31]:


data.describe()


# In[32]:


data.cov()


# In[33]:


plt.figure(figsize=(20,40), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=15 :     
        ax = plt.subplot(8,2,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show() 


# In[34]:


plt.figure(figsize=(15,15))
plt.suptitle('Multivariate Analysis', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
sns.pairplot(data,  diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)


# In[35]:


plt.figure(figsize=(20,15))
sns.heatmap(data.corr(),cmap='CMRmap',annot=True)


# In[36]:


data.Classes.value_counts()


# In[37]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)
sns.barplot(x='region',y='Classes',data=data)


# In[38]:


plt.subplots(figsize=(20,10))
sns.histplot('distribution of temperature',x=data.Temperature,color='b',kde=True)
plt.title('Temperature distribution',weight='bold',fontsize='30',pad=20 )


# In[39]:


matplotlib.rcParams['figure.figsize']=(20,10)
sns.barplot(x='Temperature',y='Classes',data=data)


# In[40]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)

sns.barplot(x="region",y="Rain",data=data)


# In[41]:


matplotlib.rcParams['figure.figsize']=(20,10)
sns.barplot(x='region',y='Temperature',data=data)


# In[42]:


num_col=[feature for feature in data.columns if data[feature].dtype!='0']
num_col


# In[43]:


plt.figure(figsize=(20,40))
for i in enumerate(num_col):
    plt.subplot(8,2,i[0]+1)
    sns.set(rc={'figure.figsize':(8,10)})
    sns.regplot(data=data,x=i[1],y='Classes')
    plt.xlabel('Classes')
    plt.title('{} vs Classes'.format(i[1]))


# In[44]:


plt.figure(figsize=(20,45), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=15 :      
        ax = plt.subplot(8,2,plotnumber)
        sns.boxplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show()


# In[45]:


def outliers(data,Temp):
    iqr=1.5*(np.percentile(data[Temp],75)-np.percentile(data[Temp],25))
    data.drop(data[data[Temp]>(iqr+np.percentile(data[Temp],75))].index,inplace=True)
    data.drop(data[data[Temp]<(np.percentile(data[Temp],25)-iqr)].index,inplace=True)
outliers(data,'Temperature')   


# In[46]:


sns.boxplot(data.Temperature )


# In[47]:


outliers(data,'Ws')       


# In[48]:


sns.boxplot(data.Ws )


# In[49]:


outliers(data,'Rain')   
sns.boxplot(data.Rain )


# In[50]:


outliers(data,'FFMC')   
sns.boxplot(data.FFMC )


# In[51]:


outliers(data,'DMC')   
sns.boxplot(data.DMC )


# In[52]:


outliers(data,'DC')   
sns.boxplot(data.DC )


# In[53]:


outliers(data,'ISI')   
sns.boxplot(data.ISI )


# In[54]:


outliers(data,'BUI')   
sns.boxplot(data.BUI )


# In[55]:


outliers(data,'FWI')   
sns.boxplot(data.FWI )


# In[56]:


X = data.drop(columns = ['Classes'])
y = data['Classes']


# In[57]:


print(X)


# In[58]:


print(y
     )


# In[59]:


plt.figure(figsize=(20,50), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=15 :
        ax = plt.subplot(8,2,plotnumber)
        sns.stripplot(y,X[column])
    plotnumber+=1
plt.tight_layout()


# In[60]:


from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[63]:


from sklearn.linear_model  import  LogisticRegression
log_reg= LogisticRegression()


# In[64]:


from sklearn.model_selection import GridSearchCV
parameter={'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]}
classifier_regressor=GridSearchCV(log_reg,param_grid=parameter,scoring='accuracy',cv=5)


# In[65]:


classifier_regressor.fit(X_train,y_train)
print(classifier_regressor.get_params().keys())


# In[66]:


print(classifier_regressor.best_params_)


# In[67]:


print(classifier_regressor.best_score_)


# In[68]:


y_pred=classifier_regressor.predict(X_test)
y_pred


# In[69]:


plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50)


# In[70]:


from sklearn.metrics import accuracy_score,classification_report 
s=accuracy_score(y_pred,y_test)
print(s)


# In[71]:


print(classification_report(y_pred,y_test))


# In[72]:


from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_pred,y_test)
print(conf_mat)


# In[73]:


from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_test,y_pred)
print(auc)


# In[74]:


import math
MSE = np.square(np.subtract(y_test,y_pred)).mean() 

RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)


# SVM

# In[75]:


from sklearn.svm import SVC 


# In[76]:


classifier=SVC(kernel='Sigmoid') 


# In[77]:


df_svm=df.copy()
X = df_svm.drop(columns = ['Classes'])
y = df_svm['Classes']
from sklearn. model_selection import train_test_split


# In[78]:


X


# In[79]:


y


# In[80]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)


# In[81]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_s=sc.fit_transform(X_train)
X_test_s=sc.fit_transform(X_test)


# In[82]:


X_train_s.shape


# In[83]:


classifier=SVC(kernel='rbf',random_state=3)
classifier.fit(X_train_s,y_train )


# In[84]:


y_pred=classifier.predict(X_test_s)



# In[85]:


y_pred 


# In[86]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[87]:


accuracy_score(y_test,y_pred)


# In[88]:


from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_test,y_pred)
print(auc)
support_vectors_per_class =classifier.n_support_
print(support_vectors_per_class)


# In[89]:


ax = plt.gca()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


red_sample=df.sample(frac=0.7)
X=red_sample.iloc[:,0:2]
y=red_sample['species']
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')


from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

model=SVC(kernel='linear').fit(X, y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

from mpl_toolkits import mplot3d
#setting the 3rd dimension with RBF centered on the middle clump
r = np.exp(-(X ** 2).sum(1))
ax = plt.subplot(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')

model=SVC(kernel='rbf').fit(X, y)




# In[90]:


support_vectors_per_class =classifier.n_support_
print(support_vectors_per_class)


# In[91]:


X_train.shape


# In[92]:


X_test.shape


# In[93]:


support_vector_indices = classifier.support_
print(support_vector_indices)


# 
# # ## parametric tunin:

# In[94]:


from sklearn.model_selection import GridSearchCV

 
param_grid = {'C': [0.001,0.1, 1, 10, 100, 1000],
			'gamma': [1, 0.1, 0.01, 0.001, 0.0001,0.000001],
			'kernel': ['rbf','sigmoid','linear']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


grid.fit(X_train, y_train)


# In[95]:


y_pred_ = grid.predict(X_test)


# In[96]:


y_pred


# In[97]:


accuracy_score(y_test,y_pred_ )


# In[98]:


from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_test,y_pred_)
print(auc)


# In[99]:


import math
MSE = np.square(np.subtract(y_test,y_pred)).mean() 

RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)


# # # Decision Tree
# 

# In[100]:


DT_Data=df.copy()


# In[101]:


DT_Data.head()


# In[102]:


X = DT_Data.drop(columns = ['Classes'])
y = DT_Data['Classes']


# In[103]:


from sklearn.tree import DecisionTreeClassifier


# In[104]:


decisiontree= DecisionTreeClassifier()
decisiontree.fit(X_train,y_train)


# In[105]:


from sklearn import tree
plt.figure(figsize=(25,20))
tree.plot_tree(decisiontree,filled=True,rounded=True)
plt.show()


# In[106]:


param_dict={
    "criterion":["gini","entropy"],
    "max_depth":[1,2,3,4,5,6,None]
    
}


# In[107]:


from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(decisiontree,param_grid=param_dict,cv=10,n_jobs=1)


# In[108]:


grid.fit(X_train,y_train)


# In[109]:


grid.best_estimator_


# In[110]:


grid.best_score_


# In[111]:


grid.best_params_


# In[112]:


accuracy_score(y_test,y_pred_ )


# In[113]:


from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_test,y_pred_)
print(auc)


# In[114]:


import math
MSE = np.square(np.subtract(y_test,y_pred)).mean() 

RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)


# In[ ]:




