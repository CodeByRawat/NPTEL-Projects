    # -*- coding: utf-8 -*-
    """
    Created on Tue Dec 10 22:44:27 2024
    
    @author: sachi
    """
    #to work with dataframes
    import pandas as pd
    
    #to perform numerical operations
    import numpy as np
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #to paartition the data
    from sklearn.model_selection import train_test_split
    
    #importing library for logistic regression
    from sklearn.linear_model import LogisticRegression
    
    #importing performance matrice - accuracy score and confusion matrix
    from sklearn.metrics import accuracy_score,confusion_matrix
    
    
    ######################################
    #=====================================
    # IMPORTING DATA 
    #=====================================
    data_income = pd.read_csv('income.csv')
    data=data_income.copy()
    
    
    #Exploratory data analysis
    
    #1.Getting to know the data
    #2.Data preprocessing(missing values)
    #3.Cross tables and data Visualisation
    
    ######################################
    #=====================================
    # Getting to know the data
    #=====================================
    
    ## **** to check variable data type
    print(data.info())
    ## alternatively you can use this data.dtypes but this doesnt show null values data
    
    ## ** check for missing values
    
    data.isnull().sum()
    
    # summary of data 
    
    summary_num = data.describe()
    
    print(summary_num)
    
    
    # summarising categories 
    
    summary_cate= data.describe(include='O')
    
    #counting what are the unique values in the column
    
    data['JobType'].value_counts()
    data['occupation'].value_counts()
    
    
    ##chcecking for unqiue classes
    
    print(np.unique(data['JobType']))
    print(np.unique(data['occupation']))
    
    
    
    ##GO BACK AND READ THE DATA BY INCLUDING na_values[' ?']
    
    data=pd.read_csv('income.csv',na_values=[' ?'])
    
    ## NOW WHEN WE USE isnull() functino again we found that
    
    data.isnull().sum()
    
    
    ## we see that now 1809 is empty data in Jobtype + 7 =never worked category
    ## and occupation has 1816 missing data
    
    ## getting all the missing data in one variable
    
    missing=data[data.isnull().any(axis=1)]
    
    #now either you find a mechanism to fill the missing values
    #or you drop the missing values
    #we are dropping the missing values
    
    data2=data.dropna(axis=0)
    
    
    ##=================================
    ###  FINDING RELATIONSHIP BETWEEN INDEPENDENT VARIABLES
    ##=================================
    
    # correlation=data2.corr() # running this code will generate error 
    # because the correlation also contain string values
    #selecting only int and float value  ===>
    
    correlation=data.select_dtypes(include=['float64','int64']).corr()
    
    
    ## points to note 
    # 1.there is no strong correlation between any of the independent variable
    
    
    
    #==================================
    # gender proportion table :
    #==================================
    gender = pd.crosstab(index=data2['gender'],
                         columns='count',
                         normalize=True)
    print(gender)
    
    #==================================
    # gender vs salary status
    #==================================
    
    gender_salstat=pd.crosstab(index=data2['gender'],
                               columns=data2['SalStat'],margins=True,normalize='index')
    print(gender_salstat)
    
    #==================================
    # frequency distribution of 'salary status'
    #==================================
    
    sns.countplot(x='SalStat',data=data2)
    plt.show()
    #==================================
    # boxplot age vs sal status
    #==================================
    
    box_plot_age_vs_salstat=sns.boxplot(x='age',y='SalStat',data=data2,palette='dark')
    plt.show()
    
    
    #checking the salstat group  mean age 
    
    data2.groupby('SalStat')['age'].mean()
    
    
    ## INSIGHTS ===================================================
    
    #People with higher incomes (more than 50,000) tend to be 
    #older compared to those with lower incomes (50,000 or less).
    #This could suggest that experience, career progression,
    # or seniority—which typically come with age—play a role in
    # achieving higher income levels.
    
    #===============================================================
    
    #==================================
    # frequency distribution of job type in contrast with sal status
    #==================================
    
    
    sns.countplot(y='JobType',data=data2,hue='SalStat')
    plt.show()
    
    #==================================
    # frequency distribution of job type in contrast with sal status
    #==================================
    
    sns.countplot(x='EdType',data=data2,hue='SalStat')
    plt.xticks(rotation=90)
    plt.show()
    
    #==================================
    # relationship of job type in contrast with sal status using crosstab
    #==================================
    
    edtype_salstat=pd.crosstab(index=data2['EdType'], columns=data2['SalStat'],normalize='index')*100
    
    #==================================
    ############INSIGHTS##############
    # From the crosstab, you know that advanced degrees (Doctorate, Prof-school)
    # significantly increase the probability of earning > 50,000.
    
    # This insight can help in threshold tuning:
    
    # For individuals with higher education, your model can assign higher probabilities of earning > 50,000.
    # For lower education levels, the model can confidently predict ≤ 50,000.
    #==================================
    
    #==================================
    # frequency distribution of occupation in contrast with sal status
    #==================================
    
    
    sns.countplot(y='occupation',data=data2,hue='SalStat')
    plt.show()
    
    #==================================
    # relationship of occupatino in contrast with sal status using crosstab
    #==================================
    
    occupation_salstat=pd.crosstab(index=data2['occupation'], columns=data2['SalStat'],normalize='index')*100
    
    #==================================
    # frequency distribution of capital gains
    #==================================
    
    
    sns.boxplot(x='hoursperweek',data=data2,hue='SalStat')
    plt.show()
    
    # ==========================================================================================
    # ################     MODEL CREATION   ###################################################
    # ==========================================================================================
    
    #reindexing the salary status names to 0,1
    data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000' :0,' greater than 50,000':1})
    
        # *. map() Function:
        #     map() is used to replace values in the Salstat column based on a dictionary.
        #     The dictionary provided:
        #         'less than or equal to 50,000' → 0
        #         'greater than 50,000' → 1
        #     This effectively maps categories to numeric labels.
    
        # *. Target Column (SalStat):
        #     The resulting mapped values are stored in a new column called SalStat.
        #     SalStat will now contain:
        #         0 for individuals earning less than or equal to 50,000.
        #         1 for individuals earning greater than 50,000.
    
    
    ## converting all categorical values into 0's and 1's for training the model 
    ## drop_first =drops the first categorical values to avoid redundancy
    new_data=pd.get_dummies(data2,drop_first=True)
    
    #storing the column names from the new dummy data as a list
    columns_list=list(new_data.columns)
    
    #separting the input variale from the output variable
    
    features= list(set(columns_list)-set(['SalStat']))
    
    
    #storing the output values in y
    
    y=new_data['SalStat'].values
    
    # Use new_data['SalStat'] (pandas Series): when you
    #  want to maintain the index or work within pandas,
    #  as it keeps the data's metadata.
     
    # Use new_data['SalStat'].values (numpy array) : when 
    #  you need the data in a raw format, typically for
    #  machine learning or numerical calculations where 
    #  you don't need the index.
    
    #storing the feature values in a new variable
    x=new_data[features].values
    
    
    # splitting the data for Train and Test
    
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
    
    # train_test_split helps divide your dataset into training and testing parts.
    # It ensures the model is trained on one portion of the data and evaluated on a separate portion.
    # test_size=0.3 means 30% of data is for testing and 70% for training.
    # random_state=0 makes sure the data split is consistent every time you run the code.
    
    model=LogisticRegression()
    
    model.fit(train_x, train_y)
    
    #checking the coefficients of the model
    
    a=model.coef_
    
    ###Coefficients tell you how much each feature (like age, hours worked) influences
    #  the model’s prediction.
    #-->  Positive coefficients mean that as the feature increases, the predicted outcome is
    #  more likely to happen (e.g., higher salary).
    #--> Negative coefficients mean that as the feature increases, the predicted outcome is
    #  less likely to happen.
    
    intercept=model.intercept_
    
    prediction=model.predict(test_x)
    
    result_check=confusion_matrix(test_y,prediction)
    
    acc_score=accuracy_score(test_y,prediction)
    print(acc_score)
    
    
    # ==========================================
    # BUILDING A KNN CLASSIFIER MODEL
    # ==========================================
    
    # importing knn classifier
    
    from sklearn.neighbors import KNeighborsClassifier
    
    #creating the instance of knn classifier
    
    knn_classifier=KNeighborsClassifier(n_neighbors=6)
    
    
    knn_classifier.fit(train_x,train_y)
    
    prediction_knn=knn.predict(test_x)
    
    result_check_knn=confusion_matrix(test_y,prediction_knn)
    
    accuracy_knn=accuracy_score(test_y,prediction_knn)
    
    misclassified=(test_y != prediction_knn).sum()
    print(misclassified)
    
    for i in range(1,20):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x,train_y) #train_x = features,Train_y=output
        predict_knn_test=knn.predict(test_x)
        missclass_knn_test=(test_y!=predict_knn_test).sum()
        print(i,' ',missclass_knn_test)
