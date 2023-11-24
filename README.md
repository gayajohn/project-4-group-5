# Predicting Short-Term Stock Price Movement with Machine Learning

## The Objective

Can we use past stock data, along with macroeconomic indicators to predict if stock prices will increase or decrease?
This project aims to predict if *tomorrow's* price will increase or decrease, based on data from a short window prior. For each row of data, we pick a stock and date within a given time period. For example, we will look at Apple's price in June 3rd, 2019, and gather data on the Apple stock prior to this date, and try to predict if the price will go up, down or stay the same on June 4th, 2019.

## Topic Selection

Financial markets are dynamic and influenced by diverse factors, making traditional analysis methods less effective. Machine learning offers the ability to detect subtle patterns and trends in vast datasets.
Embracing machine learning aligns with industry trends, allowing for a competitive edge through the development of predictive models that adapt and learn from historical market behavior.

However, it is important to note, while machine learning models offer valuable insights, they are viewed as complementary tools rather than infallible predictors, recognizing the inherent uncertainties and risks associated with predicting stock prices. As we approach this project, we recognize that it will be difficult to achieve high levels of accuracy given the erratic nature of the stock market and our resource limitations. That being said, this exercise can be fruitful to figure out which models may yield best accuracies and identifying which factors may be most important. 

## Repository Structure

- data_files: folder that contains .csv files that are relevant to our data
- data_procurement.ipynb: script to generate data
- data_exploration.ipynb: script to explore data
- ml_codes_compiled.ipynb: script that runs machine learning models

## The Data

In order to ensure data availability:
- We limited our stocks to Large Cap North American stocks
- We limited our data window to January 2015 to October 2023

### Data Gathering

In order to get an extensive picture of the market, and gather all the relevant data to make our predictions, we needed to use multiple sources of data so that we get it for free.

- Stock Screening
  
We screened North American stocks, that are Large Cap in nature using NASDAQ's free stock screener.

- Stock Price Information
  
We used a library on Python called Yahoo_fin, that scrapes Yahoo Finance. We can get price, return and volatility for this database.

- Company Financials
  
We used chromedriver to scrape a website called stockanalysis.come, where we could find financial statements for each stock. We used most recent quarterly reports to get the financial ratios for our data.

- Economic Indicators
  
We used Alphavantage's API to access economic indicators for the given time period. This API is directly linked to FRED or the Federal Economic Database.

- Technical Indicators
  
We used Finnhub.io's API to access technical indicators for each stock. They also have a python library.

## Data Exploration

We used SQL to store our database, and read it into our scripts for each model.

### The Features

- **industry**: A categorical value, representing the sector that the company is in.
#### Weekly Returns:
- **wr1**: Weekly Return (1), a float value that is the the weekly return of the week prior to the selected date.
- **wr2**: Weekly Return (2), a float value that is the weekly return of the week two weeks prior to the selected date.
- **wr3**: Weekly Return (3), a float value that is the weekly teturn of the week three weeks prior to the selected date.
- **wr4**: Weekly Return (4), a float value that is the weekly teturn of the week four weeks prior to the selected date.
#### Trading Volumes
- **vol1**: Trading Volumne (1), a float value that is the trading volume on the selected date.
- **vol2**: Trading Volumne (2), a float value that is the trading volume on the day before the selected date.
- **vol3**: Trading Volumne (3), a float value that is the trading volume two days before selected date.
- **vol4**: Trading Volumne (4), a float value that is the trading volume three days before the selected date.
#### Financial Ratios
- **pe_ratio**: The price–earnings ratio, a float value, is the ratio of a company's share price to the company's earnings per share.
- **debt_to_equity**: The debt-to-equity ratio indicates the relative proportion of shareholders' equity and debt used to finance a company's assets. A float value.
- **quick_ratio**: The quick ratio, a float value, is the ratio of cash, marketable securities, and accounts receivable by the current liabilities.
- **total_shareholder_return**: Total Shareholder Return is a financial metric that represents the total value received by a shareholder through capital appreciation and any dividends or distributions received over a specific period.  This is a float value and is represented in % units.
- **profit_margin**: Profit margin is a financial metric that expresses the percentage of a company's net income relative to its total revenue. This is a float value and is represented in % units.
- **free_cash_margin**: Free cash flow margin is a financial metric that measures the percentage of a company's total revenue that translates into free cash flow after accounting for operating expenses and capital expenditures. This is a float value and is represented in % units.
#### Risk
- **volatility**: Stock volatility, measured by standard deviation, is a statistical metric that quantifies the degree of variation of a trading price series for a given security over a specific period. This float value represents historical volatlity of the stock from its origin to the selected date.
#### Economic Indicators
- **cpi**: The Consumer Price Index is a measure that examines the average change in prices paid by consumers for a basket of goods and services over time. It is a widely used indicator for inflation. This float value represents the CPI in %, for the month and year of the selected date. 
- **interest_rate**: The interest rate in this model is the The federal funds rate, the rate at which depository institutions (banks and credit unions) lend reserve balances to each other overnight. It is a float value represented in %, for the selected date.
- **unemployment rate**: The unemployment rate for the month and year of the selected date. This is a float value in %.
#### Technical Indicators
- **sma**: Simple Moving Average represents the average of a set of prices over a specified period, with the goal of smoothing out short-term price fluctuations to identify trends. It is a float value.
- **ema**: The Exponential Moving Average is a type of weighted moving average that gives more weight to recent prices. It is a float value.
- **rsi**: The Relative Strength Index is a momentum oscillator that measures the speed and change of price movements. It is a float value.
#### Label / Classifier
- **label**: This is our y or predicted value. It is a binary classifier. We compare the price of the selected date to the next trading date and give the row a value of 1 if the price stayed the same or increased and a value of 0 if the price decreased. In essence, we are using multiple financial and economic features to try to predict short term price changes. 

The stock dataset has 1914 records, each with 23 attributes.

### Analyzing the Labels

The data is distributed evenly between 1 and 0.

![image](https://github.com/gayajohn/project-4-group-5/assets/135036996/d918106a-e17a-4645-b1c8-3b8153743ea1)

### Distribution of data by Industry

![image-1](https://github.com/gayajohn/project-4-group-5/assets/135036996/cdb98b96-e57c-410a-aa05-80e980e9507d)

### Feature Distribution

![image-3](https://github.com/gayajohn/project-4-group-5/assets/135036996/fcdd1c4f-36f2-447a-afbd-ca0ab96c1f47)

### Feature Correlation Matrix

![image-4](https://github.com/gayajohn/project-4-group-5/assets/135036996/9b54d189-c2c6-45e3-874c-accf7dcd6f6b)

### Feature Skewness

![image-5](https://github.com/gayajohn/project-4-group-5/assets/135036996/4227d625-22d8-421e-90af-5e497eaa5b02)
 
## Machine Learning Models

We performed the following steps:

- Preprocessing the data
- Test 12 different ML models
- Compare the accuracry of all models, measured by f1 score

### Preprocessing the Data

- Separate the label and features 
- Encode numerical columns, in this case, the industry column
- Split the data into training and testing data
- Scaling the data based on the training data

### List of Models


#### 1. Random Forest (RF) 

RF Accuracy: 0.5031315240083507

![image-6](https://github.com/gayajohn/project-4-group-5/assets/135036996/9f95fba9-edb3-49f0-ad51-41d95a9a0da9)

Random Forest Classification Report: Training Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       682
           1       1.00      1.00      1.00       753

    accuracy                           1.00      1435
 


Random Forest Classification Report: Testing Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.46      0.41      0.43       220
           1       0.54      0.58      0.56       259

    accuracy                           0.50       479
 

We can see a clear overfitting problem here

#### 2. XG Boost (XGB)

XGB Accuracy (Binary Classification): 0.5615866388308977

![image-7](https://github.com/gayajohn/project-4-group-5/assets/135036996/8c9c3bae-cd61-48f1-8a62-741513f38264)

XGBoost Classification Report: Training Data
---------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       682
           1       1.00      1.00      1.00       753

    accuracy                           1.00      1435


XGBoost Classification Report: Testing Data
--------------------------------------------
              precision    recall  f1-score   support

           0       0.46      0.41      0.43       220
           1       0.54      0.58      0.56       259

    accuracy                           0.50       479
 

Here too, there's an overfitting problem

#### 3. Tensor Flow (TF)

We used sigmoid as the output layer's activation function as it's ideal for binary classification.

**Model 1**
    
    Layer (type)                Output Shape              Param #   
    =================================================================

    dense_12 (Dense)            (None, 80)                2800      
                                                                 
    dense_13 (Dense)            (None, 30)                2430      
                                                                 
    dense_14 (Dense)            (None, 20)                620       
                                                                 
    dense_15 (Dense)            (None, 1)                 21        
                                                                 
    =================================================================

Loss: 3.4279682636260986, Accuracy: 0.519832968711853

**Model 2**

    
    Layer (type)                Output Shape              Param #   
    =================================================================
     dense_16 (Dense)            (None, 100)               3500      
                                                                 
     dense_17 (Dense)            (None, 60)                6060      
                                                                 
     dense_18 (Dense)            (None, 30)                1830      
                                                                 
     dense_19 (Dense)            (None, 1)                 31        
                                                                 
    =================================================================

Loss: 1.9603184461593628, Accuracy: 0.5407097935676575

#### 4. Logistic Regression (LR)

LR Accuracy: 0.5323590814196242

![image-8](https://github.com/gayajohn/project-4-group-5/assets/135036996/34ef962a-0cb9-4581-b8d3-a8927ed53955)

Logistic Regression Classification Report: Training Data
----------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       682
           1       1.00      1.00      1.00       753

    accuracy                           1.00      1435
   


Logistic Regression Classification Report: Testing Data
---------------------------------------------------------
              precision    recall  f1-score   support

           0       0.49      0.36      0.42       220
           1       0.56      0.68      0.61       259

    accuracy                           0.53       479
   

Another case of over-fitting.

#### 5. Linear Discriminant Analysis (LDA)

LDA Accuracy: 0.5302713987473904

![image-9](https://github.com/gayajohn/project-4-group-5/assets/135036996/e08b3042-af90-4455-99a6-cfd3b1d34362)

Linear Discriminant Analysis Classification Report: Training Data
-------------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.53      0.38      0.44       682
           1       0.55      0.69      0.61       753

    accuracy                           0.54      1435
   


Linear Discriminant Analysis Classification Report: Testing Data
-------------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.48      0.35      0.41       220
           1       0.55      0.68      0.61       259

    accuracy                           0.53       479
  

#### 6. Neural Network (MLP)

MLP Accuracy: 0.5177453027139874

![image-10](https://github.com/gayajohn/project-4-group-5/assets/135036996/eed7e060-5b21-47bf-8e05-76de9d35754c)

Multi-layer Perceptron Classification Report: Training Data
-------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.94      0.92      0.93       682
           1       0.93      0.94      0.93       753

    accuracy                           0.93      1435



Multi-layer Perceptron Classification Report: Testing Data
--------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.47      0.46      0.47       220
           1       0.55      0.56      0.56       259

    accuracy                           0.52       479

Overfitting observed.

#### 7. K-Nearest Neighbors Algorithm (KNN)

A quick loop showed us the optimal n_neighbours to be 1.

KNN Accuracy: 0.5532359081419624

![image-11](https://github.com/gayajohn/project-4-group-5/assets/135036996/f100b62a-7791-4afe-90b4-dcc72aa1dd2d)

K-Nearest Neighbors Classification Report: Training Data
----------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       682
           1       1.00      1.00      1.00       753

    accuracy                           1.00      1435



K-Nearest Neighbors Classification Report: Testing Data
----------------------------------------------------------
              precision    recall  f1-score   support

           0       0.51      0.52      0.52       220
           1       0.59      0.58      0.59       259

    accuracy                           0.55       479

#### 8. Decision Tree Algorithm (DT) 

DT Accuracy: 0.5052192066805845

![image-12](https://github.com/gayajohn/project-4-group-5/assets/135036996/0f86a194-2fd0-46a8-b11f-4d4f865c8814)

Decision Tree Classification Report: Training Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.49      0.56      0.53       682
           1       0.55      0.48      0.51       753

    accuracy                           0.52      1435



Decision Tree Classification Report: Testing Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.47      0.52      0.49       220
           1       0.55      0.49      0.52       259

    accuracy                           0.51       479


#### 9. Bagging Decision Tree (BGT)

BGT Accuracy: 0.5052192066805845

![image-13](https://github.com/gayajohn/project-4-group-5/assets/135036996/b228d6ab-c466-4d21-9ce4-a299a0462f47)

Bagging Decision Tree Classification Report: Training Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       682
           1       1.00      0.98      0.99       753

    accuracy                           0.99      1435



Bagging Decision Tree Classification Report: Testing Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.47      0.57      0.52       220
           1       0.56      0.46      0.50       259

    accuracy                           0.51       479


Still seeing overfitting.

#### 10. Gradient Boosting Classifier (GBT)

GBT Accuracy: 0.5156576200417536

![image-14](https://github.com/gayajohn/project-4-group-5/assets/135036996/64051ba2-036c-43ab-8e26-e73479ffba52)

Gradient Boosting Classification Report: Training Data
-------------------------------------------------------
              precision    recall  f1-score   support

           0       0.49      0.56      0.53       682
           1       0.55      0.48      0.51       753

    accuracy                           0.52      1435



Gradient Boosting Classification Report: Testing Data
-------------------------------------------------------
              precision    recall  f1-score   support

           0       0.47      0.52      0.49       220
           1       0.55      0.49      0.52       259

    accuracy                           0.51       479


#### 11. Support Vector Machine (SVM)

SVM Accuracy: 0.53

![image-15](https://github.com/gayajohn/project-4-group-5/assets/135036996/d4b99466-5945-4b05-b289-994cb06bced8)


Support Vector Machine Classification Report: Training Data
-------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.67      0.38      0.48       682
           1       0.60      0.83      0.69       753

    accuracy                           0.62      1435



Support Vector Machine Classification Report: Testing Data
-------------------------------------------------------------
              precision    recall  f1-score   support

           0       0.48      0.28      0.35       220
           1       0.55      0.75      0.63       259

    accuracy                           0.53       479



#### 12. Naive Bayes Classifier (NB) 

NB Accuracy: 0.46346555323590816

![image-16](https://github.com/gayajohn/project-4-group-5/assets/135036996/1c565b9a-b0a2-4eee-b8e7-5febb6ff34a2)

Naive Bayes Classification Report: Training Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.49      0.97      0.65       682
           1       0.73      0.08      0.14       753

    accuracy                           0.50      1435



Naive Bayes Classification Report: Testing Data
---------------------------------------------------
              precision    recall  f1-score   support

           0       0.46      0.95      0.62       220
           1       0.54      0.05      0.09       259

    accuracy                           0.46       479


### Comparing Accuracy Scores

![image-16](https://github.com/gayajohn/project-4-group-5/assets/135036996/362cb01f-297a-4a92-ba9e-d549839d9c4c)

## Attempting to Optimize Model

### Hypertuning the XGBoost, the Winning Model

The XGboost model has achieved the highest accuracy among all the models. Now we will try to increase the performance even more. We will use a cross-validation approach and use GridSearchCV to find the best hyperparameters.

**Optimal Model: Best XGBoost Accuracy (Binary Classification): 0.5615866388308977 with hyperparameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}**

### Feature Selection

In our task we will identify which features were the most valuable for our model. In our first step we will check if by any chance we can increase the accuracy of our model extracting a feature.

![image-18](https://github.com/gayajohn/project-4-group-5/assets/135036996/678e52f3-cb9f-4e33-8ec9-48638f252c5e)

#### Using Feature Importance to Reduce Chart Dimenstionality

XGB Accuracy (Binary Classification) using top 5 features: 0.5010438413361169

XGB Accuracy (Binary Classification) using top 10 features: 0.5615866388308977

XGB Accuracy (Binary Classification) using top 15 features: 0.5365344467640919


### Reducing Dimensionality with PCA

- 3 PCA's
- Explained variance:  [0.91025545 0.05117109 0.02927404]
- Sum of explained variance:  0.990700583075814
- XGB Accuracy (Binary Classification) using PCA: 0.4906054279749478

### Removing Outliers

- Original DataFrame shape: (1914, 23)
- DataFrame without outliers shape: (1654, 23)
- XGB Accuracy (Binary Classification) using No Outliers: 0.5483091787439613

## Charting Our Tuning Efforts

![image-19](https://github.com/gayajohn/project-4-group-5/assets/135036996/b76d8d23-78ea-48b3-ae85-5be8c0f2c4c7)

## Summarizing Results

- After training 12 models, XGBoost had highest accuracy.
- The best accuracy we could achieve was 56.15%, even after tuning.
- The most important features in our data turned out to be price returns and financials
- Original model, hypertuned model and model with top 10 features yielded same results.
- Precision rate of 46% for decrease, but 54% for increase or the same. Recall rate of 41% for decrease and 58% for increase or stay the same. 
- When reviewing many of our ML models, we observed that our training data had much higher accuracy than testing data

## Challenges & Limitations

- The stock market is challenging to predict with machine learning due to its inherent complexity, influenced by numerous dynamic factors, including economic indicators, geopolitical events, and human behavior, leading to non-linear and unpredictable patterns in market movements. In general, it would be difficult to achieve an accuracy of over 60%

- We are limited to use free data. Thus, we had a small dataset with limits on data quality and features. We did not have resources to capture all relevant features in our model. We only had 1914 rows and 23 features.

- We had a serious overfitting issue. This should be solved using regularization. 

## Scope for Improvement

- Paying for higher quality data and more features. 
- Try to capture market sentiment using Natural Lanaguage Processing Models, including news and social media. This is a resource-heavy task.
- Trying to find patterns in a niche may yield better results. A 2019 study on short term stock market price prediction was able to achieve high accuracy (93%) by using the chinese stock market as their niche. 

## Real Life Applications for the Model

- **High Frequency Trading**: Machine Learning can help in identifying patterns, and providing predictive insights to inform trading decision, especially for more subtle changes.

- **Portfolio Optimization**: Machine learning can aid in portfolio optimization by analyzing diverse financial data, identifying patterns, and recommending optimal asset allocations to maximize returns while managing risk.

- **Risk Management**: Machine learning can support risk management in trading by analyzing market data, predicting potential risks, and providing real-time insights to minimize financial losses.

As a conluding thought, we wanted to reiterate that these kinds of models should be used as complementary tools, in conjunction with domain knowledge, personal judgement and professional expertise. Using machine learning models to make stand alone trading decisions poses a great risk due to the uncertain nature of stock markets.

## References 

- https://www.nasdaq.com/market-activity/stocks/screener
- https://fred.stlouisfed.org/docs/api/fred/
- https://www.alphavantage.co/documentation/#intelligence
- https://pypi.org/project/yfinance/
- https://theautomatic.net/yahoo_fin-documentation/
- https://stockanalysis.com/stocks/aapl/financials/
- https://finnhub.io/docs/api/technical-indicator


## Libraries Used

- Pandas
- Numpy
- MatPlotLib
- Seaborn
- SciPy
- SciKit Learn
- XGBoost
- Tensor Flow
- Keras Tuner

## Necessary GitBash Commands

```bash
pip install yahoo_fin
```

```bash
pip install yfinance
```

```bash
pip install finnhub-python
```




