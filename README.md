# Stock Price Prediction with Machine Learning

## Objective
The objective of this project is to utilize monthly stock data obtained from Yahoo Finance, including company financials, price and return, analyst ratios, etc. Our goal is to predict whether the stock price will increase, decrease, or remain the same in the next month. We aim to achieve this by employing various machine learning models and selecting the one that yields the highest accuracy. Additionally, we will perform hyperparameter tuning to further optimize the chosen model's performance.

![image](https://github.com/gayajohn/project-4-group-5/assets/135666038/1ee1d17b-b082-4bea-be85-d193cc69cba3)

## Dataset
We will procure publicly available stock information directly from Yahoo Finance using the YFinance library and finnhub.io in Python. This dataset will include a range of features such as historical stock prices, financial indicators, and analyst ratios.

![image](https://github.com/gayajohn/project-4-group-5/assets/135666038/338e4033-5d0c-452c-82dc-5a58770e9bd6)
![image](https://github.com/gayajohn/project-4-group-5/assets/135666038/0bd00e12-a6a5-4494-90a9-4b99b69364bf)

## Methodology
- Data Collection: Use the YFinance library to fetch monthly stock data from Yahoo Finance for the selected companies.

- Data Cleaning and Pre-processing: Clean the obtained data, handle missing values, and preprocess features for machine learning input.

- Feature Engineering: Extract relevant features and create new ones that might enhance predictive power.

- Model Selection: Implement multiple machine learning models such as Decision Trees, Random Forest, Support Vector Machines, and Neural Networks.

- Model Evaluation: Assess the performance of each model using appropriate evaluation metrics, such as accuracy, precision, recall, and F1 score.

- Model Comparison: Identify the model with the highest accuracy as the baseline model for further optimization.

- Hyperparameter Tuning: Fine-tune the hyperparameters of the selected model to improve its predictive capabilities.

- Final Model Selection: Choose the model with the best overall performance after hyperparameter tuning.

## Data Procurement:

**1. Ticker List Generation**

- Created a list of tickers from the NASDAQ screener, including only American stocks categorized as mega, large, and mid-cap.

**2. Random Date Generation**

- Implemented a function to generate a random date within a given time period.

**3. Random Stock Selection**

- Developed a function to randomly choose a stock from the generated ticker list.

**4. Stock Data Availability Check**

- Implemented a function to check if data is available for a given stock using the Yahoo Finance API. Further code enhancements are in progress.

**5. Stock Industry Retrieval**

- Created a function to return the industry of a stock using information from the NASDAQ CSV.

**6. Stock Price Retrieval**

- Implemented a function to get the stock price for a given date using the Yahoo Finance API.

**7. Weekly Returns Loop**

- Established a loop to calculate and retrieve weekly returns for selected stocks.

**8. Stock Volume Retrieval**

- Developed a function to obtain stock volume for the last 4 weekdays using the Yahoo Finance API. This facilitates the retrieval of trading volume for the last 5 business days.

**9. Stock Financials Retrieval**

- Implemented a function to scrape financial data from stockanalysis.com, focusing on annual and quarterly financial statements and financial ratios. The function selects the closest previous quarterly report.

**10. Stock Volatility Calculation**

- Created a function to calculate stock volatility using historical price data up to the generated date. Utilizes the Yahoo Finance API for data retrieval.

## Data Exploration 


## Repository Structure
- data_procurement.ipynb: Python script to fetch and collect data from Yahoo Finance.
- feature_engineering.py: Perform feature engineering to enhance the dataset for machine learning.
- model_training.py: Implement and train multiple machine learning models.
- model_evaluation.py: Evaluate the performance of the trained models using various metrics.
- hyperparameter_tuning.py: Fine-tune the hyperparameters of the selected model.
- final_model.py: Implement the final selected model for stock price prediction.

## Dependencies

- Python 3.x
- yfinance
- pandas
- numpy
- scikit-learn
- tensorflow
- keras
- matplotlib
- seaborn

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

## How to Run
- Clone the repository.
- Install the required dependencies using ``` pip install -r requirements.txt.``` 
- Run the scripts in the specified order as outlined in the methodology.

## Results


## Conclusion
