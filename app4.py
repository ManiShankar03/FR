import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Load data
income_data = pd.read_csv('income.csv')
balance_data = pd.read_csv('balance.csv')

# Merge datasets on common columns
merged_data = pd.merge(income_data, balance_data, on=['symbol', 'fiscalDateEnding'])

# Company sector mapping and ticker to name mapping
company_sectors = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'META': 'Technology',
    'VZ': 'Communication Services',
    'AMZN': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary',
    'JPM': 'Financials',
    'V': 'Financials',
    'JNJ': 'Healthcare',
    'PFE': 'Healthcare',
    'WMT': 'Consumer Staples',
    'PG': 'Consumer Staples',
    'GE': 'Industrials',
    'XOM': 'Energy'
}

company_names = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms Inc.',
    'VZ': 'Verizon Communications Inc.',
    'AMZN': 'Amazon.com Inc.',
    'HD': 'The Home Depot Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble Co.',
    'GE': 'General Electric Company',
    'XOM': 'Exxon Mobil Corporation'
}

# Reverse lookup from name to ticker
name_to_symbol = {v: k for k, v in company_names.items()}

# Function to preprocess data (without scaling)
def preprocess_data(company_data, look_back=4):
    company_data = company_data.sort_values('fiscalDateEnding')
    
    features = {
        'gross_profit': company_data['grossProfit'].values.reshape(-1, 1),
        'total_revenue': company_data['totalRevenue'].values.reshape(-1, 1),
        'operating_income': company_data['operatingIncome'].values.reshape(-1, 1),
        'net_income': company_data['netIncome'].values.reshape(-1, 1),
        'total_assets': company_data['totalAssets'].values.reshape(-1, 1),
        'total_equity': company_data['totalShareholderEquity'].values.reshape(-1, 1),
        'current_assets': company_data['totalCurrentAssets'].values.reshape(-1, 1),
        'current_liabilities': company_data['totalCurrentLiabilities'].values.reshape(-1, 1),
        'total_liabilities': company_data['totalLiabilities'].values.reshape(-1, 1)
    }
    
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    datasets = {key: create_dataset(values, look_back) for key, values in features.items()}
    
    X_datasets = {key: np.reshape(X, (X.shape[0], X.shape[1], 1)) for key, (X, Y) in datasets.items()}
    Y_datasets = {key: Y for key, (X, Y) in datasets.items()}
    
    return X_datasets, Y_datasets

# Function to forecast next two quarters
def forecast_next_two_quarters(model, last_data, look_back=4):
    forecast_input = last_data.reshape(1, look_back, 1)
    forecasts = []
    for _ in range(2):
        next_forecast = model.predict(forecast_input)
        forecasts.append(next_forecast[0, 0])
        next_forecast_scaled = next_forecast
        forecast_input = np.concatenate((forecast_input[:, 1:, :], next_forecast_scaled.reshape(1, 1, 1)), axis=1)
    return np.array(forecasts)

# Function to calculate financial ratios
def calculate_financial_ratios(forecasts):
    gross_margin = forecasts['gross_profit'] / forecasts['total_revenue']
    operating_margin = forecasts['operating_income'] / forecasts['total_revenue']
    net_income_margin = forecasts['net_income'] / forecasts['total_revenue']
    return_on_assets = forecasts['net_income'] / forecasts['total_assets']
    return_on_equity = forecasts['net_income'] / forecasts['total_equity']
    current_ratio = forecasts['current_assets'] / forecasts['current_liabilities']
    debt_to_equity_ratio = forecasts['total_liabilities'] / forecasts['total_equity']
    asset_turnover_ratio = forecasts['total_revenue'] / forecasts['total_assets']
    
    return {
        'gross_margin': gross_margin,
        'operating_margin': operating_margin,
        'net_income_margin': net_income_margin,
        'return_on_assets': return_on_assets,
        'return_on_equity': return_on_equity,
        'current_ratio': current_ratio,
        'debt_to_equity_ratio': debt_to_equity_ratio,
        'asset_turnover_ratio': asset_turnover_ratio
    }

# Sector-specific thresholds
sector_thresholds = {
    'Technology': {
        'gross_margin': 0.4,
        'operating_margin': 0.15,
        'net_income_margin': 0.1,
        'return_on_assets': 0.05,
        'return_on_equity': 0.2,
        'current_ratio': (1.5, 3),
        'debt_to_equity_ratio': 0.5,
        'asset_turnover_ratio': 0.6
    },
    'Consumer Discretionary': {
        'gross_margin': 0.3,
        'operating_margin': 0.2,
        'net_income_margin': 0.15,
        'return_on_assets': 0.1,
        'return_on_equity': 0.15,
        'current_ratio': (1.5, 3),
        'debt_to_equity_ratio': 0.7,
        'asset_turnover_ratio': 0.5
    },
    'Financials': {
        'gross_margin': 0.2,
        'operating_margin': 0.25,
        'net_income_margin': 0.2,
        'return_on_assets': 0.01,
        'return_on_equity': 0.1,
        'current_ratio': (1, 2.5),
        'debt_to_equity_ratio': 0.8,
        'asset_turnover_ratio': 0.15
    },
    'Healthcare': {
        'gross_margin': 0.5,
        'operating_margin': 0.2,
        'net_income_margin': 0.1,
        'return_on_assets': 0.1,
        'return_on_equity': 0.15,
        'current_ratio': (1.5, 3),
        'debt_to_equity_ratio': 0.5,
        'asset_turnover_ratio': 0.6
    },
    'Consumer Staples': {
        'gross_margin': 0.35,
        'operating_margin': 0.15,
        'net_income_margin': 0.1,
        'return_on_assets': 0.05,
        'return_on_equity': 0.15,
        'current_ratio': (1.5, 3),
        'debt_to_equity_ratio': 0.6,
        'asset_turnover_ratio': 0.5
    },
    'Industrials': {
        'gross_margin': 0.3,
        'operating_margin': 0.1,
        'net_income_margin': 0.07,
        'return_on_assets': 0.05,
        'return_on_equity': 0.15,
        'current_ratio': (1.5, 3),
        'debt_to_equity_ratio': 0.7,
        'asset_turnover_ratio': 0.8
    },
    'Energy': {
        'gross_margin': 0.2,
        'operating_margin': 0.15,
        'net_income_margin': 0.08,
        'return_on_assets': 0.04,
        'return_on_equity': 0.1,
        'current_ratio': (1.2, 2.5),
        'debt_to_equity_ratio': 0.5,
        'asset_turnover_ratio': 0.5
    },
    'Communication Services': {
        'gross_margin': 0.4,
        'operating_margin': 0.2,
        'net_income_margin': 0.1,
        'return_on_assets': 0.05,
        'return_on_equity': 0.1,
        'current_ratio': (1.5, 3),
        'debt_to_equity_ratio': 0.6,
        'asset_turnover_ratio': 0.5
    }
}

# Function to get investment recommendation based on financial ratios
def get_investment_recommendation(ratios, company_symbol):
    sector = company_sectors.get(company_symbol, 'Technology')
    thresholds = sector_thresholds.get(sector, sector_thresholds['Technology'])
    
    recommendations = {}
    for key, values in ratios.items():
        recommendations[key] = []
        for value in values:
            if key in ['gross_margin', 'operating_margin', 'net_income_margin', 'return_on_assets', 'return_on_equity']:
                if value > thresholds[key]:
                    recommendations[key].append('Good')
                else:
                    recommendations[key].append('Bad')
            elif key == 'current_ratio':
                if thresholds[key][0] < value < thresholds[key][1]:
                    recommendations[key].append('Good')
                else:
                    recommendations[key].append('Bad')
            elif key == 'debt_to_equity_ratio':
                if value < thresholds[key]:
                    recommendations[key].append('Good')
                else:
                    recommendations[key].append('Bad')
            elif key == 'asset_turnover_ratio':
                if value > thresholds[key]:
                    recommendations[key].append('Good')
                else:
                    recommendations[key].append('Bad')
    return recommendations

# Function to get overall recommendation with weights
def get_overall_recommendation(recommendations):
    weights = {
        'gross_margin': 1.0,
        'operating_margin': 1.0,
        'net_income_margin': 1.0,
        'return_on_assets': 1.0,
        'return_on_equity': 1.0,
        'current_ratio': 1,
        'debt_to_equity_ratio': 1,
        'asset_turnover_ratio': 1
    }
    
    weighted_score = 0
    total_weight = sum(weights.values())
    
    for key, values in recommendations.items():
        weight = weights.get(key, 1.0)
        for value in values:
            if value == 'Good':
                weighted_score += weight
    
    score_ratio = weighted_score / (total_weight * len(recommendations.values()))
    return 'Invest' if score_ratio > 0.1 else 'Do not Invest'

# Streamlit app
st.title("Financial Ratios and Investment Recommendation")

selected_companies = st.multiselect("Select companies:", list(company_names.values()))

if selected_companies:
    for company in selected_companies:
        company_symbol = name_to_symbol[company]
        company_data = merged_data[merged_data['symbol'] == company_symbol]
        
        st.write(f"Loading data and models for {company}...")
        X_datasets, Y_datasets = preprocess_data(company_data, look_back=4)

        forecasts = {}
        for key in ['gross_profit', 'total_revenue', 'operating_income', 'net_income', 'total_assets', 'total_equity', 'current_assets', 'current_liabilities', 'total_liabilities']:
            model_path = os.path.join('models1', f'{company_symbol}_{key}_lstm_model.joblib')
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                
                last_data = X_datasets[key][-1]
                forecasts[key] = forecast_next_two_quarters(model, last_data)
            else:
                st.warning(f"Model not found for {key} of {company}.")
                forecasts[key] = np.array([np.nan, np.nan])
        
        st.write(f"**Forecast for {company}:**")
        forecast_df = pd.DataFrame(forecasts, index=['Quarter 1', 'Quarter 2'])
        st.dataframe(forecast_df)

        ratios = calculate_financial_ratios(forecasts)
        st.write(f"**Financial Ratios for {company}:**")
        ratios_df = pd.DataFrame(ratios)
        st.dataframe(ratios_df)
        
        # Display Recommendations and Overall Recommendation
        recommendations = get_investment_recommendation(ratios, company_symbol)
        st.write(f"**Recommendations for {company}:**")
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df)
        
        overall_recommendation = get_overall_recommendation(recommendations)
        st.write(f"**Overall Recommendation for {company}:** {overall_recommendation}")
