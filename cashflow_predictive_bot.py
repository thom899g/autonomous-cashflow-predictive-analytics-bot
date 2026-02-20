from typing import Dict, List, Optional
import logging
import pandas as pd
from datetime import datetime, timedelta
from .api_wrapper import StripeAPIWrapper, PlaidAPIWrapper
from .models.predictive_model import PredictiveModel
from ..knowledge_baseUpdater import KnowledgeBase

class CashflowPredictiveBot:
    def __init__(self):
        self.api_wrapper: Optional[StripeAPIWrapper | PlaidAPIWrapper] = None
        self.model: PredictiveModel = PredictiveModel()
        self.knowledge_base = KnowledgeBase()
        self.error_log = []
        
        # Initialize logging
        logging.basicConfig(
            filename='cashflow_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def connect_to_api(self, api_type: str, **kwargs):
        """
        Initializes connection to either Stripe or Plaid API.
        
        Args:
            api_type (str): 'stripe' or 'plaid'
            kwargs: API credentials and configuration
        """
        if api_type == 'stripe':
            self.api_wrapper = StripeAPIWrapper(**kwargs)
        elif api_type == 'plaid':
            self.api_wrapper = PlaidAPIWrapper(**kwargs)
        else:
            raise ValueError("Invalid API type specified")

    def fetch_financial_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches transaction data from connected API.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame containing transactions and other financial data
        """
        try:
            data = self.api_wrapper.get_transactions(start_date, end_date)
            return pd.DataFrame(data)
        except Exception as e:
            logging.error(f"Failed to fetch financial data: {str(e)}")
            self.error_log.append((datetime.now(), str(e)))
            raise

    def train_model(self, data: pd.DataFrame):
        """
        Trains the predictive model using historical cashflow data.
        
        Args:
            data (DataFrame): Historical financial data
        """
        try:
            # Preprocess data and train model
            self.model.train(data)
            logging.info("Model training completed successfully")
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def predict_cashflow(self, forecast_days: int) -> Dict[str, float]:
        """
        Predicts future cashflows based on trained model.
        
        Args:
            forecast_days (int): Number of days to forecast
            
        Returns:
            Dictionary containing predicted cashflow metrics
        """
        try:
            prediction = self.model.predict(forecast_days)
            return {
                'predicted_cashflow': prediction['cashflow'],
                'confidence_interval_low': prediction['low_ci'],
                'confidence_interval_high': prediction['high_ci']
            }
        except Exception as e:
            logging.error(f"Cashflow prediction failed: {str(e)}")
            self.error_log.append((datetime.now(), str(e)))
            raise

    def handle_error(self, error: Exception):
        """
        Handles exceptions and retries operations where possible.
        
        Args:
            error (Exception): The exception that occurred
        """
        # Implement retry logic for transient errors
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                if isinstance(error, ConnectionError):
                    # Retry API call after short delay
                    import time
                    time.sleep(10)
                    self.api_wrapper.reinitialize_connection()
                elif isinstance(error, DataProcessingError):
                    # Reprocess data with updated parameters
                    self.train_model(self.fetch_financial_data(
                        (datetime.now() - timedelta(days=365)).isoformat(),
                        datetime.now().isoformat()
                    ))
                break
            except Exception as e:
                retries += 1
                logging.error(f"Retrying failed operation. Attempt {retries}/{max_retries}")
                if retries == max_retries:
                    raise

    def generate_insights(self) -> Dict[str, str]:
        """
        Generates business operation insights from predictions.
        
        Returns:
            Dictionary containing actionable insights
        """
        try:
            insights = self.model.generate_insights()
            return {
                'key_metric_1': insights['metric1'],
                'key_metric_2': insights['metric2'],
                # Add more as needed
            }
        except Exception as e:
            logging.error(f"Insight generation failed: {str(e)}")
            raise

    def update_knowledge_base(self):
        """
        Updates the knowledge base with new predictions and data.
        """
        try:
            latest_predictions = self.predict_cashflow(30)
            historical_data = self.fetch_financial_data(
                (datetime.now() - timedelta(days=365)).isoformat(),
                datetime.now().isoformat()
            )
            
            # Store both predictions and historical data
            self.knowledge_base.update({
                'cashflow_predictions': latest_predictions,
                'historical_cashflow_data': historical_data.to_dict()
            })
        except Exception as e:
            logging.error(f"Failed to update knowledge base: {str(e)}")
            raise

    def run_analysis(self):
        """
        Main execution method for cashflow predictive analysis.
        """
        try:
            # Fetch data
            data = self.fetch_financial_data(
                (datetime.now() - timedelta(days=365)).isoformat(),
                datetime.now().isoformat()
            )
            
            # Train model
            self.train_model(data)
            
            # Generate predictions
            thirty_day_forecast = self.predict_cashflow(30)
            ninety_day_forecast = self.predict_cashflow(90)
            
            # Generate insights
            insights = self.generate_insights()
            
            # Update knowledge base
            self.update_knowledge_base()
            
            logging.info("Cashflow predictive analysis completed successfully")
            return {
                'forecasts': {
                    '30_days': thirty_day_forecast,
                    '90_days': ninety_day_forecast
                },
                'insights': insights
            }
            
        except Exception as e:
            logging.error(f"Main analysis run failed: {str(e)}")
            self.error_log.append((datetime.now(), str(e)))
            raise

# Example usage:
if __name__ == "__main__":
    bot = CashflowPredictiveBot()
    bot.connect_to_api('stripe', API_KEY='your_stripe_key')
    result = bot.run_analysis()
    print(result)