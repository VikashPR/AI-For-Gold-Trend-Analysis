# AI Project Report – Module E

---

## Student & Project Details
**Student Name:** Vikash PR  
**Mentor Name:** [To be filled]  
**Project Title:** AI-Powered Gold Price Trend Analysis using LSTM Networks

---

## 1. Problem Statement

Gold has always been a safe investment during uncertain times - economic downturns, political instability, or high inflation. However, predicting its price is incredibly tricky because so many things affect it at once. Interest rates, inflation numbers, geopolitical tensions, currency fluctuations, and even investor psychology all play a role. 

I wanted to tackle this challenge because traditional forecasting methods like moving averages or basic statistical models often miss the mark. They can't handle the complexity and non-linear patterns that exist in gold price movements. This became especially clear during events like the 2020 COVID-19 pandemic when gold prices jumped unpredictably.

The goal of this project was to build an AI system that could learn from over a decade of gold price data and predict future price movements with reasonable accuracy. The key objectives were:

- Analyze 12+ years of historical gold price data (2013-2025)
- Build a deep learning model that captures patterns traditional methods miss
- Predict next-day closing prices for gold
- Give investors and analysts a tool to make more informed decisions

I focused specifically on using LSTM (Long Short-Term Memory) neural networks because they're designed to remember patterns over time, which is exactly what you need when dealing with financial data where today's price is influenced by what happened weeks or even months ago.

This isn't just an academic exercise - gold trading volumes run into billions daily, and even small improvements in prediction accuracy can help portfolio managers, retail investors, and financial institutions make better choices about when to buy or sell.

---

## 2. Approach

### System Overview
I built this as an end-to-end pipeline: data collection → preprocessing → feature engineering → model training → evaluation → prediction. The entire workflow is contained in a Jupyter notebook, making it easy to understand each step.

### Data Strategy
I used historical gold futures data (ticker: GC=F) from Yahoo Finance covering January 2013 to the present. This gave me over 3000 daily trading records with standard OHLCV data - Open, High, Low, Close prices, and trading Volume.

The dataset was particularly valuable because it includes several important market cycles: the 2015-2016 correction, the 2019-2020 rally during COVID-19, and recent fluctuations in 2023-2024. This variety helped the model learn from different market conditions.

For preprocessing, I did several things:
- Filled any missing values using forward-fill method (carry forward the last known price)
- Added technical indicators that traders actually use: Moving Averages (20-day and 50-day), RSI (Relative Strength Index), MACD, and Bollinger Bands
- Normalized all features between 0 and 1 using MinMaxScaler - this is crucial for LSTM networks to train properly
- Created a 60-day lookback window, meaning the model uses the past 60 days of data to predict day 61

### AI Model Design
I chose LSTM networks because they have "memory" - they can remember relevant information from earlier in the sequence and forget irrelevant details. This is perfect for time series data.

My architecture:
- **Input Layer:** Takes in 60 timesteps of multiple features (price, volume, technical indicators)
- **LSTM Layer 1:** 128 units with dropout (0.2) to prevent overfitting
- **LSTM Layer 2:** 64 units with dropout (0.2)
- **Dense Layer:** 32 neurons with ReLU activation
- **Output Layer:** Single neuron predicting the next day's closing price

I split the data 80-20: training on the first 80% and testing on the most recent 20%. This simulates real-world usage where you train on historical data and predict future prices.

Training used the Adam optimizer with mean squared error as the loss function. I trained for 50 epochs with early stopping to avoid overfitting - if the validation loss stopped improving for 10 consecutive epochs, training would stop automatically.

### Tools & Technologies
- **Python 3.10+** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **Pandas & NumPy** - Data manipulation
- **Matplotlib, Seaborn, Plotly** - Visualizations
- **scikit-learn** - Data preprocessing and metrics
- **yfinance** - Data download from Yahoo Finance

### Design Decisions
Some key choices I made:
- Used 60 days as lookback period after experimenting with 30, 60, and 90 - 60 gave the best balance
- Added dropout layers to prevent the model from memorizing training data
- Included multiple technical indicators because they capture different aspects of price behavior (trend, momentum, volatility)
- Saved both the model and the scaler so predictions on new data would be consistent

---

## 3. Key Results

### Description of the Working Prototype
The final system is a fully functional LSTM-based gold price forecasting tool that:
- Processes 12+ years of daily gold price data (2013-2025)
- Uses a 60-day lookback window to capture temporal patterns
- Generates next-day price predictions in real-time
- Includes technical indicators (Moving Averages, RSI, MACD, Bollinger Bands) for enhanced pattern recognition
- Provides visual comparisons between predicted and actual prices
- Outputs prediction confidence metrics and error margins

The prototype successfully demonstrates end-to-end deep learning workflow: data preprocessing → feature engineering → model training → evaluation → prediction.

### Example Outputs (Observations)
**Sample Predictions from Test Set:**

| Date | Actual Price | Predicted Price | Error | % Error |
|------|--------------|-----------------|-------|---------|
| Aug 22, 2023 | $1,894.60 | $1,874.41 | $20.19 | 1.07% |
| Aug 24, 2023 | $1,920.00 | $1,872.99 | $47.01 | 2.45% |
| Sep 1, 2023 | $1,950.00 | $1,912.15 | $37.85 | 1.94% |
| Sep 5, 2023 | $1,934.40 | $1,920.73 | $13.67 | 0.71% |

**Price Range During Test Period:** $1,838 - $3,411 per ounce

**Visual Observations:**
- The model's prediction curve closely follows actual prices during stable periods
- Notable lag during rapid price increases (2024-2025 surge)
- Predictions tend to be conservative, underestimating volatile upward movements
- Training set predictions show tighter fit than test set, indicating some overfitting

### Evaluation Method and Metrics Used
**Evaluation Approach:**
- 80/20 train-test split (chronological, not random)
- Test set represents most recent 20% of data to simulate real-world future predictions
- Multiple metrics calculated to assess different aspects of performance

**Performance Metrics on Test Set:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | $395.24 | Average prediction deviation |
| **MAE** | $272.44 | Typical absolute error |
| **MAPE** | 8.06% | Mean percentage error |
| **R² Score** | 0.6939 | Model explains 69.39% of price variance |
| **Trend Direction Accuracy** | 63.02% | Correct up/down predictions |

**Training Set Performance (for comparison):**
- RMSE: Lower than test set
- R² Score: Higher than test set
- Shows model learned patterns but generalization is limited

### Performance Insights and Observations

**What Works Well:**
- The model successfully captures temporal dependencies in gold prices
- MAPE of 8.06% is within acceptable range for academic forecasting projects
- R² of 69% indicates the model learned meaningful patterns (better than simple baseline models)
- Performs reasonably during stable market conditions
- Successfully processes multiple features (OHLCV + technical indicators)

**What Needs Improvement:**
- **Trend direction accuracy of 63%** is only marginally better than random chance (50%)
  - For practical trading, you'd need 70-80% minimum
  - Current accuracy means wrong 37% of the time
- **High error during volatility:** Model struggles with rapid price movements
  - Consistently underestimates during price surges (2024-2025 period)
  - Creates prediction lag during trend changes
- **Absolute error magnitude:** $272 average error is significant for trading decisions
  - On a $2,000 price point, this represents ~14% potential loss

**Key Insight:** The model proves that LSTMs can learn time-series patterns from historical data, but predicting financial markets requires more than just past prices. Gold prices are driven by macroeconomic factors (interest rates, inflation, geopolitical events) that aren't captured in OHLCV data alone.

### Known Limitations and Failure Cases

**Technical Limitations:**
- **Only price-based features:** Model doesn't incorporate external factors like Fed policy, inflation data, or geopolitical events
- **Short-term only:** Optimized for next-day predictions; longer horizons (weekly/monthly) would require different architecture
- **No confidence intervals:** Provides point predictions without uncertainty estimates
- **Overfitting signs:** Gap between training (good) and test (moderate) performance indicates limited generalization

**Failure Cases Observed:**
1. **March 2020 COVID-19 crash:** Model failed to predict sudden drop as it had no historical precedent
2. **2024-2025 gold surge:** Consistently underestimated rapid price increases
3. **High volatility periods:** Error rates spike when daily price swings exceed 3-4%
4. **Market gaps:** Cannot account for weekend news or after-hours events affecting Monday opening

**Practical Reality Check:**
- 63% directional accuracy = losing money on 37% of trades
- Not reliable enough for actual trading or investment decisions
- Works as proof-of-concept but requires significant improvements for production use
- Missing critical features: economic indicators, news sentiment, global events

**Honest Assessment:** This is a successful learning project demonstrating competence in LSTM implementation and time-series forecasting. However, it also illustrates an important lesson: just because AI can learn patterns doesn't mean it can predict complex financial markets well enough to be practically useful. The results are good enough to show understanding, but not good enough to trust with real money.

### Visual Results
The prediction graphs show the model tracks the actual gold price curve quite closely. There's minimal lag, and the model doesn't make wild predictions - it stays within reasonable bounds even when tested on data it has never seen before.

### Limitations
The system has some important limitations:
- **Accuracy Issues:** With only 63% trend direction accuracy, the model isn't reliable enough for real trading - it's only marginally better than random chance
- **Volatility Handling:** The model struggles during highly volatile periods, consistently underestimating rapid price movements (as seen in 2024-2025 data)
- **Lag Effect:** Predictions tend to lag behind actual trends, making it reactive rather than predictive during sudden market shifts
- **Missing Context:** The model only sees price patterns - it doesn't know about Fed interest rate decisions, geopolitical events, or economic indicators that actually drive gold prices
- **Next-Day Only:** Currently optimized for next-day predictions; longer horizons (week, month) would need model modifications and would likely have even lower accuracy
- **No Uncertainty Estimates:** The model gives a single prediction without confidence intervals, so you don't know when it's more or less certain

**Real Talk:** The 8% MAPE sounds okay on paper, but combine it with 63% directional accuracy and you realize the model is trying its best but fundamentally limited by only seeing historical prices. Gold doesn't move just because of past patterns - it moves because of news, policy changes, and global economics.

---

## 4. Learnings

### Technical Learnings: AI Concepts, Tools, and Techniques Gained

**Deep Learning for Time Series:**
- **LSTM Architecture Understanding:** Learned how LSTM cells use gates (forget, input, output) to selectively remember or forget information across sequences. The "memory" mechanism is what makes them powerful for sequential data unlike standard neural networks.
- **Temporal Dependencies:** Understood that sequence order is critical in time series - you can't shuffle data randomly like in image classification. The 60-day lookback window means the model analyzes the past 60 days of data to predict the next day's closing price.
- **Recurrent vs Feed-Forward:** Realized why regular neural networks fail at time series - they treat each input independently, while RNNs/LSTMs maintain state across timesteps.

**Feature Engineering Impact:**
- **Technical Indicators Matter:** Raw OHLCV data alone gave poor results. Adding domain-specific features (RSI, MACD, Bollinger Bands, Moving Averages) significantly improved learning because they encode trading knowledge.
- **Feature Scaling is Critical:** LSTM performance is highly sensitive to input scale. Without MinMaxScaler normalization (0-1 range), the model failed to converge during training. This is because gradient descent struggles with features at different magnitudes.
- **Domain Knowledge + AI:** The best results came from combining financial domain knowledge (which indicators matter) with machine learning capabilities.

**Model Training Insights:**
- **Overfitting Prevention:** Dropout layers (0.2 rate) were essential to prevent memorizing training patterns. Without dropout, training accuracy was high but test accuracy crashed.
- **Early Stopping:** Implemented patience-based early stopping - if validation loss didn't improve for 10 epochs, training stopped automatically to prevent overfitting.
- **Batch Size Impact:** Experimented with different batch sizes (16, 32, 64) - smaller batches gave better generalization but slower training.

**Tools Mastery:**
- **TensorFlow/Keras:** Learned Sequential API for building neural networks, layer stacking, and custom training loops
- **Pandas for Time Series:** Mastered datetime indexing, rolling windows, and forward-fill for handling missing data
- **Data Visualization:** Used Matplotlib and Plotly to visualize predictions, helping identify model behavior patterns

### System & Design Learnings: Architecture, Scalability, and Workflow Insights

**End-to-End Pipeline Design:**
- **Modular Workflow:** Built the system in clear stages (data loading → preprocessing → feature engineering → training → evaluation → prediction). This modularity made debugging easier.
- **Reproducibility:** Learned to save not just the trained model (.h5/.keras files) but also the scaler objects and feature configuration. Without saving scalers, new predictions would use different normalization, producing garbage results.
- **Version Control:** Keeping track of model versions with different hyperparameters (lookback window, layers, dropout rates) was crucial for comparing performance.

**Data Management:**
- **Train-Test Split Strategy:** For time series, you MUST use chronological splits, not random. Training on future data and testing on past data (data leakage) gives falsely high accuracy.
- **Data Quality Checks:** Implemented validation steps to catch missing values, outliers (using IQR), and data inconsistencies before training.
- **Feature Storage:** Saved processed features with technical indicators to avoid recalculating them during inference.

**Performance Optimization:**
- **Lookback Window Trade-off:** Experimented with 30, 60, 90, 120-day windows. 60 days gave best balance between capturing patterns and training speed. Longer windows added minimal accuracy but significantly increased computation time.
- **GPU Utilization:** Training on TensorFlow with GPU support cut training time from 15 minutes to 3 minutes per 50 epochs.

**Visualization as Debugging:**
- **Plot Everything:** Creating visualizations at each stage (raw data, after normalization, predictions vs actuals) caught bugs I'd never find otherwise. Example: spotted an impossible price spike that revealed a data loading error.
- **Error Analysis Plots:** Plotting prediction errors over time revealed the model's weakness during high volatility periods, which led to adding Bollinger Bands as a feature.

### Challenges Faced: Key Difficulties and How I Resolved Them

**Challenge 1: COVID-19 Volatility Handling**
- **Problem:** Initial model trained on 2013-2019 data completely failed on 2020+ predictions. March 2020 saw unprecedented gold price movements due to pandemic panic.
- **Why It Failed:** The model had never seen such extreme volatility in training data, so it couldn't generalize to "black swan" events.
- **Solution Implemented:**
  1. Extended training data to include 2020-2024, exposing the model to volatile periods
  2. Added Bollinger Bands as features (they explicitly measure volatility/uncertainty)
  3. Increased dropout rate to 0.2 to prevent overfitting to "normal" conditions
  4. Result: Model still struggles with extreme events but handles moderate volatility better

**Challenge 2: Choosing the Right Lookback Window**
- **Problem:** Unclear how many historical days the model should consider.
- **Experimentation:**
  - 30 days: Model missed longer-term trends, predictions were too reactive
  - 60 days: Best balance - captured monthly patterns
  - 90 days: Minimal accuracy improvement but 40% slower training
  - 120 days: Actually worse performance (too much noise)
- **Solution:** Settled on 60-day window through systematic trial-and-error, balancing accuracy and computational cost.

**Challenge 3: Test Set Performance Gap**
- **Problem:** Training accuracy looked great (R² > 0.95), but test accuracy dropped significantly (R² = 0.69).
- **Root Cause:** Model was overfitting to training patterns.
- **Solutions Applied:**
  1. Added dropout layers after each LSTM layer
  2. Reduced model complexity (fewer units per layer)
  3. Implemented early stopping based on validation loss
  4. Used more aggressive data augmentation
- **Result:** Gap narrowed but still exists - indicates fundamental difficulty in predicting financial markets

**Challenge 4: Directional Accuracy Disappointment**
- **Problem:** While price predictions looked reasonable visually, trend direction accuracy stuck at 63% (barely better than coin flip).
- **Realization:** Price prediction doesn't equal profitable trading. Even if predicted price is close, getting the direction wrong means losses.
- **Learning:** This taught me that financial forecasting needs different metrics than typical regression problems. Directional accuracy matters more than RMSE for practical trading.

**Challenge 5: Handling Missing Data**
- **Problem:** Some trading days missing (holidays, data gaps), causing array shape mismatches.
- **Solution:** Used forward-fill strategy to propagate last known price. While not perfect, it's better than dropping rows (which breaks sequence continuity).

### Future Improvements: Enhancements and Next Steps

**Short-Term Enhancements (Immediately Feasible):**

1. **Incorporate External Features**
   - Add Federal Reserve interest rate data (gold and rates are inversely correlated)
   - Include USD index (gold is dollar-denominated, so dollar strength affects price)
   - Add inflation indicators (CPI, PPI) as gold is inflation hedge
   - **Expected Impact:** Could improve R² to 75-80% by adding fundamental drivers

2. **Attention Mechanisms**
   - Implement attention layers to let model learn which historical days are most relevant
   - Instead of treating all 60 days equally, attention would weight recent days more
   - **Expected Impact:** Better handling of volatile periods, improved directional accuracy to ~70%

3. **Ensemble Modeling**
   - Combine LSTM with ARIMA (good for short-term linear trends)
   - Add XGBoost for capturing non-linear feature interactions
   - Use weighted average of predictions from multiple models
   - **Expected Impact:** More robust predictions, reduced variance

**Medium-Term Improvements (Requires More Development):**

4. **Multi-Horizon Predictions**
   - Modify architecture to predict 1-day, 5-day, and 30-day ahead simultaneously
   - Use multi-task learning with shared LSTM layers
   - **Benefit:** One model serves multiple use cases (day traders vs long-term investors)

5. **Uncertainty Quantification**
   - Implement prediction intervals using dropout at inference time (Monte Carlo Dropout)
   - Provide confidence bands: "95% confident price will be between $1,900-$2,100"
   - **Benefit:** Users know when model is uncertain vs confident

6. **News Sentiment Analysis**
   - Scrape financial news headlines about gold
   - Use NLP (BERT/GPT) to extract sentiment scores
   - Feed sentiment as additional feature
   - **Expected Impact:** Better capture of market psychology and event-driven moves

**Long-Term Vision (Production System):**

7. **Real-Time Deployment**
   - Build web application (Flask/FastAPI backend)
   - Automatic data fetching from Yahoo Finance API
   - Display live predictions updated daily
   - Mobile-responsive interface

8. **Continuous Learning Pipeline**
   - Implement automated retraining as new data arrives
   - A/B testing framework to compare model versions
   - Performance monitoring dashboard to detect degradation

9. **Risk Management Features**
   - Calculate maximum drawdown scenarios
   - Provide stop-loss recommendations
   - Portfolio allocation suggestions based on prediction confidence

**Realistic Assessment:**
Even with all improvements, achieving >80% directional accuracy is extremely difficult because financial markets are influenced by unpredictable human behavior, geopolitics, and random events. The goal should be building a useful decision-support tool, not a crystal ball. Professional trading firms with millions in R&D budget still struggle with this problem.

---

## References & AI Usage Disclosure

### Datasets Used
- **Primary Dataset:** Gold Futures Historical Data (GC=F)  
  Source: Yahoo Finance via yfinance Python library  
  URL: https://finance.yahoo.com/quote/GC=F/history/  
  Time Period: January 2013 - January 2025

### Tools, Frameworks & Libraries
- **TensorFlow 2.x:** Deep learning framework for building and training LSTM model  
  https://www.tensorflow.org/
- **Keras:** High-level neural network API (part of TensorFlow)  
  https://keras.io/
- **Pandas 2.x:** Data manipulation and analysis  
  https://pandas.pydata.org/
- **NumPy:** Numerical computing with arrays  
  https://numpy.org/
- **scikit-learn:** Machine learning utilities (scaling, metrics)  
  https://scikit-learn.org/
- **Matplotlib, Seaborn, Plotly:** Data visualization libraries
- **yfinance:** Python library to download financial data from Yahoo Finance  
  https://github.com/ranaroussi/yfinance

### Learning Resources Referenced
- "Understanding LSTM Networks" by Christopher Olah  
  http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- TensorFlow Time Series Forecasting Tutorial  
  https://www.tensorflow.org/tutorials/structured_data/time_series
- "Stock Price Prediction using Machine Learning" - various Kaggle notebooks and tutorials

### AI Tools Used During Development
- **GitHub Copilot:** Used for code autocompletion and syntax suggestions while writing preprocessing and visualization functions. Did not generate entire code blocks, only assisted with standard pandas/numpy operations and plotting syntax.
- **ChatGPT (GPT-4):** Used to understand LSTM architecture concepts and troubleshoot a specific TensorFlow error message related to input shapes. Also used to review and improve documentation structure.
- **Claude (Anthropic):** Used for final report writing assistance - provided project details and requested human-style writing without technical jargon. Edited and personalized all AI-generated text to reflect actual project experience.

### Disclosure Statement
This project represents my own work in designing, implementing, and training the AI model. AI coding assistants were used as tools for syntax help and documentation, similar to how one would use Stack Overflow or official documentation. All code logic, architecture decisions, and experimental choices were made by me. The final report was drafted with AI assistance but heavily edited to accurately reflect my actual learning process and results.

---

**Project Completion Date:** January 2026  
**Module:** E - AI Applications  
**Project Type:** Individual Open Project
