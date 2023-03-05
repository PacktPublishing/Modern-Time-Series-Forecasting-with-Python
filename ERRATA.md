Thanks for picking up my book on timeseries forecasting and I hope you’re enjoying it!

Unfortunately, a few typos have managed to sneak through our proofreading process. We apologize for any confusion these may have caused.

We’d like to make sure everyone has the best experience possible, so please raise an issue in the repository if you spot any additional typos. We'll keep collating the known typos in this document.

Thanks for helping us make our book even better!

1. Chapter: 3 - Analyzing and Visualizing Time Series Data - Section: _Interquartile Range (IQR)_   

    **Error**:     
    `IQR = Q3-Q2`    
    **Actual**:    
    `IQR = Q3-Q1`    

2. Chapter: 4 - Setting a Strong Baseline Forecast - Section: _Kaboudan metric_     
    **Error**:    
    >If the time series contains some predictable signals, would be lower than and η would approach <span style="color:red">*zero*</span>. This is because there was some information or patterns that were broken due to the  block shuffling. On the other hand, if a series is just white noise (which is unpredictable by definition) there would be hardly any difference between and , and η would approach <span style="color:red">*one*</span>   

    **Actual**:    
    >If the time series contains some predictable signals, would be lower than and η would approach <span style="color:red">*one*</span>. This is because there was some information or patterns that were broken due to the  block shuffling. On the other hand, if a series is just white noise (which is unpredictable by definition) there would be hardly any difference between and , and η would approach <span style="color:red">*zero*</span>

3. Chapter: 10 - Global Forecasting Models - Section: _Strategies to improve GFMs_    
    **Error**:
    > <span style="color:red">$E^{Local}_{out}$ and $E^{Global}_{out}$</span> are the average in-sample errors across all the time series using the local and global approaches, respectively. <span style="color:red">$E^{Global}_{out}$ and $E^{Global}_{in}$</span> are the out-of-sample expectations under the local and global approaches, respectively.

    **Actual**:
    > <span style="color:red">$E^{Local}_{in}$ and $E^{Global}_{in}$</span> are the average in-sample errors across all the time series using the local and global approaches, respectively. <span style="color:red">$E^{Global}_{out}$ and $E^{Global}_{out}$</span> are the out-of-sample expectations under the local and global approaches, respectively.

4. Chapter: 10 - Global Forecasting Models - Section: _Target Mean Encoding_    
    **Error**:    
    $$S_i = \lambda(n_i) \times \frac{\sum_{k \in TR_i}Y_k}{n_i} + (1-\lambda{n_i}) {\color{red} \frac{\sum_{k \in LTR}Y_k}{n_{TR}}}$$
    >Here, $TR_i$ is all the rows where $category = 1$ and $\sum_{k \in TR_i}Y_k$ is the sum of Y for $TR_i$. <span style="color:red">$\sum Y_{k \in TR}Y_k$</span> is the sum  of Y for all the rows in the training dataset.
    **Actual**:    
    $$S_i = \lambda(n_i) \times \frac{\sum_{k \in TR_i}Y_k}{n_i} + (1-\lambda{n_i}) {\color{red} \frac{\sum_{k \in TR}Y_k}{n_{TR}}}$$
    >Here, $TR_i$ is all the rows where $category = 1$ and $\sum_{k \in TR_i}Y_k$ is the sum of Y for $TR_i$. <span style="color:red">$\sum_{k \in TR}Y_k$</span> is the sum  of Y for all the rows in the training dataset.

5. Chapter: 13 - Attention and Transformers for Time Series - Section: _Self Attention_ and _Multi-headed Attention_    
    **Error**:    
    >In <span style="color:red">Figure 14.7 and 14.10</span>, the text says:
    `attn_scores = q @ v .permute(0,2,1)`    

    **Actual**:    
    >In <span style="color:red">Figure 14.7 and 14.10 <\span>, the text should be:
    `attn_scores = q @ k .permute(0,2,1)`

6. Preface: The required folder structure that was provided in the preface has a small correction.     
    **Error**:    
    ```
    data
    ├── london_smart_meters
    │   ├── hhblock_dataset
    │   │   ├── hhblock_dataset
    │   │       ├── block_0.csv
    │   │       ├── block_1.csv
    │   │       ├── ...
    │   │       ├── block_109.csv
    │── acorn_details.csv
    ├── informations_households.csv
    ├── uk_bank_holidays.csv
    ├── weather_daily_darksky.csv
    ├── weather_hourly_darksky.csv
    ```
    **Actual**: 
    ```
    data
    ├── london_smart_meters
    │   ├── hhblock_dataset
    │   │   ├── hhblock_dataset
    │   │       ├── block_0.csv
    │   │       ├── block_1.csv
    │   │       ├── ...
    │   │       ├── block_109.csv
    │   │── acorn_details.csv
    │   ├── informations_households.csv
    │   ├── uk_bank_holidays.csv
    │   ├── weather_daily_darksky.csv
    │   ├── weather_hourly_darksky.csv
    ```