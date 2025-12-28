# @title iTransformer Alpha System (Fixed Training Version)

!pip install -q yfinance pandas numpy torch scikit-learn matplotlib

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import matplotlib.pyplot as plt
import os
import re
import copy

# 1. 환경 설정 및 데이터 로더
plt.style.use('seaborn-v0_8-darkgrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"System Device: {device}")

MODEL_PATH = 'alpha_iTransformer_fixed.pth'
WINDOW_SIZE = 60

def clean_currency(x):
    if isinstance(x, str):
        return float(re.sub(r'[$,]', '', x))
    return x

def load_user_data(file_paths):
    data_dict = {}
    name_map = {
        'PALANTIR': 'PLTR', 'NVIDIA': 'NVDA', 'BROADCOM': 'AVGO',
        'ALPHABET': 'GOOGL', 'AMAZON': 'AMZN', 'NETFLIX': 'NFLX',
        'TESLA': 'TSLA', 'META': 'META', 'MICROSOFT': 'MSFT', 'APPLE': 'AAPL'
    }

    print("Loading user CSV files...")
    for path in file_paths:
        filename = os.path.basename(path).upper()
        ticker = next((val for key, val in name_map.items() if key in filename), None)

        if ticker:
            try:
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                if 'Close/Last' in df.columns:
                    df['Close'] = df['Close/Last'].apply(clean_currency)
                elif 'Close' in df.columns:
                     df['Close'] = df['Close'].apply(clean_currency)
                df = df[['Date', 'Close']].set_index('Date').sort_index()
                data_dict[ticker] = df['Close']
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not data_dict:
        raise ValueError("No data loaded. Please check file paths.")

    combined_df = pd.DataFrame(data_dict).dropna()
    print(f"Data merge complete: {len(combined_df)} days, {len(combined_df.columns)} assets")
    return combined_df

# 2. 데이터 전처리 (Rolling Normalization)
def create_rolling_dataset(returns_df, window_size):
    X, y = [], []
    data_values = returns_df.values
    
    if len(data_values) <= window_size:
        return np.array([]), np.array([])

    for i in range(len(data_values) - window_size):
        # 1. 입력 윈도우 (과거 60일)
        raw_window = data_values[i : i + window_size]
        
        # 2. 윈도우 내부 통계로 정규화
        w_mean = np.mean(raw_window, axis=0)
        w_std = np.std(raw_window, axis=0) + 1e-6
        norm_window = (raw_window - w_mean) / w_std
        
        # 3. 타겟
        raw_target = data_values[i + window_size]
        norm_target = (raw_target - w_mean) / w_std

        X.append(norm_window)
        y.append(norm_target) 
        
    return np.array(X), np.array(y)

# 3. 모델 정의
class AlphaLoss(nn.Module):
    def __init__(self, lambda_ic=2.0, lambda_sign=1.0):
        super(AlphaLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_ic = lambda_ic
        self.lambda_sign = lambda_sign

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        
        # IC (Information Coefficient) Loss
        vx = pred - torch.mean(pred, dim=1, keepdim=True)
        vy = target - torch.mean(target, dim=1, keepdim=True)
        cov = torch.sum(vx * vy, dim=1)
        std_x = torch.sqrt(torch.sum(vx ** 2, dim=1))
        std_y = torch.sqrt(torch.sum(vy ** 2, dim=1))
        ic = cov / (std_x * std_y + 1e-6)
        ic_loss = 1 - torch.mean(ic)
        
        # Sign Loss (방향성)
        sign_mismatch = torch.mean(torch.relu(-torch.sign(pred) * torch.sign(target)))
        
        return mse + (self.lambda_ic * ic_loss) + (self.lambda_sign * sign_mismatch)

class iTransformerAlpha(nn.Module):
    def __init__(self, num_stocks, lookback_window, d_model=128, n_heads=4):
        super(iTransformerAlpha, self).__init__()
        self.embedding = nn.Linear(lookback_window, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            batch_first=True, 
            dropout=0.3, 
            dim_feedforward=d_model*2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.predictor = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x_emb = self.embedding(x)
        enc_out = self.encoder(x_emb)
        return self.predictor(enc_out).squeeze(-1)

# 4. 학습 함수
def train_model(data_df):
    print("\n[Train Mode] Start Training (Fixed Epochs / No Early Stopping)...")
    
    returns_df = np.log(data_df / data_df.shift(1)).dropna()
    
    # Train / Validation 분리 (최근 60일은 Val, 그 전은 Train)
    train_raw = returns_df.iloc[:-60]
    val_raw = returns_df 

    X_train, y_train = create_rolling_dataset(train_raw, WINDOW_SIZE)
    X_val_all, y_val_all = create_rolling_dataset(val_raw, WINDOW_SIZE)
    
    val_len = 60
    if len(X_val_all) > val_len:
        X_val = X_val_all[-val_len:]
        y_val = y_val_all[-val_len:]
    else:
        X_val, y_val = X_val_all, y_val_all

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    model = iTransformerAlpha(num_stocks=data_df.shape[1], lookback_window=WINDOW_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = AlphaLoss()

    model.train()
    epochs = 300

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Training Loop
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation Loop (모니터링용)
        with torch.no_grad():
            model.eval()
            val_preds = model(X_val_tensor)
            val_loss = criterion(val_preds, y_val_tensor)
            model.train()
        
        if (epoch+1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss.item():.5f}")

    # 학습 종료 후 최종 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training Complete. Model saved to: {MODEL_PATH}")

# 5. 백테스팅
def rolling_backtest_report(full_data_df):
    print("\n" + "="*80)
    print("BACKTEST REPORT (Fully Invested / No Cash)")
    print("="*80)

    returns_df = np.log(full_data_df / full_data_df.shift(1)).dropna()
    tickers = returns_df.columns.tolist()
    num_assets = len(tickers)

    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    model = iTransformerAlpha(num_stocks=num_assets, lookback_window=WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    TEST_DAYS = 30
    if len(returns_df) < TEST_DAYS + WINDOW_SIZE:
        print("Not enough data.")
        return

    test_idx_start = len(returns_df) - TEST_DAYS
    test_dates = returns_df.index[test_idx_start:]

    # QQQ Benchmark
    try:
        start_q = test_dates[0] - pd.Timedelta(days=5)
        end_q = test_dates[-1] + pd.Timedelta(days=5)
        qqq_raw = yf.download('QQQ', start=start_q, end=end_q, progress=False)
        if isinstance(qqq_raw.columns, pd.MultiIndex):
            qqq_close = qqq_raw['Close']['QQQ']
        else:
            qqq_close = qqq_raw['Close']
        qqq_close = qqq_close.dropna().tz_localize(None)
        qqq_daily_ret = np.log(qqq_close / qqq_close.shift(1)).dropna()
    except:
        qqq_daily_ret = pd.Series(dtype=float)

    portfolio_value = [1.0]
    qqq_value = [1.0]
    logs = []

    print(f"Test Period: {test_dates[0].date()} ~ {test_dates[-1].date()}")

    for t in range(test_idx_start, len(returns_df)):
        current_date = returns_df.index[t]

        # 1. Prediction with Rolling Norm
        raw_window = returns_df.iloc[t-WINDOW_SIZE : t].values
        
        w_mean = np.mean(raw_window, axis=0)
        w_std = np.std(raw_window, axis=0) + 1e-6
        scaled_window = (raw_window - w_mean) / w_std
        
        input_tensor = torch.FloatTensor(scaled_window).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_alpha = model(input_tensor).cpu().numpy().flatten()

        # 2. Score Calculation
        score_mean = np.mean(pred_alpha)
        score_std = np.std(pred_alpha) + 1e-6
        z_scores = (pred_alpha - score_mean) / score_std

        # 3. Softmax Allocation
        TEMPERATURE = 0.75 
        exp_scores = np.exp(z_scores / TEMPERATURE)
        weights = exp_scores / np.sum(exp_scores)
        
        weights = np.round(weights, 4)
        weights = weights / np.sum(weights)

        # 4. Return Calculation
        real_ret = returns_df.iloc[t].values
        my_ret = np.dot(weights, real_ret)

        if not qqq_daily_ret.empty and current_date in qqq_daily_ret.index:
            q_ret = qqq_daily_ret.loc[current_date]
        else:
            q_ret = 0.0

        portfolio_value.append(portfolio_value[-1] * (1 + my_ret))
        qqq_value.append(qqq_value[-1] * (1 + q_ret))

        log_entry = {
            "Date": current_date.date(),
            "My Return": f"{my_ret*100:6.2f}%",
            "QQQ Return": f"{q_ret*100:6.2f}%",
            "Result": "WIN" if my_ret > q_ret else "LOSS"
        }
        for idx, ticker in enumerate(tickers):
            log_entry[ticker] = f"{weights[idx]*100:.1f}%"
        logs.append(log_entry)

    # Final Report
    df_log = pd.DataFrame(logs)
    cols = ["Date", "My Return", "QQQ Return", "Result"] + tickers
    df_log = df_log[cols]

    print("\n[Daily Portfolio Weight & Performance Report]")
    print("-" * 150)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_log.to_string(index=False))
    print("-" * 150)

    final_my = (portfolio_value[-1] - 1) * 100
    final_qqq = (qqq_value[-1] - 1) * 100
    wins = df_log['Result'].value_counts().get('WIN', 0)

    print("\n[Final Summary]")
    print(f" Win Rate: {(wins/len(df_log)*100):.1f}%")
    print(f" My Return: {final_my:.2f}%")
    print(f" QQQ Return: {final_qqq:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, np.array(portfolio_value[1:])-1, label='OUR MODEL', color='blue')
    plt.plot(test_dates, np.array(qqq_value[1:])-1, label='QQQ', color='gray', linestyle='--')
    plt.title("Performance: AI vs QQQ")
    plt.legend()
    plt.show()

# 6. 메인 실행
file_list = [
    'PALANTIR 5년치.csv', 'NVIDIA 5년.csv', 'BROADCOM 5년치.csv',
    'ALPHABET C 5년치.csv', 'AMAZON 5년치.csv', 'NETFLIX 5년치.csv',
    'TESLA 5년치.csv', 'META 5년치.csv', 'MICROSOFT 5년치.csv', 'APPLE 5년치.csv'
]

try:
    full_data = load_user_data(file_list)
    train_model(full_data)
    rolling_backtest_report(full_data)
except Exception as e:
    print(f"\nCritical Error: {e}")
