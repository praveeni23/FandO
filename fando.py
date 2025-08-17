import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model3 = XGBClassifier(eval_metric='logloss', random_state=42)
model4 = LogisticRegression(max_iter=7000)

from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()



nifty = yf.download("^NSEI",period="10y",interval="1d",auto_adjust="True")

print("Column Names from yfinance:")
print(nifty.columns)

print(nifty.head())
print(f"Total trading days fetched: {len(nifty)}")
print(nifty.tail())

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = [col[0] for col in nifty.columns]



nifty['Target'] = (nifty['Close'].shift(-1)>nifty['Close']).astype(int)
nifty['Return'] = nifty['Close'].pct_change()

nifty['MA_10'] = nifty['Close'].rolling(10).mean()
nifty['MA_50'] = nifty['Close'].rolling(50).mean()

nifty.dropna(inplace=True)
delta = nifty['Close'].diff()

gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
nifty['RSI_14'] = 100 - (100 / (1 + rs))

# Bollinger Bands
nifty['BB_upper'] = nifty['MA_10'] + 2 * nifty['Close'].rolling(10).std()
nifty['BB_lower'] = nifty['MA_10'] - 2 * nifty['Close'].rolling(10).std()

# MACD
ema_12 = nifty['Close'].ewm(span=12).mean()
ema_26 = nifty['Close'].ewm(span=26).mean()
nifty['MACD'] = ema_12 - ema_26
nifty['MACD_signal'] = nifty['MACD'].ewm(span=9).mean()

# ATR
nifty['H-L'] = nifty['High'] - nifty['Low']
nifty['H-PC'] = abs(nifty['High'] - nifty['Close'].shift(1))
nifty['L-PC'] = abs(nifty['Low'] - nifty['Close'].shift(1))
nifty['TR'] = nifty[['H-L', 'H-PC', 'L-PC']].max(axis=1)
nifty['ATR'] = nifty['TR'].rolling(14).mean()

# Drop NaNs
nifty.dropna(inplace=True)
nifty['MA_10'] = nifty['Close'].rolling(10).mean()
rolling_std = nifty['Close'].rolling(10).std()

nifty['BB_upper'] = nifty['MA_10'] + 2 * rolling_std
nifty['BB_lower'] = nifty['MA_10'] - 2 * rolling_std




features =['Open','High','Low','Close','Return', 'MA_10','MA_50','RSI_14','BB_upper',
           'BB_lower','MACD','MACD_signal','ATR']
X = nifty[features]
y = nifty['Target']

X = X.fillna(X.mean())
y = y.loc[X.index]

nifty = nifty.loc[X.index]


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)


X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

voting_clf = VotingClassifier(
    estimators=[('rf',model1),('gb',model2),('xgb',model3),('lr',model4)],voting='soft')

voting_clf.fit(X_train,y_train)
preds =voting_clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test , preds))

nifty_test = nifty.iloc[len(X_train):].copy()
nifty_test.loc[:, 'Prediction'] = preds


print(nifty['Target'].value_counts(normalize=True))
print(classification_report(y_test, preds))
print(nifty_test[['Close','Target','Prediction']].tail())

if X.empty or y.empty:
    print("Data is empty. Check your internet connection or ticker symbol.")
    exit()

