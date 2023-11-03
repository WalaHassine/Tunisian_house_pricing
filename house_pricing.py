import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('housing_data.csv')

df['price_eur'] = df['price_tnd'] * 0.31

missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    df = df.dropna()

X = df[['area', 'room', 'bathroom', 'age', 'state', 'latt', 'long', 'distance_to_capital', 'garage',
       'garden', 'concierge', 'beach_view', 'mountain_view', 'pool', 'elevator', 'furnished',
       'equipped_kitech', 'central_heating', 'air_conditionting']]
y = df['price_eur']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
