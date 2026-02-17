import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("laptop_data.csv")

# Encode categorical features
le_brand = LabelEncoder()
le_processor = LabelEncoder()
le_os = LabelEncoder()

df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Processor'] = le_processor.fit_transform(df['Processor'])
df['Operating_System'] = le_os.fit_transform(df['Operating_System'])

# Features and Target
X = df[['Brand', 'Processor', 'Ram', 'Screen_Size', 'SSD', 'Operating_System']]
y = df['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(brand, processor, ram, screen_size, ssd, os):
    brand = le_brand.transform([brand])[0]
    processor = le_processor.transform([processor])[0]
    os = le_os.transform([os])[0]

    input_data = [[brand, processor, ram, screen_size, ssd, os]]
    prediction = model.predict(input_data)

    return round(prediction[0], 2)
