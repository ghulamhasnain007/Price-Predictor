from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

app = Flask(__name__)

# Load data and preprocess as previously discussed
df = pd.read_csv('zameen-property-data.csv')
df.drop(['property_id', 'location_id', 'page_url', 'date_added', 'longitude', 'latitude', 'agency', 'agent'], axis='columns', inplace=True)

def convert_area_to_sqft(area):
    try:
        area = str(area).replace(',', '')
        match = re.match(r'([0-9]*\.?[0-9]+)\s*(\w+)', area)
        if match:
            value, unit = match.groups()
            value = float(value)
            unit = unit.lower()
            if unit == 'marla':
                return value * 272.25
            elif unit == 'kanal':
                return value * 5445
            elif unit in ['sqft', 'square feet']:
                return value
            else:
                print(f"Unrecognized unit: {unit}")
                return None
        else:
            print(f"Unrecognized format: {area}")
            return None
    except Exception as e:
        print(f"Error processing area '{area}': {e}")
        return None

df['area'] = df['area'].apply(convert_area_to_sqft)

X = df.drop('price', axis=1)
y = df['price']

categorical_cols = ['property_type', 'location', 'city', 'province_name', 'purpose']
numerical_cols = ['baths', 'area', 'bedrooms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

clf.fit(X_train, y_train)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        property_type = request.form['property_type']
        location = request.form['location']
        city = request.form['city']
        province_name = request.form['province_name']
        baths = int(request.form['baths'])
        area = float(request.form['area'])
        purpose = request.form['purpose']
        bedrooms = int(request.form['bedrooms'])

        # Debug statements
        print(f"Received data: {request.form}")

        # Validate city and province_name against dataset
        if city not in df['city'].unique():
            print(f"City '{city}' is not in the dataset")
            return render_template('result.html', predicted_price='Error: City is not in the dataset!')
        if province_name not in df['province_name'].unique():
            print(f"Province '{province_name}' is not in the dataset")
            return render_template('result.html', predicted_price='Error: Province is not in the dataset!')

        # Predict price
        input_data = {
            'property_type': [property_type],
            'location': [location],
            'city': [city],
            'province_name': [province_name],
            'baths': [baths],
            'area': [area],
            'purpose': [purpose],
            'bedrooms': [bedrooms]
        }

        predicted_price = clf.predict(pd.DataFrame(input_data))[0]
        predicted_price = float(predicted_price)  # Convert to native Python float
        return render_template('result.html', predicted_price=f'{predicted_price:,.2f} PKR')

    except Exception as e:
        print(f"Error: {e}")
        return render_template('result.html', predicted_price=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
