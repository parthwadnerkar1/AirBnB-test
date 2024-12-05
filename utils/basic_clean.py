import pandas as pd
def clean_data(df): 
    
    def categorize_property_type(property_type):
        
        property_type = str(property_type).lower()  # Ensure it's a string and lowercase
        if property_type.startswith('entire'):
            return 'Entire Space'
        elif 'private' in property_type:
            return 'Private Small Space'
        elif 'shared' in property_type:
            return 'Shared Space'
        else:
            return 'Other'
    def categorize_price_range(price):
        """
        Categorizes price into specific price ranges.
        """
        price = pd.to_numeric(price, errors='coerce')
        if price < 10:
            return "Below $10"
        elif 10 <= price < 500:
            return "$10 - $500"
        elif 500 <= price < 1000:
            return "$500 - $1000"
        elif 1000 <= price < 5000:
            return "$1000 - $5000"
        elif 5000 <= price < 10000:
            return "$5000 - $10000"
        elif 10000 <= price < 50000:
            return "$10000 - $50000"
        else:
            return "$50000+"
    # df['Price'] = pd.to_numeric(df['Price'])
    df['Price'] = df['Price'].replace({'\$': '', ',': ''}, regex=True)
    df['price_range'] = df['Price'].apply(categorize_price_range)


    # Apply the nested function to the 'property_type' column of the DataFrame
    df['property_category'] = df['Property Type'].apply(categorize_property_type)
    
    return df
