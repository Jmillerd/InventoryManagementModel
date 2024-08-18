import pandas as pd

# Sample inventory data in a DataFrame
data = {
    'ProductID': [101, 102, 103, 104, 105],
    'ProductName': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
    'InitialStock': [150, 200, 100, 75, 300],
    'Sales': [120, 180, 80, 60, 240],
    'Purchases': [200, 250, 120, 90, 280],
}

inventory_df = pd.DataFrame(data)

# Calculate inventory turnover for each product
inventory_df['InventoryTurnover'] = inventory_df['Sales'] / ((inventory_df['InitialStock'] + inventory_df['Purchases']) / 2)

# Calculate stockout rate for each product
inventory_df['StockoutRate'] = (inventory_df['InitialStock'] + inventory_df['Purchases'] - inventory_df['Sales']) / inventory_df['Purchases']

# Display the inventory data with turnover and stockout rate
print("Inventory Data with Turnover and Stockout Rate:")
print(inventory_df[['ProductID', 'ProductName', 'InventoryTurnover', 'StockoutRate']])
