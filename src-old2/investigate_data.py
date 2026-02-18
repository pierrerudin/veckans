"""
Quick script to investigate what data is being fetched for item 528208
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fetch_data import fetch_data
import pandas as pd

# Fetch data for item 528208
print("=" * 80)
print("Fetching data for item 528208...")
print("=" * 80)

df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=['528208'])

print("\n" + "=" * 80)
print("ITEM 528208 DETAILS")
print("=" * 80)
item_528208 = df_items[df_items['itemId'] == '528208']
if len(item_528208) > 0:
    item = item_528208.iloc[0]
    print(f"Item ID: {item['itemId']}")
    print(f"Item Name: {item.get('itemDesc', 'N/A')}")
    print(f"Cluster ID: {item['cluster1To1Id']}")
    print(f"Weight (prepared): {item.get('weightKgPreparedItemComparisonUnit', 'N/A')} kg")
    print(f"Weight (net): {item.get('netWeightKgComparisonUnit', 'N/A')} kg")
    print(f"Weight (gross): {item.get('grossWeightKgComparisonUnit', 'N/A')} kg")
    
    cluster_id = item['cluster1To1Id']
    
    print("\n" + "=" * 80)
    print(f"ALL ITEMS IN CLUSTER {cluster_id}")
    print("=" * 80)
    
    cluster_items = df_items[df_items['cluster1To1Id'] == cluster_id]
    print(f"\nTotal items in cluster: {len(cluster_items)}")
    print(f"\nItems (first 20):")
    print(cluster_items[['itemId', 'itemDesc', 'weightKgPreparedItemComparisonUnit']].head(20).to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"ITEMS IN CLUSTER WITH ORDER HISTORY")
    print("=" * 80)
    
    items_with_orders = df_orders[df_orders['cluster1To1Id'] == cluster_id]['itemId'].unique()
    print(f"\nItems in cluster with order history: {len(items_with_orders)}")
    print(f"Item IDs: {sorted(items_with_orders)}")
    
    # Check if 528208 has orders
    item_orders = df_orders[df_orders['itemId'] == '528208']
    print(f"\nOrders for item 528208: {len(item_orders)} rows")
    if len(item_orders) > 0:
        print(f"Date range: {item_orders['orderDate'].min()} to {item_orders['orderDate'].max()}")
        print(f"Sample orders:\n{item_orders[['orderDate', 'salesQuantity']].head(10)}")
    
else:
    print("ERROR: Item 528208 not found in fetched data!")
    print(f"\nTotal items fetched: {len(df_items)}")
    print(f"Sample item IDs: {df_items['itemId'].head(20).tolist()}")
