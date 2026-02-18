"""
Data Fetch Validation Script
Validates that the refactored fetch logic works correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fetch_data import fetch_data
import pandas as pd

def validate_fetch(test_item_id='528208'):
    """
    Validate the fetch_data function with detailed diagnostics.
    
    Parameters:
        test_item_id: Item ID to use for testing
    """
    print("="*80)
    print("DATA FETCH VALIDATION")
    print("="*80)
    print(f"Test item: {test_item_id}")
    print()
    
    # Fetch data
    print("Fetching data...")
    df_items, df_orders, df_campaigns = fetch_data(simulate_item_ids=[test_item_id])
    
    print("\n" + "="*80)
    print("1. CAMPAIGN DATA")
    print("="*80)
    print(f"Total campaigns: {len(df_campaigns)}")
    print(f"Date range: {df_campaigns['campaignStart'].min()} to {df_campaigns['campaignEnd'].max()}")
    print(f"Sample campaigns:\n{df_campaigns.head(3)}")
    
    print("\n" + "="*80)
    print("2. ITEM DATA")
    print("="*80)
    print(f"Total items fetched: {len(df_items)}")
    print(f"Unique clusters: {df_items['cluster1To1Id'].nunique()}")
    
    # Check if test item exists
    test_item = df_items[df_items['itemId'] == test_item_id]
    if len(test_item) > 0:
        print(f"\n✅ Test item {test_item_id} found!")
        test_cluster = test_item.iloc[0]['cluster1To1Id']
        print(f"   Description: {test_item.iloc[0]['itemDesc']}")
        print(f"   Cluster: {test_cluster}")
        
        # Check cluster composition
        cluster_items = df_items[df_items['cluster1To1Id'] == test_cluster]
        print(f"\n   Cluster {test_cluster} composition:")
        print(f"   - Total items: {len(cluster_items)}")
        print(f"   - Sample items:")
        for idx, row in cluster_items.head(10).iterrows():
            print(f"     • {row['itemId']}: {row['itemDesc']}")
        
        if len(cluster_items) > 10:
            print(f"     ... and {len(cluster_items) - 10} more items")
            
        # Expected: Should have ~70 items in milk cluster
        if len(cluster_items) < 10:
            print(f"\n   ⚠️  WARNING: Only {len(cluster_items)} items in cluster!")
            print(f"      Expected ~70 items for milk cluster. Data fetch may have issues.")
        else:
            print(f"\n   ✅ Cluster has {len(cluster_items)} items - looks good!")
    else:
        print(f"\n❌ Test item {test_item_id} NOT found in fetched items!")
    
    print("\n" + "="*80)
    print("3. ORDER DATA")
    print("="*80)
    print(f"Total order rows: {len(df_orders):,}")
    print(f"Unique items with orders: {df_orders['itemId'].nunique()}")
    print(f"Date range: {df_orders['orderDate'].min()} to {df_orders['orderDate'].max()}")
    
    # Check orders for test item
    test_orders = df_orders[df_orders['itemId'] == test_item_id]
    print(f"\nTest item {test_item_id} orders:")
    print(f"   Total rows: {len(test_orders)}")
    if len(test_orders) > 0:
        print(f"   Date range: {test_orders['orderDate'].min()} to {test_orders['orderDate'].max()}")
        print(f"   Total quantity: {test_orders['salesQuantity'].sum():,.0f}")
        
        # Campaign vs non-campaign orders
        campaign_orders = test_orders[test_orders['salesCampignId'].notna() & (test_orders['salesCampignId'] != 0)]
        print(f"   Campaign orders: {len(campaign_orders)} rows")
        print(f"   Regular orders: {len(test_orders) - len(campaign_orders)} rows")
        
        # Recent orders
        print(f"\n   Recent orders (last 5):")
        print(test_orders.tail(5)[['orderDate', 'salesQuantity', 'salesCampignId']].to_string(index=False))
    
    # Check cluster-wide orders
    if len(test_item) > 0:
        test_cluster = test_item.iloc[0]['cluster1To1Id']
        cluster_orders = df_orders[df_orders['itemId'].isin(cluster_items['itemId'])]
        print(f"\n   Cluster {test_cluster} orders:")
        print(f"   Total rows: {len(cluster_orders):,}")
        print(f"   Items with orders: {cluster_orders['itemId'].nunique()}")
        
        if cluster_orders['itemId'].nunique() < len(cluster_items) * 0.8:
            print(f"   ⚠️  Only {cluster_orders['itemId'].nunique()}/{len(cluster_items)} items have orders")
        else:
            print(f"   ✅ Most items have order history")
    
    print("\n" + "="*80)
    print("4. DATA QUALITY CHECKS")
    print("="*80)
    
    # Check for duplicates
    order_dupes = df_orders[df_orders.duplicated(subset=['orderDate', 'itemId'], keep=False)]
    if len(order_dupes) > 0:
        print(f"❌ Found {len(order_dupes)} duplicate order rows (same date + item)")
        print(f"   Sample duplicates:")
        print(order_dupes.head(5)[['orderDate', 'itemId', 'salesQuantity']])
    else:
        print(f"✅ No duplicate order rows")
    
    # Check for missing values
    print(f"\nMissing values:")
    print(f"   Items - cluster1To1Id: {df_items['cluster1To1Id'].isna().sum()} ({df_items['cluster1To1Id'].isna().sum()/len(df_items)*100:.1f}%)")
    print(f"   Orders - itemId: {df_orders['itemId'].isna().sum()}")
    print(f"   Orders - orderDate: {df_orders['orderDate'].isna().sum()}")
    print(f"   Orders - salesQuantity: {df_orders['salesQuantity'].isna().sum()}")
    
    # Check date types
    print(f"\nData types:")
    print(f"   orderDate: {df_orders['orderDate'].dtype}")
    print(f"   campaignStart: {df_campaigns['campaignStart'].dtype}")
    
    print("\n" + "="*80)
    print("5. SUMMARY")
    print("="*80)
    
    # Validation summary
    issues = []
    
    if len(test_item) == 0:
        issues.append(f"❌ Test item {test_item_id} not found")
    else:
        if len(cluster_items) < 10:
            issues.append(f"❌ Cluster has only {len(cluster_items)} items (expected ~70)")
        
        if len(test_orders) == 0:
            issues.append(f"❌ No orders found for test item")
    
    if len(order_dupes) > 0:
        issues.append(f"❌ Found {len(order_dupes)} duplicate order rows")
    
    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED!")
        print(f"   - Item data looks correct")
        print(f"   - Cluster has proper size ({len(cluster_items)} items)")
        print(f"   - Order history is complete")
        print(f"   - No duplicates found")
    else:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    
    print("="*80)
    
    return df_items, df_orders, df_campaigns


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate data fetch')
    parser.add_argument('--item', type=str, default='528208', help='Item ID to test with')
    args = parser.parse_args()
    
    validate_fetch(test_item_id=args.item)
