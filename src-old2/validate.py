"""
Quick validation script to test the forecasting pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data():
    """Check if preprocessed data exists and is valid"""
    orders_path = Path("data/processed/orders_processed.parquet")
    items_path = Path("data/processed/items_processed.parquet")
    
    if not orders_path.exists():
        logging.error("❌ orders_processed.parquet not found")
        return False
    
    if not items_path.exists():
        logging.error("❌ items_processed.parquet not found")
        return False
    
    logging.info("✓ Found preprocessed data files")
    
    # Load and validate
    df_orders = pd.read_parquet(orders_path)
    df_items = pd.read_parquet(items_path)
    
    logging.info(f"✓ Orders shape: {df_orders.shape}")
    logging.info(f"✓ Items shape: {df_items.shape}")
    
    # Check required columns
    required_order_cols = [
        'orderDate', 'itemId', 'salesQuantityKgL', 'is_campaign_week',
        'cluster1To1Id', 'week', 'month', 'year'
    ]
    
    missing = [col for col in required_order_cols if col not in df_orders.columns]
    if missing:
        logging.error(f"❌ Missing order columns: {missing}")
        return False
    
    logging.info("✓ All required columns present")
    
    # Check for campaign data
    campaign_items = df_orders[df_orders['is_campaign_week'] == 1]['itemId'].nunique()
    logging.info(f"✓ Found {campaign_items} unique items with campaign history")
    
    # Sample some item IDs for testing
    sample_items = df_orders['itemId'].value_counts().head(5).index.tolist()
    logging.info(f"✓ Sample item IDs for testing: {sample_items[:3]}")
    
    return True


def get_sample_items(n=3):
    """Get sample item IDs that have campaign history"""
    orders_path = Path("data/processed/orders_processed.parquet")
    
    if not orders_path.exists():
        logging.error("No preprocessed data found. Run preprocessing first.")
        return []
    
    df_orders = pd.read_parquet(orders_path)
    
    # Get items with campaign history and sufficient data
    campaign_items = (
        df_orders[df_orders['is_campaign_week'] == 1]
        .groupby('itemId')
        .size()
        .sort_values(ascending=False)
    )
    
    sample = campaign_items.head(n).index.tolist()
    
    print("\n" + "="*60)
    print("SAMPLE ITEMS FOR TESTING")
    print("="*60)
    for i, item_id in enumerate(sample, 1):
        count = campaign_items[item_id]
        print(f"{i}. Item ID: {item_id} ({count} campaign weeks)")
    print("="*60)
    print(f"\nTest command:")
    print(f'python src/forecast.py --items "{",".join(sample)}" --start "2025-11-18" --end "2025-12-08"')
    print("="*60 + "\n")
    
    return sample


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        get_sample_items(n=5)
    else:
        validate_data()
