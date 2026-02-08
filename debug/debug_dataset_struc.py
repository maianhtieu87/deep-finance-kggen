# debug_dataset_structure.py - Check what's in unified_dataset

import pickle
import os
from configs.config import GlobalConfig

def debug_dataset():
    """Debug unified_dataset structure to find the issue"""
    
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    
    print("=" * 60)
    print("🔍 DEBUG UNIFIED DATASET STRUCTURE")
    print("=" * 60)
    
    if not os.path.exists(pkl_path):
        print(f"❌ Dataset not found: {pkl_path}")
        return
    
    print(f"\n📥 Loading dataset...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   Total dates: {len(data)}")
    
    # Sample 3 dates
    sample_dates = list(data.keys())[:3]
    
    print(f"\n📊 ANALYZING STRUCTURE:")
    
    for date in sample_dates:
        day_data = data[date]
        
        print(f"\n{'='*60}")
        print(f"Date: {date}")
        print(f"{'='*60}")
        
        # Show all top-level keys
        print(f"\n1. Top-level keys:")
        for key in day_data.keys():
            print(f"   - {key}")
        
        # Check news_embedding (OLD)
        print(f"\n2. news_embedding field (OLD FORMAT):")
        if "news_embedding" in day_data:
            news_emb = day_data["news_embedding"]
            if isinstance(news_emb, dict):
                print(f"   Type: dict")
                print(f"   Keys (tickers): {list(news_emb.keys())}")
                
                # Check first ticker
                if news_emb:
                    first_ticker = list(news_emb.keys())[0]
                    first_val = news_emb[first_ticker]
                    print(f"   Sample ticker: {first_ticker}")
                    print(f"   Sample value type: {type(first_val)}")
                    if isinstance(first_val, list):
                        print(f"   Sample value length: {len(first_val)}")
            else:
                print(f"   Type: {type(news_emb)}")
        else:
            print(f"   ❌ Field NOT present")
        
        # Check kg_tensor (NEW)
        print(f"\n3. kg_tensor field (NEW FORMAT):")
        if "kg_tensor" in day_data:
            kg_tensor = day_data["kg_tensor"]
            if isinstance(kg_tensor, dict):
                print(f"   Type: dict")
                print(f"   Keys (tickers): {list(kg_tensor.keys())}")
                
                if kg_tensor:
                    first_ticker = list(kg_tensor.keys())[0]
                    first_val = kg_tensor[first_ticker]
                    print(f"   Sample ticker: {first_ticker}")
                    print(f"   Sample value type: {type(first_val)}")
                    
                    if isinstance(first_val, str):
                        print(f"   Sample value (path): {first_val}")
                        print(f"   Path exists: {os.path.exists(first_val)}")
                else:
                    print(f"   ⚠️  Empty dict")
            else:
                print(f"   Type: {type(kg_tensor)}")
        else:
            print(f"   ❌ Field NOT present")
        
        # Check price
        print(f"\n4. price field:")
        if "price" in day_data:
            price = day_data["price"]
            print(f"   Type: {type(price)}")
            if isinstance(price, dict):
                print(f"   Tickers with price: {list(price.keys())}")
        
    # Summary
    print(f"\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    # Count days with each field
    has_news_emb = 0
    has_kg_tensor = 0
    kg_tensor_with_data = 0
    
    for date, day_data in data.items():
        if "news_embedding" in day_data and day_data["news_embedding"]:
            has_news_emb += 1
        
        if "kg_tensor" in day_data:
            has_kg_tensor += 1
            if day_data["kg_tensor"]:  # Not empty
                kg_tensor_with_data += 1
    
    print(f"\nDates with news_embedding: {has_news_emb}/{len(data)} ({has_news_emb/len(data)*100:.1f}%)")
    print(f"Dates with kg_tensor field: {has_kg_tensor}/{len(data)} ({has_kg_tensor/len(data)*100:.1f}%)")
    print(f"Dates with kg_tensor data: {kg_tensor_with_data}/{len(data)} ({kg_tensor_with_data/len(data)*100:.1f}%)")
    
    # Diagnosis
    print(f"\n" + "=" * 60)
    print("💡 DIAGNOSIS")
    print("=" * 60)
    
    if has_news_emb > 0 and kg_tensor_with_data == 0:
        print("\n❌ PROBLEM FOUND:")
        print("   - Dataset has 'news_embedding' (old format) ✅")
        print("   - Dataset has NO 'kg_tensor' data (new format) ❌")
        print("\n🔧 SOLUTION:")
        print("   Option 1: Use old data_loader.py (works with news_embedding)")
        print("   Option 2: Rebuild dataset with kg_tensor:")
        print("             python main_test.py")
        
    elif kg_tensor_with_data > 0:
        print("\n✅ Dataset has kg_tensor data")
        print("   Issue may be in data_loader logic")
        
    else:
        print("\n⚠️  Dataset missing both news_embedding and kg_tensor")
        print("   Need to rebuild: python main_test.py")

if __name__ == "__main__":
    debug_dataset()