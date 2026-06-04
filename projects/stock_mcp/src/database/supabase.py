from supabase import create_client, Client
from src.core.config import settings

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def fetch_all_portfolios(user_id: int) -> list[dict]:
    response = supabase.table("portfolios").select("*").eq("user_id", user_id).execute()
    return response.data


# 💡 This block ONLY runs if this file is executed directly! python -m src.database.supabase
if __name__ == "__main__":
    import asyncio
    from pprint import pprint # Pretty-print makes JSON data look nice in the terminal
    
    print("🚀 Testing Supabase Database Connection...")
    try:
        # Assuming User ID 1 is what you seeded earlier
        test_user_id = 1
        data = fetch_all_portfolios(user_id=test_user_id)
        
        print(f"\n✅ Query Successful! Found {len(data)} records for User {test_user_id}:")
        pprint(data)
        
    except Exception as e:
        print(f"\n❌ Error connecting to Supabase: {e}")