try:
    from core.enums import OrderSide, OrderType
    print("Import Successful")
except ImportError as e:
    print(f"Import Failed: {e}")
except Exception as e:
    print(f"Other Error: {e}")
