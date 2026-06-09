try:
    assert False
except AssertionError:
    print("AssertionError works")
except Exception as e:
    print(f"Different exception: {type(e).__name__}")
