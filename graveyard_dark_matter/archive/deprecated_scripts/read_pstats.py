import pstats
try:
    p = pstats.Stats('profile.pstats')
    p.sort_stats('tottime').print_stats(30)
except Exception as e:
    print(f"Error: {e}")
