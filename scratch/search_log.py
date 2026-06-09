import os

def search_log():
    log_path = r"C:\Users\jhona\.gemini\antigravity\brain\8a667ead-cb41-4594-b841-e22b336556fa\.system_generated\tasks\task-252.log"
    if not os.path.exists(log_path):
        print("Log file not found.")
        return
        
    keywords = ["[TREND", "[ORACLE", "[VOLATILITY", "reject", "blocked", "Signal", "Setup", "BBO", "Killed", "Veto", "RSI"]
    
    print("Searching log for keywords...")
    matches = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            for kw in keywords:
                if kw.lower() in line.lower():
                    # Check if it's not just the epoch status line (to avoid clutter)
                    if "Epoch" in line and "Open: 0" in line:
                        continue
                    print(f"L{i}: {line.strip()}")
                    matches += 1
                    break
            if matches >= 100:
                print("Showed first 100 matches.")
                break
                
    print(f"Total matches found: {matches}")

if __name__ == "__main__":
    search_log()
