import json
import os

def main():
    transcript_path = r"C:\Users\jhona\.gemini\antigravity\brain\6b0bf5e2-4c5f-42eb-9c0a-cf861eb08d00\.system_generated\logs\transcript.jsonl"
    if not os.path.exists(transcript_path):
        print("Transcript path does not exist.")
        return
        
    print(f"Reading transcript from {transcript_path}...")
    
    # We want to find the last user message
    user_inputs = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("type") == "USER_INPUT":
                    user_inputs.append(data)
            except Exception as e:
                pass
                
    if not user_inputs:
        print("No USER_INPUT found.")
        return
        
    last_user_input = user_inputs[-1]
    content = last_user_input.get("content", "")
    print(f"Found user input. Length: {len(content)} characters.")
    
    output_path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\scratch\untruncated_user_prompt.txt"
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(content)
        
    print(f"Untruncated prompt written to {output_path}")

if __name__ == "__main__":
    main()
