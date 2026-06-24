import re

path = r'C:/Users/jhona/Documents/Proyectos/Trader Gemini/risk/risk_manager.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# First replace the assignment from metrics dictionary
content = content.replace(
    "order_flow = signal_metadata['metrics']['order_flow']",
    "order_flow = signal_metadata.get('metrics', {}).get('order_flow', {})"
)
content = content.replace(
    "order_flow = signal_metadata['metrics']['order_flow'] if signal_metadata else {}",
    "order_flow = signal_metadata.get('metrics', {}).get('order_flow', {}) if signal_metadata else {}"
)

# Then safely replace array index accesses
content = re.sub(r"order_flow\['tick_volatility'\]", "order_flow.get('tick_volatility', 0.0)", content)
content = re.sub(r"order_flow\['toxicity_index'\]", "order_flow.get('toxicity_index', 0.0)", content)
content = re.sub(r"order_flow\['delta'\]", "order_flow.get('delta', 0.0)", content)
content = re.sub(r"order_flow\['is_spoofing'\]", "order_flow.get('is_spoofing', False)", content)
content = re.sub(r"order_flow\['spoofing_side'\]", "order_flow.get('spoofing_side', None)", content)
content = re.sub(r"order_flow\['gamma_expansion_risk'\]", "order_flow.get('gamma_expansion_risk', False)", content)
content = re.sub(r"order_flow\['magnetic_pull_up'\]", "order_flow.get('magnetic_pull_up', 0.0)", content)
content = re.sub(r"order_flow\['magnetic_pull_down'\]", "order_flow.get('magnetic_pull_down', 0.0)", content)
content = re.sub(r"order_flow\['high_micro_entropy'\]", "order_flow.get('high_micro_entropy', False)", content)
content = re.sub(r"order_flow\['entropy'\]", "order_flow.get('entropy', 0.0)", content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fix completed.")
