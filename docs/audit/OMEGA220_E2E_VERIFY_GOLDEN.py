"""Independent binary64 check of the synthetic GOLDEN witness from CI logs."""
import json
import math
import re
import struct
import sys
from pathlib import Path


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


source = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
expected = int(sys.argv[2])
records = []
for line in source.splitlines():
    if "[GOLDEN-LEDGER VERIFIED]" not in line:
        continue
    values = {
        name: struct.unpack(">d", bytes.fromhex(raw))[0]
        for name, raw in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)=[^ /]+/0x([0-9a-f]{16})", line)
    }
    assert all(math.isfinite(v) for v in values.values()), line
    is_long = "long=Some(true)" in line
    assert is_long or "long=Some(false)" in line, line
    q, pe, px = values["q"], values["Pe"], values["Px"]
    assert q > 0 and pe > 0 and px > 0
    through_quote = values["bid"] >= values["TP_reconstructed"] if is_long else values["ask"] <= values["TP_reconstructed"]
    mid = (values["bid"] + values["ask"]) / 2.0
    tp_fraction = abs((values["TP"] - pe) / pe)
    pnl_fraction = (mid - pe) / pe if is_long else (pe - mid) / pe
    assert through_quote == (pnl_fraction >= tp_fraction), line
    rate_e, rate_x = 0.0005, 0.0002 if through_quote else 0.0005
    assert bits(values["rE"]) == bits(rate_e)
    assert bits(values["rX"]) == bits(rate_x)
    fee_e = (q * pe) * rate_e
    gross = (px - pe) * q if is_long else (pe - px) * q
    fee_x = (q * px) * rate_x
    realized = gross - fee_x
    computed = {
        "Fe": fee_e,
        "G": gross,
        "Fx": fee_x,
        "net": realized - fee_e,
        "C1": values["C0"] - fee_e,
        "C2": (values["C0"] - fee_e) + realized,
    }
    for name, value in computed.items():
        assert bits(values[name]) == bits(value), (name, values[name], value, line)
    assert bits(computed["C2"]) == expected, line
    legacy_mid_exit_fee = (q * mid) * rate_x
    legacy_mid_cash = (values["C0"] - fee_e) + (gross - legacy_mid_exit_fee)
    records.append({"cashBits": expected, "cash": computed["C2"], "quantity": q, "entryPrice": pe, "exitPrice": px, "entryFee": fee_e, "exitFee": fee_x, "gross": gross, "net": computed["net"], "makerExit": through_quote, "counterfactualMidExitCashBits": bits(legacy_mid_cash), "counterfactualMatchesLegacyGolden": bits(legacy_mid_cash) == 4652010048691470773})
assert len(records) == 2, f"Expected two independently replayed GOLDEN cycles, found {len(records)}"
assert records[0] == records[1], records
print(json.dumps({"status": "success", "witnesses": len(records), "binary64FormulaChecks": records}, indent=2))
