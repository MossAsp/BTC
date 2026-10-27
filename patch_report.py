import sys
path = "generate_report.py"
s = open(path, encoding="utf-8-sig").read()
if "Buy & Hold ให้ผลปรับความเสี่ยงดีกว่า" in s:
    print("ไฟล์นี้แก้แล้ว ไม่ต้องทำซ้ำ"); sys.exit(0)
start = s.find("    if best_rule and float(rules.loc[best_rule")
end = s.find("    if ml_ci[0] <= 0 <= ml_ci[1]:")
if start < 0 or end < start:
    sys.exit("หาจุดที่ต้องแก้ไม่เจอ - ส่งไฟล์มาให้ดูได้")
new = '''    ml_sh, bh_sh = float(ml["sharpe_ratio"]), float(bh["sharpe_ratio"])
    if best_rule:
        rule_sh = float(rules.loc[best_rule, "sharpe_ratio"])
        if rule_sh > ml_sh + 0.05:
            notes.append(f"**ML Ensemble ไม่ใช่กลยุทธ์ที่ Sharpe ดีที่สุด** — {best_rule} ให้ Sharpe {rule_sh:.2f} สูงกว่า ML ({ml_sh:.2f}) ควรรายงานตามจริง")
        elif abs(rule_sh - ml_sh) <= 0.05:
            notes.append(f"Sharpe ของ ML ({ml_sh:.2f}) ใกล้เคียง {best_rule} ({rule_sh:.2f}) มากจนแยกไม่ออกทางสถิติ — "
                         "ความซับซ้อนของโมเดลไม่ได้สร้างความได้เปรียบเหนือ baseline ง่ายๆ")
    if bh_sh > ml_sh:
        notes.append(f"**Buy & Hold ให้ผลปรับความเสี่ยงดีกว่า ML**: Sharpe {bh_sh:.2f} เทียบ {ml_sh:.2f}, "
                     f"Calmar {bh['calmar_ratio']:.2f} เทียบ {ml['calmar_ratio']:.2f} (Sharpe ไม่ขึ้นกับสัดส่วนเงินลงทุน จึงไม่ใช่แค่เพราะ ML ถือสถานะ {ml['exposure_pct']:.0f}% ของเวลา) "
                     f"ข้อดีที่เหลือของ ML คือ Max Drawdown ต่ำกว่า ({ml['max_drawdown_pct']:.1f}% เทียบ {bh['max_drawdown_pct']:.1f}%) ส่วนผลตอบแทนรวมคือ "
                     f"{ml['total_return_pct']:.1f}% เทียบ {bh['total_return_pct']:.1f}% (ถ้า BTC ขาขึ้นแรงในช่วงข้อมูล กลยุทธ์ long-only ที่ถือสถานะบางช่วงจะเสียเปรียบเชิงโครงสร้าง)")
    elif float(ml["total_return_pct"]) < float(bh["total_return_pct"]):
        notes.append(f"ผลตอบแทนรวมของ ML ({ml['total_return_pct']:.1f}%) ต่ำกว่า Buy & Hold ({bh['total_return_pct']:.1f}%) "
                     f"แต่ปรับความเสี่ยงแล้วดีกว่า (Sharpe {ml_sh:.2f} เทียบ {bh_sh:.2f}, Max DD {ml['max_drawdown_pct']:.1f}% เทียบ {bh['max_drawdown_pct']:.1f}%)")
'''
open(path, "w", encoding="utf-8").write(s[:start] + new + s[end:])