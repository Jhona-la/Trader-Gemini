import sqlite3
import pandas as pd
from utils.logger import logger
from data.database import DatabaseHandler

class TradeHistorian:
    """
    Trade Historian (Omniscient Forensic Auditor)
    Retrieves the full lifecycle of a trade by unifying data from:
    - thoughts (why did we open?)
    - exit_decisions (why did we close?)
    - trade_lifecycle (what happened in between? MAE/MFE)
    - trades (final PnL)
    
    Generates human-readable post-mortem autopsies.
    """
    def __init__(self, db_path: str = "trader_gemini.db"):
        self.db = DatabaseHandler(db_path)

    def generate_autopsy(self, trade_id: str) -> str:
        """Generates a full autopsy report for a specific trade."""
        conn = self.db.get_connection()
        if not conn: return "Database connection failed."

        try:
            # 1. Get Trade Details
            trade = pd.read_sql_query("SELECT * FROM trades WHERE trade_id = ?", conn, params=(trade_id,))
            if trade.empty:
                return f"No trade found with ID: {trade_id}"
            trade = trade.iloc[0]

            # 2. Get Thought (Entry context)
            thought = pd.read_sql_query("SELECT * FROM thoughts WHERE trade_id = ?", conn, params=(trade_id,))
            entry_context = "No entry thought recorded."
            if not thought.empty:
                t = thought.iloc[0]
                entry_context = f"Opened by {t['strategy_id']} ({t['horizon']}) for {t['direction']}. Market state: {t['market_state']}."

            # 3. Get Exit Decisions
            exits = pd.read_sql_query("SELECT * FROM exit_decisions WHERE trade_id = ?", conn, params=(trade_id,))
            exit_context = "No centralized exit decision found. (Hard stop or legacy exit?)"
            if not exits.empty:
                # Get the final approved exit
                approved = exits[exits['oracle_verdict'] == 'APPROVED']
                if not approved.empty:
                    e = approved.iloc[0]
                    exit_context = f"Exit proposed by {e['proposing_strategy']}. Reason: {e['exit_reason']}. Oracle approved at PnL {e['pnl_at_decision']*100:.2f}%."
                else:
                    # Look at denials
                    denials = exits[exits['oracle_verdict'] == 'DENIED']
                    if not denials.empty:
                        exit_context = f"Oracle denied {len(denials)} exit proposals before final closure."

            # 4. Get Lifecycle (MFE/MAE)
            lifecycle = pd.read_sql_query("SELECT * FROM trade_lifecycle WHERE trade_id = ?", conn, params=(trade_id,))
            mfe = lifecycle['mfe'].max() if not lifecycle.empty else 0.0
            mae = lifecycle['mae'].min() if not lifecycle.empty else 0.0

            # 5. Format Report
            report = f"""
            ==================================================
            🔍 OMNISCIENT AUTOPSY REPORT
            Trade ID : {trade_id}
            Symbol   : {trade['symbol']}
            Side     : {trade['side']}
            Final PnL: {trade['pnl']:.4f} USD
            MFE (Max Favorable)  : {mfe*100:.2f}%
            MAE (Max Adverse)    : {mae*100:.2f}%
            ==================================================
            🤔 ENTRY:
            {entry_context}
            
            🚪 EXIT:
            {exit_context}
            
            💡 HISTORIAN CONCLUSION:
            """
            
            # Simple AI heuristic for conclusion
            if trade['pnl'] < 0:
                if mfe > 0.01:
                    report += "We had over 1% profit but gave it all back. Exit Oracle was too greedy or trailing stop was too loose.\n"
                elif mae < -0.02:
                    report += "Trade went immediately against us. Entry thesis was wrong or market regime shifted instantly.\n"
                else:
                    report += "Slow bleed. We lost money gradually. Alpha decay was likely the culprit.\n"
            else:
                report += "Profitable trade. Good execution.\n"

            return report

        except Exception as e:
            logger.error(f"Error generating autopsy for {trade_id}: {e}")
            return f"Error generating autopsy: {e}"

    def audit_losing_streaks(self, limit=10):
        """Audits the last N losing trades to find patterns."""
        conn = self.db.get_connection()
        if not conn: return
        
        try:
            losers = pd.read_sql_query(
                "SELECT trade_id FROM trades WHERE pnl < 0 ORDER BY timestamp DESC LIMIT ?", 
                conn, params=(limit,)
            )
            
            if losers.empty:
                logger.info("No losing trades found to audit.")
                return
                
            for _, row in losers.iterrows():
                print(self.generate_autopsy(row['trade_id']))
                
        except Exception as e:
            logger.error(f"Error auditing losing streaks: {e}")

if __name__ == "__main__":
    import sys
    historian = TradeHistorian()
    if len(sys.argv) > 1:
        print(historian.generate_autopsy(sys.argv[1]))
    else:
        historian.audit_losing_streaks()
