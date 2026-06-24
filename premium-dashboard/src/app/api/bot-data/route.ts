import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    // process.cwd() inside premium-dashboard
    const projectRoot = path.join(process.cwd(), '..');
    
    // Find the latest bot log
    const today = new Date().toISOString().slice(0,10).replace(/-/g, '');
    const logPath = path.join(projectRoot, 'logs', `bot_${today}.json`);
    
    let logs: any[] = [];
    if (fs.existsSync(logPath)) {
      const content = fs.readFileSync(logPath, 'utf-8');
      const lines = content.split('\n').filter(Boolean).slice(-100);
      logs = lines.map(l => {
        try { return JSON.parse(l); } catch(e) { return null; }
      }).filter(Boolean);
    }

    const healthPath = path.join(projectRoot, 'dashboard', 'data', 'futures', 'system_health.json');
    let health = {};
    if (fs.existsSync(healthPath)) {
      try {
        health = JSON.parse(fs.readFileSync(healthPath, 'utf-8'));
      } catch(e) {}
    }

    return NextResponse.json({ logs, health });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
