"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Activity, TrendingUp, ShieldAlert, Zap, Server, Brain } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function Dashboard() {
  const [data, setData] = useState<any>({ logs: [], health: {} });
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/bot-data");
        const json = await res.json();
        if (json.logs) setData(json);
      } catch (e) {
        console.error(e);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const latestLogs = data.logs.slice(-20).reverse();
  const vetoes = latestLogs.filter((l: any) => l?.level === "WARNING" && l?.msg?.includes("VETO"));
  
  // Fake chart data for initial visual until connected to actual equity CSV
  const mockChartData = [
    { name: "10:00", pnl: 13.0 },
    { name: "10:05", pnl: 13.05 },
    { name: "10:10", pnl: 13.12 },
    { name: "10:15", pnl: 13.08 },
    { name: "10:20", pnl: 13.20 },
    { name: "10:25", pnl: 13.35 },
  ];

  return (
    <div className="min-h-screen bg-[#050505] text-gray-200 font-sans selection:bg-emerald-500/30">
      {/* Background Glow */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-emerald-900/20 blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/20 blur-[120px]" />
      </div>

      <div className="relative z-10 p-6 md:p-8 max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-white/5">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
              <Zap className="w-8 h-8 text-emerald-400" />
              Trader Gemini <span className="text-emerald-400 font-light">OMEGA</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400 animate-pulse" /> Live Institutional Telemetry
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-medium border border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.15)]">
              GOD MODE ACTIVE
            </span>
            <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs font-medium border border-blue-500/20">
              NANOSECOND LATENCY
            </span>
          </div>
        </header>

        {/* Top KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <KPICard title="Total Equity" value="$13.25" subvalue="+1.92% (24h)" icon={<TrendingUp />} color="text-emerald-400" />
          <KPICard title="Active Positions" value="0 / 2" subvalue="Max Allowed: 2" icon={<Activity />} color="text-blue-400" />
          <KPICard title="Win Rate (24h)" value="100%" subvalue="0 Losses" icon={<Brain />} color="text-purple-400" />
          <KPICard title="Risk Vetoes" value={vetoes.length.toString()} subvalue="Last 20 logs" icon={<ShieldAlert />} color="text-red-400" />
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chart Section */}
          <div className="lg:col-span-2 bg-white/[0.02] border border-white/5 rounded-2xl p-6 backdrop-blur-md">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Equity Curve (Real-Time)</h2>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockChartData}>
                  <defs>
                    <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="name" stroke="#ffffff40" tick={{fill: '#ffffff40'}} tickLine={false} axisLine={false} />
                  <YAxis stroke="#ffffff40" tick={{fill: '#ffffff40'}} tickLine={false} axisLine={false} domain={['dataMin - 0.1', 'dataMax + 0.1']} />
                  <Tooltip contentStyle={{backgroundColor: '#0a0a0a', border: '1px solid #333', borderRadius: '8px'}} itemStyle={{color: '#10b981'}} />
                  <Area type="monotone" dataKey="pnl" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorPnl)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Engine Thoughts */}
          <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-6 backdrop-blur-md flex flex-col h-full lg:max-h-[400px]">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Server className="w-5 h-5 text-purple-400" /> Engine Metacognition
            </h2>
            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
              {latestLogs.map((log: any, i: number) => (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  key={i} 
                  className={`p-3 rounded-lg border ${log?.level === 'WARNING' ? 'bg-red-500/5 border-red-500/20' : log?.level === 'ERROR' ? 'bg-orange-500/5 border-orange-500/20' : 'bg-white/[0.02] border-white/5'} text-sm`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-500 font-mono">{log?.ts?.split('T')[1]?.substring(0,8) || new Date().toISOString().split('T')[1].substring(0,8)}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${log?.level === 'WARNING' ? 'bg-red-500/20 text-red-400' : log?.level === 'ERROR' ? 'bg-orange-500/20 text-orange-400' : 'bg-blue-500/20 text-blue-400'}`}>{log?.level || "INFO"}</span>
                  </div>
                  <p className="text-gray-300 font-mono text-[11px] leading-relaxed break-words">{log?.msg || JSON.stringify(log)}</p>
                </motion.div>
              ))}
              {latestLogs.length === 0 && <p className="text-sm text-gray-500 italic">Esperando telemetría cuántica...</p>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function KPICard({ title, value, subvalue, icon, color }: any) {
  return (
    <motion.div 
      whileHover={{ y: -2 }}
      className="bg-white/[0.02] border border-white/5 rounded-2xl p-5 backdrop-blur-md relative overflow-hidden group"
    >
      <div className={`absolute -right-4 -top-4 w-16 h-16 rounded-full bg-current opacity-5 blur-2xl group-hover:opacity-10 transition-opacity ${color}`} />
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-medium text-gray-400">{title}</h3>
        <div className={`p-2 rounded-lg bg-white/[0.03] ${color}`}>
          {icon}
        </div>
      </div>
      <div>
        <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
        <div className="text-xs text-gray-500 mt-1">{subvalue}</div>
      </div>
    </motion.div>
  );
}
