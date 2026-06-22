const { spawn } = require('child_process');
const os = require('os');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');

const isWindows = process.platform === 'win32';
const HW_MONITOR_INTERVAL_MS = 1000;

let prevCpuInfo = null;

function getCpuUsagePct() {
    const cpus = os.cpus();
    if (!prevCpuInfo) {
        prevCpuInfo = cpus;
        return 0;
    }
    let totalDelta = 0, idleDelta = 0;
    cpus.forEach((cpu, i) => {
        const prev = prevCpuInfo[i];
        if (!prev) return;
        const prevTotal = Object.values(prev.times).reduce((a, b) => a + b, 0);
        const currTotal = Object.values(cpu.times).reduce((a, b) => a + b, 0);
        totalDelta += currTotal - prevTotal;
        idleDelta += cpu.times.idle - prev.times.idle;
    });
    prevCpuInfo = cpus;
    if (totalDelta === 0) return 0;
    return Math.round((1 - idleDelta / totalDelta) * 100);
}

function terminateChildProcess(child, { force = false } = {}) {
    if (!child || child.pid == null) return;

    if (isWindows) {
        const args = ['/PID', String(child.pid), '/T'];
        if (force) args.splice(2, 0, '/F');
        const killer = spawn('taskkill', args, { stdio: 'ignore', windowsHide: true });
        killer.on('error', () => { });
        return;
    }

    const signal = force ? 'SIGKILL' : 'SIGTERM';
    try {
        process.kill(child.pid, signal);
    } catch (_) {
        try { process.kill(-child.pid, signal); } catch (__) { }
    }
}

function runSingleFlightProbe(state, spawnChild, timeoutMs, parseResult) {
    if (state.pending) return Promise.resolve(null);
    state.pending = true;

    return new Promise((resolve) => {
        const child = spawnChild();
        let stdout = '';
        let settled = false;
        let timedOut = false;
        let forceKillTimer = null;

        const finish = (value) => {
            if (settled) return;
            settled = true;
            state.pending = false;
            clearTimeout(timeout);
            if (forceKillTimer) clearTimeout(forceKillTimer);
            resolve(value);
        };

        const timeout = setTimeout(() => {
            timedOut = true;
            terminateChildProcess(child);
            forceKillTimer = setTimeout(() => terminateChildProcess(child, { force: true }), 1500);
        }, timeoutMs);

        child.stdout.on('data', (data) => {
            stdout += data;
        });

        child.on('close', (code) => {
            if (timedOut || code !== 0) {
                finish(null);
                return;
            }
            try {
                finish(parseResult(stdout));
            } catch (_) {
                finish(null);
            }
        });

        child.on('error', () => {
            finish(null);
        });
    });
}

const cpuTempProbeState = { pending: false };

function getCpuTemp() {
    return runSingleFlightProbe(
        cpuTempProbeState,
        () => {
            if (isWindows) {
                return spawn('powershell', [
                    '-NoProfile', '-NonInteractive', '-Command',
                    'Get-WmiObject -Namespace root/wmi -Class MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature'
                ], { windowsHide: true });
            }

            return spawn('bash', ['-c',
                'paste -sd+ /sys/class/thermal/thermal_zone*/temp 2>/dev/null | bc'
            ], { windowsHide: true });
        },
        4000,
        (stdout) => {
            if (!stdout.trim()) return null;

            if (isWindows) {
                const vals = stdout.trim().split('\n')
                    .map(l => parseFloat(l.trim()))
                    .filter(v => !isNaN(v) && v > 0);
                if (!vals.length) return null;
                return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length / 10 - 273.15);
            }

            const val = parseFloat(stdout.trim());
            return isNaN(val) ? null : Math.round(val / 1000);
        }
    );
}

// GPU stats via Python/torch — the only backend on Windows + ROCm.
const pythonGpuStatsProbeState = { pending: false };

function getGpuStatsPython() {
    return runSingleFlightProbe(
        pythonGpuStatsProbeState,
        () => {
            const ROOT_DIR = path.resolve(__dirname, '..', '..');
            // Read venv_path from global_config.toml if available
            let venvPath = path.join(ROOT_DIR, 'venv');
            const cfgPath = path.join(__dirname, 'global_config.toml');
            try {
                if (fs.existsSync(cfgPath)) {
                    const raw = fs.readFileSync(cfgPath, 'utf8');
                    const m = raw.match(/^venv_path\s*=\s*"([^"]*)"/m);
                    if (m && m[1].trim()) venvPath = m[1].trim();
                }
            } catch (_) { /* use default */ }

            const exe = path.join(venvPath, 'Scripts', 'python.exe');
            const pythonExe = fs.existsSync(exe) ? exe : 'python';

            const script = 'import torch,json;'
                + 'r=[];'
                + "[r.append({'index':i,'name':torch.cuda.get_device_name(i),"
                + "'util':0,"
                + "'memUsed':round(torch.cuda.memory_allocated(i)/1048576),"
                + "'memTotal':round(torch.cuda.get_device_properties(i).total_memory/1048576),"
                + "'temp':0,'powerDraw':0,'powerLimit':0}) "
                + 'for i in range(torch.cuda.device_count())];'
                + 'print(json.dumps(r))';

            return spawn(pythonExe, ['-c', script], { windowsHide: true });
        },
        5000,
        (stdout) => {
            if (!stdout.trim()) return null;
            return JSON.parse(stdout.trim());
        }
    );
}

// wss       - WebSocket.Server instance
// getActiveGpus - () => { [gpuIndex]: 'training'|'sampling' }
function startHardwareMonitor(wss, getActiveGpus) {
    setInterval(async () => {
        if (wss.clients.size === 0) return;

        const cpuPct = getCpuUsagePct();
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        let cpuTemp = await getCpuTemp();

        const gpus = await getGpuStatsPython();
        if (gpus === null) return;

        const activeGpus = getActiveGpus();
        gpus.forEach(gpu => {
            gpu.activity = activeGpus[String(gpu.index)] || null;
        });

        const payload = JSON.stringify({
            type: 'hw_stats',
            data: {
                cpu: cpuPct,
                cpuTemp,
                ram: { total: totalMem, used: totalMem - freeMem },
                gpus
            }
        });

        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(payload);
            }
        });
    }, HW_MONITOR_INTERVAL_MS);
}

module.exports = { startHardwareMonitor };
