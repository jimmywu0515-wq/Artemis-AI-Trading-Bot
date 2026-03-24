document.addEventListener('DOMContentLoaded', () => {
    
    console.log("Artemis AI: Frontend Loaded.");

    // Elements
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const marketSentimentSpan = document.getElementById('market-sentiment');
    const systemStatusSpan = document.getElementById('system-status');
    const dashPrice = document.getElementById('dash-price');
    const dashSpread = document.getElementById('dash-spread');
    const btnTrain = document.getElementById('btn-train');
    const btnEval = document.getElementById('btn-eval');
    const symbolSelect = document.getElementById('symbol-select');
    const bufferSlider = document.getElementById('buffer-slider');
    const bufferValLabel = document.getElementById('buffer-val');
    const toggleMaInput = document.getElementById('toggle-ma');
    const sensitivityContainer = document.getElementById('sensitivity-container');

    // Chart State
    let mainChart, candleSeries, ma5Series, ma10Series;
    let netWorthChart, rlSeries, staticSeries, maWorthSeries;

    function initCharts() {
        console.log("Artemis AI: Initializing charts...");
        const ohlcvContainer = document.getElementById('ohlcv-chart');
        const perfContainer = document.getElementById('performance-chart');

        if (typeof LightweightCharts === 'undefined') {
            console.error("Artemis AI: LightweightCharts library failed to load.");
            const loadingMsg = document.getElementById('chart-loading');
            if (loadingMsg) loadingMsg.innerHTML = '<p class="text-red-400 font-bold uppercase tracking-widest"><i class="fa-solid fa-triangle-exclamation mr-2"></i> Chart Library Blocked / Offline</p>';
            return false;
        }

        try {
            // Main Candlestick Chart
            mainChart = LightweightCharts.createChart(ohlcvContainer, {
                layout: { backgroundColor: '#0f172a', textColor: '#94a3b8' },
                grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: '#1e293b' },
                timeScale: { borderColor: '#1e293b', timeVisible: true },
            });
            candleSeries = mainChart.addCandlestickSeries({
                upColor: '#10b981', downColor: '#ef4444', borderVisible: false,
                wickUpColor: '#10b981', wickDownColor: '#ef4444'
            });

            // Add MA Series to Main Chart
            ma5Series = mainChart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, title: '5MA', visible: toggleMaInput ? toggleMaInput.checked : true });
            ma10Series = mainChart.addLineSeries({ color: '#3b82f6', lineWidth: 1.5, title: '10MA', visible: toggleMaInput ? toggleMaInput.checked : true });

            // Performance Chart (Lines)
            netWorthChart = LightweightCharts.createChart(perfContainer, {
                layout: { backgroundColor: 'transparent', textColor: '#64748b' },
                grid: { vertLines: { visible: false }, horzLines: { color: '#1e293b' } },
                rightPriceScale: { borderColor: '#1e293b' },
                timeScale: { borderColor: '#1e293b', timeVisible: true },
            });
            rlSeries = netWorthChart.addLineSeries({ color: '#8b5cf6', lineWidth: 2, title: 'RL Bot' });
            staticSeries = netWorthChart.addLineSeries({ color: '#64748b', lineWidth: 1, title: 'Static' });
            maWorthSeries = netWorthChart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, title: 'MA Cross' });

            // Handle Resize
            const resizeObserver = new ResizeObserver(entries => {
                if (entries.length === 0 || !entries[0].contentRect) return;
                mainChart.applyOptions({ width: ohlcvContainer.clientWidth, height: ohlcvContainer.clientHeight });
                netWorthChart.applyOptions({ width: perfContainer.clientWidth, height: perfContainer.clientHeight });
            });
            resizeObserver.observe(ohlcvContainer);
            resizeObserver.observe(perfContainer);

            // UI Event Listeners
            if (toggleMaInput) {
                toggleMaInput.addEventListener('change', () => {
                    ma5Series.applyOptions({ visible: toggleMaInput.checked });
                    ma10Series.applyOptions({ visible: toggleMaInput.checked });
                });
            }

            if (bufferSlider) {
                bufferSlider.addEventListener('input', () => {
                    if (bufferValLabel) bufferValLabel.innerText = `${parseFloat(bufferSlider.value).toFixed(1)}%`;
                });
            }
            return true;
        } catch (err) {
            console.error("Artemis AI: Chart initialization failed:", err);
            return false;
        }
    }

    // Render Sensitivity Bar Chart
    function renderSensitivity(data, rlFinalWorth) {
        if (!sensitivityContainer) return;
        sensitivityContainer.innerHTML = '';
        if (!data || data.length === 0) return;
        
        const minVal = 10000;
        const maxVal = Math.max(...data.map(d => d.net_worth), rlFinalWorth, 10001);
        const range = maxVal - minVal;

        // RL Benchmark Line
        const rlHeightPct = ((rlFinalWorth - minVal) / range) * 100;
        const rlLine = document.createElement('div');
        rlLine.className = "absolute left-0 right-0 border-t-2 border-dashed border-purple-500/50 z-10 pointer-events-none";
        rlLine.style.bottom = `${Math.max(5, rlHeightPct)}%`;
        rlLine.innerHTML = `<span class="bg-purple-600/80 text-[8px] text-white px-1 rounded absolute right-2 -top-3">RL BEST</span>`;
        sensitivityContainer.appendChild(rlLine);

        // Create Bars
        data.forEach(item => {
            const barWrapper = document.createElement('div');
            barWrapper.className = "flex-1 flex flex-col items-center group relative h-full justify-end";
            
            const heightPct = Math.max(((item.net_worth - minVal) / range) * 100, 5);
            
            const bar = document.createElement('div');
            const isBest = item.net_worth >= Math.max(...data.map(d => d.net_worth));
            bar.className = `w-full rounded-t-sm transition-all duration-500 ${isBest ? 'bg-orange-500 shadow-[0_0_10px_rgba(249,115,22,0.4)]' : 'bg-slate-700 hover:bg-slate-600'}`;
            bar.style.height = `${heightPct}%`;
            
            const tooltip = document.createElement('div');
            tooltip.className = "absolute -top-10 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800 text-[10px] p-2 rounded border border-white/10 z-20 whitespace-nowrap shadow-xl";
            tooltip.innerHTML = `Buffer: ${item.buffer}%<br>Profit: $${(item.net_worth - 10000).toFixed(2)}`;
            
            barWrapper.appendChild(tooltip);
            barWrapper.appendChild(bar);
            sensitivityContainer.appendChild(barWrapper);
        });
    }

    // Append Message to Chat UI
    function appendMessage(text, isUser=false) {
        if (!chatMessages) return;
        const div = document.createElement('div');
        div.className = `chat-bubble ${isUser ? 'user-msg text-white' : 'ai-msg text-white'}`;
        const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        div.innerHTML = formattedText;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return div;
    }

    // Chat Form Submit
    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const msg = chatInput.value.trim();
            if (!msg) return;
            appendMessage(msg, true);
            chatInput.value = '';
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                if (data.reply) {
                    let replyText = data.reply;
                    
                    // Parse Commands from AI response
                    const commandRegex = /\[COMMAND: (.*?)\]/g;
                    let match;
                    let hasCommand = false;
                    while ((match = commandRegex.exec(replyText)) !== null) {
                        try {
                            const cmd = JSON.parse(match[1]);
                            console.log("Artemis AI: Executing Chat Command:", cmd);
                            executeChatCommand(cmd);
                            hasCommand = true;
                        } catch(e) { console.error("Command parse error", e); }
                    }
                    
                    // Clean text (remove tags)
                    replyText = replyText.replace(commandRegex, '').trim();
                    appendMessage(replyText, false);
                    
                    if (hasCommand && btnEval) {
                        setTimeout(() => btnEval.click(), 500);
                    }
                }
            } catch (error) {
                appendMessage("Error: Could not connect to API server.", false);
            }
        });
    }

    function executeChatCommand(cmd) {
        if (cmd.type === 'buffer') {
            const val = parseFloat(cmd.value);
            if (bufferSlider) {
                bufferSlider.value = val;
                if (bufferValLabel) bufferValLabel.innerText = `${val.toFixed(1)}%`;
            }
        } else if (cmd.type === 'toggle_ma') {
            if (toggleMaInput) {
                toggleMaInput.checked = cmd.value;
                // Trigger visibility manually as it's not a native event for programmatic changes
                if (ma5Series) ma5Series.applyOptions({ visible: cmd.value });
                if (ma10Series) ma10Series.applyOptions({ visible: cmd.value });
            }
        }
    }

    // Interaction Handlers
    if (btnTrain) {
        btnTrain.addEventListener('click', async () => {
            try {
                const symbol = symbolSelect ? symbolSelect.value : "BTC/USDT";
                const res = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        dsn: "postgresql://postgres:postgres@db:5432/crypto",
                        symbol: symbol
                    })
                });
                const data = await res.json();
                alert(data.message);
            } catch (e) {
                alert("Failed to start training session.");
            }
        });
    }

    if (btnEval) {
        btnEval.addEventListener('click', async () => {
            console.log("Artemis AI: Starting evaluation...");
            const symbol = symbolSelect ? symbolSelect.value : "BTC/USDT";
            const bufferPct = bufferSlider ? (parseFloat(bufferSlider.value) / 100.0) : 0.01;
            const loading = document.getElementById('chart-loading');
            if (loading) loading.classList.remove('hidden');
            
            try {
                const url = `/evaluate?dsn=postgresql://postgres:postgres@db:5432/crypto&symbol=${encodeURIComponent(symbol)}&buffer_pct=${bufferPct}`;
                console.log("Artemis AI: Fetching evaluation data from", url);
                const res = await fetch(url);
                const data = await res.json();
                if (loading) loading.classList.add('hidden');
                
                if (data.error) {
                    console.error("Artemis AI: Evaluation Error:", data.error);
                    alert("Evaluation Error: " + data.error);
                    return;
                }

                console.log("Artemis AI: Evaluation data received.", data.ohlcv.length, "points");

                if (candleSeries) candleSeries.setData(data.ohlcv);

                if (ma5Series && data.ma5 && data.ma5.length > 0) {
                    const ma5Data = data.ma5.map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
                    ma5Series.setData(ma5Data);
                }
                if (ma10Series && data.ma10 && data.ma10.length > 0) {
                    const ma10Data = data.ma10.map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
                    ma10Series.setData(ma10Data);
                }
                
                const rlMarkers = (data.rl_trades || []).map(t => ({
                    time: t.time,
                    position: t.type === 'buy' ? 'belowBar' : 'aboveBar',
                    color: t.type === 'buy' ? '#10b981' : '#f43f5e',
                    shape: t.type === 'buy' ? 'arrowUp' : 'arrowDown',
                    text: 'RL ' + t.type.toUpperCase()
                }));
                
                const maMarkers = (data.ma_trades || []).map(t => ({
                    time: t.time,
                    position: t.type === 'buy' ? 'belowBar' : 'aboveBar',
                    color: t.type === 'buy' ? '#f59e0b' : '#fb923c',
                    shape: t.type === 'buy' ? 'arrowUp' : 'arrowDown',
                    text: 'MA ' + t.type.toUpperCase(),
                    size: 0.5
                }));

                if (candleSeries) candleSeries.setMarkers([...rlMarkers, ...maMarkers].sort((a,b) => a.time - b.time));

                const rlData = (data.rl_net_worth || []).map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
                const staticData = (data.static_net_worth || []).map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
                const maWorthData = (data.ma_net_worth || []).map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
                
                if (rlSeries) rlSeries.setData(rlData);
                if (staticSeries) staticSeries.setData(staticData);
                if (maWorthSeries) maWorthSeries.setData(maWorthData);

                if (mainChart) mainChart.timeScale().fitContent();
                if (netWorthChart) netWorthChart.timeScale().fitContent();

                const rlFinalWorth = (data.rl_net_worth && data.rl_net_worth.length > 0) ? data.rl_net_worth[data.rl_net_worth.length - 1] : 10000;
                renderSensitivity(data.sensitivity, rlFinalWorth);

            } catch (e) {
                if (loading) loading.classList.add('hidden');
                console.error("Artemis AI: Evaluation request failed:", e);
                alert("Evaluation request failed. Check server logs.");
            }
        });
    }

    // Polling Logic
    async function pollStatus() {
        try {
            const res = await fetch('/status');
            if (res.ok) {
                const data = await res.json();
                if (systemStatusSpan) {
                    systemStatusSpan.innerHTML = `
                        <span class="w-2 h-2 rounded-full ${data.is_training ? 'bg-blue-400 pulse-anim' : 'bg-green-400'} mr-2"></span>
                        Agent: ${data.status.toUpperCase()}
                    `;
                }
                if (dashPrice) dashPrice.innerText = `${(data.current_price || 0).toFixed(2)} USDT`;
                if (dashSpread) dashSpread.innerText = `${((data.grid_width_pct || 0) * 100).toFixed(2)}%`;
            } else {
                console.warn("Artemis AI: Status poll returned non-ok status:", res.status);
            }
        } catch (e) {
            console.error("Artemis AI: Status poll failed:", e);
        }
    }

    async function pollSentiment() {
        try {
            const res = await fetch('/sentiment');
            if (res.ok) {
                const data = await res.json();
                const score = data.score || 0;
                let color = "text-gray-400", label = "Neutral";
                if (score > 0.2) { color = "text-green-400"; label = "Bullish"; }
                else if (score < -0.2) { color = "text-red-400"; label = "Bearish"; }
                if (marketSentimentSpan) {
                    marketSentimentSpan.innerHTML = `<i class="fa-solid fa-cloud mr-2 ${color}"></i> Sentiment: ${label} (${score.toFixed(2)})`;
                }
            }
        } catch (e) {
            console.error("Artemis AI: Sentiment poll failed:", e);
        }
    }

    // Initial Trigger
    initCharts();
    
    // Polling - Always start these even if charts fail
    setInterval(pollStatus, 3000);
    setInterval(pollSentiment, 30000);
    pollStatus();
    pollSentiment();
    
    // Initial evaluation
    if (btnEval) setTimeout(() => btnEval.click(), 1000);
});
