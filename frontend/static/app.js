document.addEventListener('DOMContentLoaded', () => {
    
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

    // Chart State
    let mainChart, candleSeries, netWorthChart, rlSeries, staticSeries;

    function initCharts() {
        const ohlcvContainer = document.getElementById('ohlcv-chart');
        const perfContainer = document.getElementById('performance-chart');

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

        // Performance Chart (Lines)
        netWorthChart = LightweightCharts.createChart(perfContainer, {
            layout: { backgroundColor: 'transparent', textColor: '#64748b' },
            grid: { vertLines: { visible: false }, horzLines: { color: '#1e293b' } },
            rightPriceScale: { borderColor: '#1e293b' },
            timeScale: { borderColor: '#1e293b', timeVisible: true },
        });
        rlSeries = netWorthChart.addLineSeries({ color: '#8b5cf6', lineWidth: 2, title: 'RL Bot' });
        staticSeries = netWorthChart.addLineSeries({ color: '#64748b', lineWidth: 1, title: 'Static' });

        // Handle Resize
        const resizeObserver = new ResizeObserver(entries => {
            if (entries.length === 0 || !entries[0].contentRect) return;
            mainChart.applyOptions({ width: ohlcvContainer.clientWidth, height: ohlcvContainer.clientHeight });
            netWorthChart.applyOptions({ width: perfContainer.clientWidth, height: perfContainer.clientHeight });
        });
        resizeObserver.observe(ohlcvContainer);
        resizeObserver.observe(perfContainer);
    }

    // Append Message to Chat UI
    function appendMessage(text, isUser=false) {
        const div = document.createElement('div');
        div.className = `chat-bubble ${isUser ? 'user-msg text-white' : 'ai-msg text-white'}`;
        const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        div.innerHTML = formattedText;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return div;
    }

    // Chat Form Submit
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
            if (data.reply) appendMessage(data.reply, false);
        } catch (error) {
            appendMessage("Error: Could not connect to API server.", false);
        }
    });

    // Interaction Handlers
    btnTrain.addEventListener('click', async () => {
        try {
            const symbol = symbolSelect.value;
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

    btnEval.addEventListener('click', async () => {
        const symbol = symbolSelect.value;
        const loading = document.getElementById('chart-loading');
        loading.classList.remove('hidden');
        
        try {
            const res = await fetch(`/evaluate?dsn=postgresql://postgres:postgres@db:5432/crypto&symbol=${encodeURIComponent(symbol)}`);
            const data = await res.json();
            loading.classList.add('hidden');
            
            if (data.error) {
                alert("Evaluation Error: " + data.error);
                return;
            }

            // Update Candlesticks
            candleSeries.setData(data.ohlcv);
            
            // Add Trade Markers
            const markers = data.trades.map(t => ({
                time: t.time,
                position: t.type === 'buy' ? 'belowBar' : 'aboveBar',
                color: t.type === 'buy' ? '#10b981' : '#f43f5e',
                shape: t.type === 'buy' ? 'arrowUp' : 'arrowDown',
                text: t.type.toUpperCase() + ' @ ' + t.price.toFixed(2)
            }));
            candleSeries.setMarkers(markers);

            // Update Performance Lines
            const rlData = data.rl_net_worth.map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
            const staticData = data.static_net_worth.map((val, i) => ({ time: data.ohlcv[i].time, value: val }));
            rlSeries.setData(rlData);
            staticSeries.setData(staticData);

            // Fit content
            mainChart.timeScale().fitContent();
            netWorthChart.timeScale().fitContent();

        } catch (e) {
            loading.classList.add('hidden');
            console.error(e);
            alert("Evaluation request failed.");
        }
    });

    // Polling Logic
    async function pollStatus() {
        try {
            const res = await fetch('/status');
            if (res.ok) {
                const data = await res.json();
                systemStatusSpan.innerHTML = `
                    <span class="w-2 h-2 rounded-full ${data.is_training ? 'bg-blue-400 pulse-anim' : 'bg-green-400'} mr-2"></span>
                    Agent: ${data.status.toUpperCase()}
                `;
                dashPrice.innerText = `${data.current_price.toFixed(2)} USDT`;
                dashSpread.innerText = `${(data.grid_width_pct * 100).toFixed(2)}%`;
            }
        } catch (e) {}
    }

    async function pollSentiment() {
        try {
            const res = await fetch('/sentiment');
            if (res.ok) {
                const data = await res.json();
                const score = data.score;
                let color = "text-gray-400", label = "Neutral";
                if (score > 0.2) { color = "text-green-400"; label = "Bullish"; }
                else if (score < -0.2) { color = "text-red-400"; label = "Bearish"; }
                marketSentimentSpan.innerHTML = `<i class="fa-solid fa-cloud mr-2 ${color}"></i> Sentiment: ${label} (${score.toFixed(2)})`;
            }
        } catch (e) {}
    }

    // Initial Trigger
    initCharts();
    setInterval(pollStatus, 3000);
    setInterval(pollSentiment, 30000);
    pollStatus();
    pollSentiment();
    
    // Trigger initial evaluation to fill chart
    setTimeout(() => btnEval.click(), 1000);
});
