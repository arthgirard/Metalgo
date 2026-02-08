// --- GLOBALS ---
let isGlobalOpen = document.body.getAttribute('data-ouvert') === 'true';

// --- ON LOAD ---
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    updateHistory();
    setInterval(checkOpenStatus, 60000); 
});

// --- NAVIGATION ---
function changerOnglet(viewId, button) {
    // Hide all views
    document.querySelectorAll('.tab-content').forEach(div => {
        div.classList.remove('active');
    });

    // Deactivate buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Activate view and button
    document.getElementById(viewId).classList.add('active');
    button.classList.add('active');

    // Update data
    if (viewId === 'view-stats' || viewId === 'view-predict') {
        updateStats();
    }
}

// --- LOGS & ACTIONS ---
function sendLog(type, detail) {
    if (navigator.vibrate) {
        navigator.vibrate(15); 
    }
    
    fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: type, detail: detail })
    })
    .then(response => response.json())
    .then(data => {
        updateStats();
        updateHistory();
    })
    .catch(err => console.error(err));
}

function undoLastAction() {
    fetch('/api/undo', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            updateStats();
            updateHistory();
        } else {
            alert("Rien à annuler !");
        }
    });
}

// --- HISTORY ---
function updateHistory() {
    fetch('/api/history')
    .then(res => res.json())
    .then(data => {
        const container = document.getElementById('history-list');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = '<div style="color:#666; font-style:italic; font-size:0.8rem; text-align:center; padding:10px;">Aucune action récente</div>';
            return;
        }

        data.forEach(item => {
            let icon = item.type === 'VENTE' ? '🧀' : '🔄';
            let label = item.type === 'VENTE' ? 'Vente' : 'Conv.';
            
            const div = document.createElement('div');
            div.className = `history-item type-${item.type}`;
            div.innerHTML = `
                <div class="hist-left">
                    <span>${icon}</span>
                    <span>${label} <strong>${item.detail}</strong></span>
                </div>
                <div class="hist-time">${item.heure}</div>
            `;
            container.appendChild(div);
        });
    });
}

// --- STATS & ANALYSIS ---
function updateStats() {
    fetch('/api/stats')
    .then(res => res.json())
    .then(data => {
        // Sales Counters
        document.getElementById('count-250g').innerText = data.c250 || 0;
        document.getElementById('count-1kg').innerText = data.c1kg || 0;
        document.getElementById('count-2kg').innerText = data.c2kg || 0;

        // Analysis Grid
        if(document.getElementById('stat-peak')) {
            document.getElementById('stat-peak').innerText = data.peak_hour || "--";
            document.getElementById('stat-top').innerText = data.top_format || "--";
            document.getElementById('stat-mass').innerText = data.total_mass || "0 kg";
            document.getElementById('stat-conv').innerText = data.total_conv || "0";
        }
    })
    .catch(err => console.error("Stats error:", err));

    // Predictions
    fetch('/api/prediction')
    .then(res => res.json())
    .then(data => {
        const trendContainer = document.getElementById('trend-container');
        const trendBadge = document.getElementById('trend-badge');
        const infoText = document.getElementById('prediction-info');
        const eventBanner = document.getElementById('event-banner');

        document.getElementById('time-left').innerText = data.heures_restantes !== undefined ? data.heures_restantes : "--";
        document.getElementById('meteo-label').innerText = data.meteo || "--";

        if (data.evenement && eventBanner) {
            eventBanner.style.display = 'block';
            document.getElementById('event-name').innerText = data.evenement;
        } else if (eventBanner) {
            eventBanner.style.display = 'none';
        }

        if (data.previsions) {
            document.getElementById('pred-250g').innerText = data.previsions['250g'] || 0;
            document.getElementById('pred-1kg').innerText = data.previsions['1kg'] || 0;
            document.getElementById('pred-2kg').innerText = data.previsions['2kg'] || 0;
        }

        const debugInfo = data.debug_info || "";
        if(infoText) infoText.innerText = debugInfo;

        if(trendContainer) {
            const match = debugInfo.match(/(\d+)%/);
            if (match) {
                let percent = parseInt(match[1]);
                trendContainer.style.display = 'block';
                trendBadge.className = 'trend-badge';
                
                if (percent > 110) {
                    trendBadge.className += ' rush';
                    trendBadge.innerHTML = `🔥 Rush (${percent}%)`;
                } else if (percent < 90) {
                    trendBadge.className += ' calm';
                    trendBadge.innerHTML = `🧊 Calme (${percent}%)`;
                } else {
                    trendBadge.className += ' normal';
                    trendBadge.innerHTML = `✅ Normal (${percent}%)`;
                }
            } else {
                trendContainer.style.display = 'none'; 
            }
        }
    })
    .catch(err => console.error("Prediction error:", err));
}

// --- FORECAST MANAGEMENT ---
function switchPredict(mode, btn) {
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (mode === 'day') {
        document.getElementById('predict-day-content').style.display = 'block';
        document.getElementById('predict-week-content').style.display = 'none';
    } else {
        document.getElementById('predict-day-content').style.display = 'none';
        document.getElementById('predict-week-content').style.display = 'block';
        loadWeeklyForecast();
    }
}

function loadWeeklyForecast() {
    const container = document.getElementById('week-list');
    const loading = document.getElementById('loading-week');
    
    if(container.innerHTML.trim() !== "") return;

    fetch('/api/forecast_week')
    .then(res => res.json())
    .then(data => {
        loading.style.display = 'none';
        container.innerHTML = '';

        if(data.error || data.length === 0) {
            container.innerHTML = `<div class="info-text">⚠️ Météo indisponible.</div>`;
            return;
        }

        data.forEach(jour => {
            let eventBadge = '';
            if (jour.event) {
                eventBadge = `<span style="
                    background: linear-gradient(90deg, #8e44ad, #c0392b);
                    color: white;
                    font-size: 0.7rem;
                    padding: 2px 8px;
                    border-radius: 6px;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    white-space: nowrap;
                    vertical-align: middle;
                ">
                    ${jour.event}
                </span>`;
            }

            const dateText = jour.date_affichee || jour.date;

            let html = `
                <div class="glass-card" style="padding: 15px; margin-bottom: 15px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <div style="display:flex; align-items:center; gap: 10px;">
                            <span style="font-weight:bold; font-size:1.1rem;">${dateText}</span>
                            ${eventBadge}
                        </div>
                        <div style="font-size:0.85rem; color:#a0a7b8; text-align:right;">${jour.meteo}</div>
                    </div>
            `;

            if (jour.ferme) {
                html += `<div style="text-align:center; color:#e74c3c; font-weight:bold; padding:10px; background:rgba(231, 76, 60, 0.1); border-radius:8px;">FERMÉ</div>`;
            } else {
                html += `
                    <div style="display: flex; justify-content: space-around; align-items: flex-end;">
                        <div style="text-align:center;">
                            <div style="font-size:1.3rem; font-weight:bold; color:var(--text-primary); line-height:1;">${jour.totals['250g']}</div>
                            <div style="font-size:0.7rem; color:var(--text-secondary); margin-top:4px;">250g</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.3rem; font-weight:bold; color:var(--text-primary); line-height:1;">${jour.totals['1kg']}</div>
                            <div style="font-size:0.7rem; color:var(--text-secondary); margin-top:4px;">1 kg</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.3rem; font-weight:bold; color:var(--text-primary); line-height:1;">${jour.totals['2kg']}</div>
                            <div style="font-size:0.7rem; color:var(--text-secondary); margin-top:4px;">2 kg</div>
                        </div>
                    </div>
                `;
            }
            html += `</div>`;
            container.innerHTML += html;
        });
    });
}

function retrainModel() {
    const btn = document.getElementById('btn-retrain');
    const msg = document.getElementById('retrain-msg');
    
    btn.innerHTML = "⚙️ Calcul en cours...";
    btn.classList.add('btn-loading');
    btn.disabled = true;
    msg.style.display = 'none';

    fetch('/api/retrain', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        btn.innerHTML = "✅ Modèle à jour !";
        btn.classList.remove('btn-loading');
        btn.style.background = "var(--accent-sale-start)"; 
        
        msg.innerText = "Les nouvelles données ont été intégrées.";
        msg.style.display = 'block';
        msg.style.color = "#2ecc71";

        setTimeout(() => {
            btn.innerHTML = "🧠 Réentrainer le modèle";
            btn.style.background = ""; 
            btn.disabled = false;
            msg.style.display = 'none';
            updateStats();
        }, 3000);
    })
    .catch(err => {
        console.error(err);
        btn.innerHTML = "❌ Erreur";
        btn.classList.remove('btn-loading');
        btn.disabled = false;
        msg.innerText = "Erreur lors du réentraînement.";
        msg.style.display = 'block';
        msg.style.color = "#e74c3c";
    });
}

function checkOpenStatus() {
    fetch('/api/status')
    .then(res => res.json())
    .then(data => {
        isGlobalOpen = data.ouvert;
        document.body.setAttribute('data-ouvert', isGlobalOpen);
    })
    .catch(err => console.log("Check error", err));
}