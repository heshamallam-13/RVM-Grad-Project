// ==============================================
// EcoVend RVM — Client-side Socket.IO + FPS Chart
// ==============================================

const socket = io();

// DOM refs
const videoFeed   = document.getElementById("video-feed");
const fpsBadge    = document.getElementById("fps-badge");
const valPoints   = document.getElementById("val-points");
const valPet      = document.getElementById("val-pet");
const valCan      = document.getElementById("val-can");
const valDetected = document.getElementById("val-detected");
const toastEl     = document.getElementById("toast");

// ---- FPS Chart (Chart.js) ----
const MAX_FPS_POINTS = 60;
const fpsData = [];
const fpsLabels = [];

const fpsChart = new Chart(document.getElementById("fps-chart"), {
    type: "line",
    data: {
        labels: fpsLabels,
        datasets: [{
            label: "FPS",
            data: fpsData,
            borderColor: "#00b894",
            backgroundColor: "rgba(0,184,148,0.08)",
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.35,
        }],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: { display: false },
            y: {
                beginAtZero: true,
                suggestedMax: 30,
                ticks: { color: "#8891a5", font: { size: 11 } },
                grid: { color: "rgba(255,255,255,0.04)" },
            },
        },
        plugins: {
            legend: { display: false },
        },
    },
});

// ---- Socket events ----

socket.on("frame", (data) => {
    videoFeed.src = "data:image/jpeg;base64," + data.img;
});

socket.on("stats", (s) => {
    fpsBadge.textContent = s.fps + " FPS";
    valPoints.textContent = s.points;
    valPet.textContent = s.pet;
    valCan.textContent = s.can;

    // Detected type display
    if (s.type === "pet") {
        valDetected.textContent = "PET " + (s.conf * 100).toFixed(0) + "%";
        valDetected.className = "stat-value det-pet";
    } else if (s.type === "can") {
        valDetected.textContent = "CAN " + (s.conf * 100).toFixed(0) + "%";
        valDetected.className = "stat-value det-can";
    } else {
        valDetected.textContent = "—";
        valDetected.className = "stat-value det-none";
    }

    // Update FPS chart
    fpsLabels.push("");
    fpsData.push(s.fps);
    if (fpsData.length > MAX_FPS_POINTS) {
        fpsLabels.shift();
        fpsData.shift();
    }
    fpsChart.update();
});

socket.on("state", (s) => {
    valPoints.textContent = s.total_points;
    valPet.textContent = s.pet_count;
    valCan.textContent = s.can_count;
});

// ---- Toast ----
let toastTimer = null;
socket.on("toast", (data) => {
    showToast(data.msg);
});

function showToast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.remove("hidden");
    toastEl.classList.add("show");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toastEl.classList.remove("show");
        toastEl.classList.add("hidden");
    }, 2500);
}

// ---- Button handlers ----
function nextItem() {
    socket.emit("next_item");
}

function resetSession() {
    socket.emit("reset_session");
}
