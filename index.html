<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AstroVeda | AI & Ancient Wisdom</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #050505; color: #e5e5e5; }
        .gold-glow { box-shadow: 0 0 15px rgba(212, 175, 55, 0.2); }
        .tab-active { border-bottom: 2px solid #d4af37; color: #d4af37; }
        .glass { background: rgba(20, 20, 20, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); }
    </style>
</head>
<body class="min-h-screen">

    <nav class="flex justify-center space-x-8 p-6 glass sticky top-0 z-50">
        <button onclick="showTab('nadi')" class="tab-btn tab-active font-bold">NADI</button>
        <button onclick="showTab('chat')" class="tab-btn font-bold">AI GURU</button>
        <button onclick="showTab('shalaka')" class="tab-btn font-bold">RAM SHALAKA</button>
    </nav>

    <main class="max-w-4xl mx-auto p-6">
        
        <section id="nadi-tab" class="tab-content">
            <div class="glass p-8 rounded-3xl gold-glow">
                <h2 class="text-3xl font-serif text-amber-500 mb-6">Birth Insights</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <input type="text" id="name" placeholder="Name" class="bg-black/50 p-4 rounded-xl border border-gray-800">
                    <input type="date" id="date" class="bg-black/50 p-4 rounded-xl border border-gray-800">
                    <input type="time" id="time" class="bg-black/50 p-4 rounded-xl border border-gray-800">
                    <input type="text" id="location" placeholder="Birth City" class="bg-black/50 p-4 rounded-xl border border-gray-800">
                </div>
                <button onclick="getPrediction()" class="w-full mt-8 bg-amber-600 py-4 rounded-xl font-bold text-black">REVEAL DESTINY</button>
            </div>
            
            <div id="results" class="hidden mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass p-6 rounded-2xl border-l-4 border-blue-500">
                    <h3 class="text-blue-400 text-xs uppercase mb-2">Career Guidance</h3>
                    <p id="career-res" class="text-lg">--</p>
                </div>
                <div class="glass p-6 rounded-2xl border-l-4 border-pink-500">
                    <h3 class="text-pink-400 text-xs uppercase mb-2">Child Prediction (Astrological)</h3>
                    <p id="child-res" class="text-lg">--</p>
                </div>
            </div>
        </section>

        <section id="chat-tab" class="tab-content hidden">
            <div class="glass h-[500px] rounded-3xl p-6 flex flex-col">
                <div id="chat-box" class="flex-1 overflow-y-auto space-y-4 mb-4 text-sm text-gray-400">
                    <p class="bg-white/5 p-3 rounded-lg">Ask me anything about your career, marriage, or health...</p>
                </div>
                <div class="flex gap-2">
                    <input type="text" id="chat-input" class="flex-1 bg-black/50 p-4 rounded-xl outline-none border border-gray-800" placeholder="Type your question...">
                    <button onclick="sendChat()" class="bg-amber-600 px-6 rounded-xl text-black font-bold">ASK</button>
                </div>
            </div>
        </section>

        <section id="shalaka-tab" class="tab-content hidden">
            <div class="text-center glass p-8 rounded-3xl">
                <h2 class="text-2xl font-serif text-amber-500 mb-4">Shri Ram Shalaka</h2>
                <p class="text-gray-400 mb-6">Concentrate on your question and click anywhere on the grid.</p>
                <div class="grid grid-cols-9 gap-1 max-w-sm mx-auto bg-amber-900/20 p-2 rounded-lg" onclick="getShalaka()">
                    <script>
                        for(let i=0; i<81; i++) document.write(`<div class="aspect-square flex items-center justify-center border border-amber-900/30 text-xs cursor-pointer hover:bg-amber-500/20">${"मदसरनभ".charAt(i%6)}</div>`);
                    </script>
                </div>
                <div id="shalaka-res" class="mt-8 hidden p-6 bg-amber-600/10 border border-amber-600 rounded-xl">
                    <p id="shalaka-text" class="italic text-xl mb-2"></p>
                    <p id="shalaka-meaning" class="text-amber-500"></p>
                </div>
            </div>
        </section>

    </main>

    <script>
        const BACKEND = "YOUR_RAILWAY_URL";

        function showTab(tab) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-active'));
            document.getElementById(tab + '-tab').classList.remove('hidden');
            event.target.classList.add('tab-active');
        }

        async function getPrediction() {
            // Fetch logic for /predict
            // Update #career-res and #child-res
            document.getElementById('results').classList.remove('hidden');
        }

        async function getShalaka() {
            const res = await fetch(BACKEND + "/shalaka");
            const data = await res.json();
            document.getElementById('shalaka-res').classList.remove('hidden');
            document.getElementById('shalaka-text').innerText = data.text;
            document.getElementById('shalaka-meaning').innerText = data.meaning;
        }

        async function sendChat() {
            const input = document.getElementById('chat-input');
            const box = document.getElementById('chat-box');
            box.innerHTML += `<p class="text-white">You: ${input.value}</p>`;
            // Fetch logic for /chat
            input.value = "";
        }
    </script>
</body>
</html>
