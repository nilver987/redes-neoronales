let currentTab = 'train';

function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tab + '-tab').classList.add('active');
  document.querySelector(`.tab-btn[onclick="switchTab('${tab}')"]`).classList.add('active');
}

async function uploadData() {
  const file = document.getElementById('fileInput').files[0];
  if (!file) return showStatus('Selecciona un archivo', 'error');
  
  const formData = new FormData();
  formData.append('file', file);
  const status = document.getElementById('trainStatus');
  showStatus('Entrenando modelo...', 'loading');
  
  try {
    const res = await fetch('/upload_data', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.success) {
      showStatus(`${data.message}<br>Precisión: ${data.accuracy} | Premium: ${data.premium_ratio}`, 'success');
    } else {
      showStatus(data.error, 'error');
    }
  } catch (e) {
    showStatus('Error de red. Verifica que el servidor esté corriendo.', 'error');
    console.error(e);
  }
}

function showStatus(message, type) {
  const status = document.getElementById('trainStatus');
  status.innerHTML = message;
  status.className = `status ${type}`;
}

document.getElementById('analyzeForm').onsubmit = async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData);
  
  const result = document.getElementById('analysisResult');
  const spinner = document.getElementById('spinner');
  const content = document.getElementById('resultContent');
  
  result.style.display = 'block';
  spinner.style.display = 'block';
  content.style.display = 'none';
  
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const json = await res.json();
    
    spinner.style.display = 'none';
    content.style.display = 'block';
    
    if (json.success) {
      document.getElementById('resultTitle').textContent = json.isPremium ? 'PREMIUM' : 'ESTÁNDAR';
      document.getElementById('resultTitle').className = json.isPremium ? 'premium' : 'standard';
      document.getElementById('resultConfidence').textContent = json.confidence;
      document.getElementById('resultRecommendation').innerHTML = `<em>${json.recommendation}</em>`;
      document.getElementById('resultPrice').textContent = '$' + json.estimatedPrice.toLocaleString();
      document.getElementById('resultThreshold').textContent = '$' + json.threshold.toLocaleString();
      e.target.style.display = 'none';
    } else {
      content.innerHTML = `<div class="error">${json.error}</div>`;
    }
  } catch (e) {
    spinner.style.display = 'none';
    content.style.display = 'block';
    content.innerHTML = `<div class="error">Error de conexión. Intenta de nuevo.</div>`;
    console.error(e);
  }
};

function editAgain() {
  document.getElementById('analyzeForm').style.display = 'block';
  document.getElementById('analysisResult').style.display = 'none';
}

function openModelModal() { document.getElementById('modelModal').style.display = 'block'; }
function closeModelModal() { document.getElementById('modelModal').style.display = 'none'; }

document.getElementById('modelForm').onsubmit = async (e) => {
  e.preventDefault();
  const type = document.getElementById('modelType').value;
  try {
    const res = await fetch('/train_model', { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify({ type }) 
    });
    const data = await res.json();
    alert(data.message);
    closeModelModal();
  } catch (e) {
    alert('Error al cambiar modelo');
  }
};

window.onclick = (e) => { 
  if (e.target == document.getElementById('modelModal')) closeModelModal(); 
};