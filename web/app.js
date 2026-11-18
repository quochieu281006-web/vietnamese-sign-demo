"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const capture = document.createElement("canvas"), cap = capture.getContext("2d");
const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const labels = {mu:"Mũ",moc:"Móc",trang:"Trăng",sac:"Sắc",huyen:"Huyền",hoi:"Hỏi",nga:"Ngã",nang:"Nặng",space:"Khoảng cách",del:"Xóa",nothing:"Chưa thấy ký hiệu"};
let session = "", running = false, starting = false, stream = null, text = "", ready = false;
let serial = Promise.resolve(), epoch = 0, audio = null, audioUrl = null, speechController = null, reading = false;
function message(value, error = false) { $("message").textContent = value; $("message").classList.toggle("error", error); }
function enqueue(job) { const next = serial.then(job); serial = next.catch(() => {}); return next; }
async function request(path, body, mime = "application/json", signal) {
  const response = await fetch(path, {method:"POST", headers:{"Content-Type":mime,"X-Session":session}, body, signal:signal || AbortSignal.timeout(30000)});
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 401) { ready = false; updateButtons(); }
    throw new Error(data.error || `Lỗi máy chủ (${response.status}).`);
  }
  return response;
}
function updateButtons() {
  $("start").disabled = !ready || running || starting;
  $("stop").disabled = !running && !starting;
  document.querySelectorAll("[data-action]").forEach(b => b.disabled = !ready);
  $("speak").disabled = !ready || !text.trim() || reading;
  $("copy").disabled = !text.trim();
  $("cancelSpeech").disabled = !reading;
}
function render(data, isFrame = false) {
  text = data.text;
  $("output").textContent = text || "Ký tự của bạn sẽ xuất hiện ở đây…";
  $("output").classList.toggle("empty", !text);
  $("output").scrollTop = $("output").scrollHeight;
  $("charCount").textContent = `${Array.from(text).length} ký tự`;
  $("tokens").textContent = data.tokens.length ? data.tokens.map(t => labels[t] || t).join(" · ") : "Chưa có ký tự được chốt";
  $("warnings").textContent = data.warnings.join(" · ");
  const status = data.status.split(" ");
  $("prediction").textContent = status[0] ? `${labels[status[0].replace("~", "")] || status[0].toUpperCase()} ${status.slice(1).join(" ")}` : "—";
  $("progress").value = data.progress;
  $("progressText").textContent = `${Math.round(data.progress * 100)}%`;
  $("toneMode").textContent = data.tone_enabled ? "Nhận dấu thanh bằng cử chỉ đang bật. Có thể dùng các nút trên để hỗ trợ." : "Mô hình dấu thanh chưa bật. Dùng các nút hoặc phím 1–5 để thêm dấu.";
  if (isFrame) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (data.landmarks) {
      const p = data.landmarks.map(([x,y]) => [x * overlay.width,y * overlay.height]);
      ctx.strokeStyle = "#d4ed8c"; ctx.lineWidth = 3;
      connections.forEach(([a,b]) => {ctx.beginPath();ctx.moveTo(...p[a]);ctx.lineTo(...p[b]);ctx.stroke();});
      ctx.fillStyle = "#fff"; p.forEach(([x,y]) => {ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();});
    }
    if (data.committed) message(`Đã chốt: ${labels[data.committed] || data.committed.toUpperCase()}. Tiếp tục ký tự mới hoặc nhả tay để nhập lại.`);
  }
  updateButtons();
}
async function action(name) {
  try {
    await enqueue(async () => render(await (await request("/api/action", JSON.stringify({action:name}))).json()));
    if (name !== "pause") message(name === "clear" ? "Đã xóa văn bản. Bạn có thể bắt đầu ký tự mới." : "Đã cập nhật văn bản đã ghép.");
  }
  catch (e) { message(e.message, true); }
}
async function stopCamera() {
  running = false; starting = false; epoch++;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; video.srcObject = null;
  ctx.clearRect(0,0,overlay.width,overlay.height);
  $("placeholder").hidden = false; $("cameraState").textContent = "Đã dừng"; $("fps").textContent = "";
  updateButtons();
  if (ready) await action("pause");
}
async function startCamera() {
  if (running || starting || !ready) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    message("Camera cần localhost hoặc link HTTPS. Hãy mở http://localhost:8000 trên máy chủ; thiết bị khác cần link HTTPS.", true); return;
  }
  starting = true; const currentEpoch = ++epoch; updateButtons();
  try {
    const next = await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}},audio:false});
    if (currentEpoch !== epoch) {next.getTracks().forEach(t=>t.stop());return;}
    stream = next; video.srcObject = stream; await video.play();
    if (currentEpoch !== epoch) return;
    capture.width = Math.min(video.videoWidth,640);
    capture.height = Math.round(capture.width * video.videoHeight / video.videoWidth);
    overlay.width = capture.width; overlay.height = capture.height;
    video.parentElement.style.aspectRatio = `${capture.width}/${capture.height}`;
    running = true; starting = false; $("placeholder").hidden = true; $("cameraState").textContent = "Đang nhận diện";
    message("Đã bật camera. Giữ một hình tay khoảng 1 giây để chốt ký tự."); updateButtons();
    stream.getVideoTracks()[0].addEventListener("ended", () => { if (running) stopCamera(); });
    loop(currentEpoch);
  } catch (e) {
    if (currentEpoch !== epoch) return;
    await stopCamera();
    const errors = {NotAllowedError:"Bạn chưa cho phép camera. Hãy cấp quyền camera trong trình duyệt rồi thử lại.",NotFoundError:"Không tìm thấy camera trên thiết bị.",NotReadableError:"Không mở được camera. Hãy đóng ứng dụng khác đang dùng webcam."};
    message(errors[e.name] || `Không bật được camera: ${e.message}`,true);
  }
}
async function loop(currentEpoch) {
  while (running && currentEpoch === epoch) {
    const begin = performance.now();
    try {
      cap.drawImage(video,0,0,capture.width,capture.height);
      const blob = await new Promise(resolve => capture.toBlob(resolve,"image/jpeg",0.75));
      if (!blob) throw new Error("Không lấy được hình ảnh từ camera.");
      if (!running || currentEpoch !== epoch) return;
      await enqueue(async () => {
        if (!running || currentEpoch !== epoch) return;
        const data = await (await request("/api/frame",blob,"image/jpeg")).json();
        if (running && currentEpoch === epoch) render(data,true);
      });
      if (!running || currentEpoch !== epoch) return;
      const elapsed = performance.now() - begin;
      $("fps").textContent = `${Math.round(1000/Math.max(elapsed,50))} FPS`;
      await new Promise(resolve => setTimeout(resolve,Math.max(0,50-elapsed)));
    } catch(e) {
      await stopCamera(); message(`Đã dừng camera. ${e.message}`,true); return;
    }
  }
}
function cancelSpeech() {
  speechController?.abort(); speechController = null;
  if (audio) {audio.pause();audio = null;}
  if (audioUrl) {URL.revokeObjectURL(audioUrl);audioUrl = null;}
  window.speechSynthesis?.cancel();
  reading = false; $("speak").textContent = "◖)) Đọc thành tiếng"; updateButtons();
}
function browserSpeech(snapshot) {
  const voice = window.speechSynthesis?.getVoices().find(v=>v.lang.toLowerCase().startsWith("vi"));
  if (!voice) return false;
  const utterance = new SpeechSynthesisUtterance(snapshot);utterance.voice=voice;utterance.lang="vi-VN";utterance.rate=.85;
  utterance.onend=cancelSpeech;
  utterance.onerror=()=>{cancelSpeech();message("Giọng đọc trên thiết bị gặp lỗi. Hãy thử lại.",true);};
  message("Đang dùng giọng tiếng Việt của trình duyệt do dịch vụ trực tuyến không khả dụng.");
  speechController=null;window.speechSynthesis.speak(utterance);return true;
}
async function speak() {
  if (!text.trim() || reading) return;
  const snapshot = text; const controller = new AbortController(); speechController = controller;
  reading = true; $("speak").textContent = "Đang tạo giọng đọc…";updateButtons();
  try {
    const response = await request("/api/speech","{}","application/json",controller.signal);
    const blob = await response.blob();
    if (controller.signal.aborted) return;
    audioUrl = URL.createObjectURL(blob); audio = new Audio(audioUrl);
    audio.onended=cancelSpeech;
    audio.onerror=()=>{cancelSpeech();message("Không phát được âm thanh trên thiết bị này.",true);};
    await audio.play(); $("speak").textContent="Đang đọc…";
  } catch(e) {
    if (controller.signal.aborted) return;
    if (!browserSpeech(snapshot)) {cancelSpeech();message(`${e.message} Thiết bị chưa có giọng Việt dự phòng.`,true);}
  }
}
$("start").onclick = startCamera;
$("stop").onclick = async () => {await stopCamera();message("Camera đã dừng. Văn bản vẫn được giữ trong phiên hiện tại.");};
document.querySelectorAll("[data-action]").forEach(b=>b.onclick=()=>action(b.dataset.action));
$("speak").onclick=speak; $("cancelSpeech").onclick=cancelSpeech;
$("copy").onclick=async()=>{try{await navigator.clipboard.writeText(text);message("Đã sao chép văn bản.");}catch{message("Không sao chép tự động được. Bạn có thể chọn văn bản trong ô rồi sao chép.",true);}};
document.addEventListener("keydown", e=>{
  if (!ready || e.ctrlKey || e.metaKey || e.altKey || e.repeat || /INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName)) return;
  const actions={"1":"sac","2":"huyen","3":"hoi","4":"nga","5":"nang","Backspace":"backspace"," ":"space"};
  if (actions[e.key]) {e.preventDefault();action(actions[e.key]);}
});
document.addEventListener("visibilitychange",()=>{if(document.hidden && (running||starting)) stopCamera();});
window.addEventListener("pagehide",()=>{
  if(stream) stream.getTracks().forEach(t=>t.stop());cancelSpeech();
  if(session) fetch("/api/close",{method:"POST",headers:{"X-Session":session},body:"",keepalive:true}).catch(()=>{});
});
window.addEventListener("pageshow",event=>{if(event.persisted) window.location.reload();});
async function init() {
  updateButtons();
  try {
    const data=await (await request("/api/session","{}")).json(); session=data.session;ready=true;render(data);
    message("Mô hình đã sẵn sàng. Nhấn Bật camera để bắt đầu nhận diện ký tự.");
  } catch(e) {message(`Chưa kết nối được máy chủ. ${e.message} Hãy kiểm tra cửa sổ chạy ứng dụng và tải lại trang.`,true);}
}
init();
