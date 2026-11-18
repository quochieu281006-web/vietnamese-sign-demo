"use strict";
const $ = id => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const capture = document.createElement("canvas"), cap = capture.getContext("2d");
const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const labels = {mu:"Mũ",moc:"Móc",trang:"Trăng",sac:"Sắc",huyen:"Huyền",hoi:"Hỏi",nga:"Ngã",nang:"Nặng",space:"Khoảng cách",del:"Xóa",nothing:"Chưa thấy ký hiệu"};
let session = "", running = false, starting = false, stream = null, text = "", ready = false;
let serial = Promise.resolve(), epoch = 0, audio = null, audioUrl = null, speechController = null, reading = false;
let cameraFacing = "user", switchingCamera = false;
let processingMode = "browser", handWorker = null, workerRequest = null;
let pendingSamples = [], upload = null, lastUpload = 0, networkMs = 0;
function trackerError(value) {const error=new Error(value);error.name="TrackerError";return error;}
function stopTracker() {
  handWorker?.terminate(); handWorker=null;
  if(workerRequest) {clearTimeout(workerRequest.timer);workerRequest.reject(trackerError("Đã dừng xử lý khớp tay."));workerRequest=null;}
  pendingSamples=[];upload=null;
}
function workerCall(data, transfer=[], timeout=30000) {
  return new Promise((resolve,reject)=>{
    if(!handWorker || workerRequest) {reject(trackerError("Bộ tìm khớp tay đang bận hoặc đã dừng."));return;}
    const timer=setTimeout(()=>{workerRequest=null;reject(trackerError("Tải hoặc xử lý khớp tay quá lâu. Thử lại hoặc chọn chế độ Tương thích."));},timeout);
    workerRequest={resolve,reject,timer};
    try {handWorker.postMessage(data,transfer);} catch(e) {clearTimeout(timer);workerRequest=null;reject(trackerError(e.message));}
  });
}
async function startTracker() {
  if(!window.Worker || !window.OffscreenCanvas || !window.createImageBitmap) {
    throw trackerError("Thiết bị chưa hỗ trợ chế độ Nhanh. Hãy chọn Tương thích hoặc thử trình duyệt mới hơn.");
  }
  $("trackingInfo").textContent="Đang tải bộ tìm khớp tay. Lần đầu có thể cần chờ tải mô hình.";
  handWorker=new Worker("/hand-worker.js");
  handWorker.onmessage=({data})=>{
    const waiting=workerRequest;if(!waiting)return;
    workerRequest=null;clearTimeout(waiting.timer);
    if(data.type==="error")waiting.reject(trackerError(data.message));else waiting.resolve(data);
  };
  handWorker.onerror=()=>{
    if(workerRequest){const waiting=workerRequest;workerRequest=null;clearTimeout(waiting.timer);waiting.reject(trackerError("Không tải được bộ tìm khớp tay. Kiểm tra tài nguyên web hoặc thử chế độ Tương thích."));}
  };
  await workerCall({type:"init"},[],60000);
}
function drawLandmarks(landmarks) {
  ctx.clearRect(0,0,overlay.width,overlay.height);
  if(!landmarks)return;
  const points=landmarks.map(([x,y])=>[x*overlay.width,y*overlay.height]);
  ctx.strokeStyle="#d4ed8c";ctx.lineWidth=3;
  connections.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(...points[a]);ctx.lineTo(...points[b]);ctx.stroke();});
  ctx.fillStyle="#fff";points.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();});
}
function flushSamples(currentEpoch,force=false) {
  if(upload?.epoch===currentEpoch)return upload.promise;
  if(!pendingSamples.length || (!force && performance.now()-lastUpload<150))return Promise.resolve();
  const frames=pendingSamples.splice(0,60);
  const job={epoch:currentEpoch,promise:null};lastUpload=performance.now();
  job.promise=enqueue(async()=>{
    if(!running || currentEpoch!==epoch)return;
    const begin=performance.now();
    const result=await (await request("/api/landmarks",JSON.stringify({frames}))).json();
    if(running && currentEpoch===epoch){networkMs=performance.now()-begin;render(result);}
  }).catch(async error=>{
    if(currentEpoch!==epoch)return;
    await stopCamera();message(`Đã dừng camera. ${error.message}`,true);
  }).finally(()=>{if(upload===job)upload=null;});
  upload=job;return job.promise;
}
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
  $("start").disabled = !ready || running || starting || switchingCamera;
  $("stop").disabled = !running && !starting;
  $("cameraFacing").disabled = !ready || starting || switchingCamera;
  $("processingMode").disabled = !ready || starting || switchingCamera;
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
    drawLandmarks(data.landmarks);
  }
  if (data.committed) message(`Đã chốt: ${labels[data.committed] || data.committed.toUpperCase()}. Tiếp tục ký tự mới hoặc nhả tay để nhập lại.`);
  updateButtons();
}
async function action(name) {
  try {
    if(name!=="pause" && running && processingMode==="browser") {
      await flushSamples(epoch,true);
      await flushSamples(epoch,true);
    }
    await enqueue(async () => render(await (await request("/api/action", JSON.stringify({action:name}))).json()));
    if (name !== "pause") message(name === "clear" ? "Đã xóa văn bản. Bạn có thể bắt đầu ký tự mới." : "Đã cập nhật văn bản đã ghép.");
  }
  catch (e) { message(e.message, true); }
}
async function stopCamera() {
  running = false; starting = false; epoch++;
  stopTracker();
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
    if(processingMode==="browser")await startTracker();
    if(currentEpoch!==epoch)return;
    // Require the rear camera instead of silently reusing the front one.
    // Keep a preference for the front so ordinary desktop webcams still work.
    const facingMode = cameraFacing === "environment" ? {exact:"environment"} : {ideal:"user"};
    if (cameraFacing === "environment" && !navigator.mediaDevices.getSupportedConstraints?.().facingMode) {
      const unsupported = new Error("Trình duyệt chưa hỗ trợ chọn camera sau. Hãy thử Chrome hoặc Safari trên điện thoại.");
      unsupported.name = "CameraSelectionError"; throw unsupported;
    }
    const next = await navigator.mediaDevices.getUserMedia({video:{facingMode,width:{ideal:640},height:{ideal:480}},audio:false});
    if (currentEpoch !== epoch) {next.getTracks().forEach(t=>t.stop());return;}
    stream = next; video.srcObject = stream; await video.play();
    if (currentEpoch !== epoch) return;
    const activeFacing = next.getVideoTracks()[0].getSettings?.().facingMode || cameraFacing;
    cameraFacing = activeFacing === "environment" ? "environment" : "user";
    $("cameraFacing").value = cameraFacing;
    video.parentElement.classList.toggle("rear-camera", cameraFacing === "environment");
    document.querySelector(".camera-caption").textContent = cameraFacing === "environment" ? "CAMERA SAU · MỘT TAY" : "CAMERA TRƯỚC · HÌNH ẢNH LẬT GƯƠNG";
    // Also bound portrait video by its longest side on mobile devices.
    const scale = Math.min(1,640 / Math.max(video.videoWidth,video.videoHeight));
    capture.width = Math.round(video.videoWidth * scale);
    capture.height = Math.round(video.videoHeight * scale);
    overlay.width = capture.width; overlay.height = capture.height;
    video.parentElement.style.aspectRatio = `${capture.width}/${capture.height}`;
    running = true; starting = false; $("placeholder").hidden = true; $("cameraState").textContent = "Đang nhận diện";
    message(`Đã bật ${cameraFacing === "environment" ? "camera sau" : "camera trước"}. Giữ một hình tay khoảng 1 giây để chốt ký tự.`); updateButtons();
    next.getVideoTracks()[0].addEventListener("ended", () => { if (running && stream === next) stopCamera(); });
    if(processingMode==="browser") {
      pendingSamples=[];lastUpload=0;networkMs=0;
      browserLoop(currentEpoch);
    } else {
      $("trackingInfo").textContent="Chế độ tương thích gửi ảnh tới máy chủ; nhịp dấu thanh phụ thuộc máy chủ và mạng.";
      loop(currentEpoch);
    }
  } catch (e) {
    if (currentEpoch !== epoch) return;
    await stopCamera();
    const errors = {NotAllowedError:"Bạn chưa cho phép camera. Hãy cấp quyền camera trong trình duyệt rồi thử lại.",NotFoundError:"Không tìm thấy camera đã chọn. Hãy thử Camera trước hoặc dùng thiết bị khác.",OverconstrainedError:"Thiết bị không có camera phù hợp. Hãy chọn Camera trước hoặc thử trình duyệt khác.",NotReadableError:"Không mở được camera. Hãy đóng ứng dụng khác đang dùng webcam."};
    message(errors[e.name] || (e.name==="TrackerError" ? e.message : `Không bật được camera: ${e.message}`),true);
  }
}
async function changeCamera() {
  cameraFacing = $("cameraFacing").value;
  if (!running) {
    message(`Đã chọn ${cameraFacing === "environment" ? "camera sau" : "camera trước"}. Nhấn Bật camera để bắt đầu.`);
    return;
  }
  switchingCamera = true;
  // Release the old track and clear only tracking history, preserving text.
  const stopped = stopCamera();
  const switchEpoch = epoch;
  try {
    await stopped;
    if (epoch === switchEpoch && !document.hidden && ready) await startCamera();
  } finally {
    switchingCamera = false;
    updateButtons();
  }
}
async function changeProcessingMode() {
  processingMode=$("processingMode").value;
  if(!running) {
    $("trackingInfo").textContent=processingMode==="browser" ? "Tìm khớp tay trên thiết bị; chỉ gửi tọa độ về máy chủ." : "Gửi ảnh tới máy chủ; có thể chậm trên hosting miễn phí.";
    return;
  }
  switchingCamera=true;
  const stopped=stopCamera();const switchEpoch=epoch;
  try {await stopped;if(epoch===switchEpoch && !document.hidden && ready)await startCamera();}
  finally {switchingCamera=false;updateButtons();}
}
async function browserLoop(currentEpoch) {
  let previousVideoTime=-1,rateSince=performance.now(),count=0,fps=0;
  while(running && currentEpoch===epoch) {
    const begin=performance.now();
    try {
      if(video.currentTime===previousVideoTime) {await new Promise(resolve=>setTimeout(resolve,10));continue;}
      previousVideoTime=video.currentTime;
      const timestamp=performance.now();
      const bitmap=await createImageBitmap(video);
      if(!running || currentEpoch!==epoch){bitmap.close();return;}
      const result=await workerCall({type:"frame",bitmap,timestamp},[bitmap]);
      if(!running || currentEpoch!==epoch)return;
      drawLandmarks(result.landmarks);
      const landmarks=result.landmarks?.map(p=>p.map(n=>Number(n.toFixed(6)))) || null;
      pendingSamples.push({t:timestamp/1000,landmarks});
      if(pendingSamples.length>60)throw new Error("Máy chủ hoặc mạng chưa theo kịp chuỗi chuyển động. Hãy thử lại khi kết nối ổn định hoặc dùng nút dấu dự phòng.");
      // Do not await the upload: joint tracking must keep capturing during RTT.
      flushSamples(currentEpoch);
      count++;
      if(performance.now()-rateSince>=1000){fps=Math.round(count*1000/(performance.now()-rateSince));count=0;rateSince=performance.now();}
      $("fps").textContent=`${fps||"…"} FPS · trên thiết bị`;
      $("trackingInfo").textContent=`Khớp tay trên thiết bị${fps ? `: ${fps} FPS` : ""}. Máy chủ: ${Math.round(networkMs)} ms/gói.${fps>0 && fps<8 ? " Nhịp hình thấp, dấu chuyển động có thể khó nhận." : ""}`;
      await new Promise(resolve=>setTimeout(resolve,Math.max(0,50-(performance.now()-begin))));
    } catch(error) {
      if(currentEpoch!==epoch)return;
      await stopCamera();message(error.message,true);return;
    }
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
      if (currentEpoch !== epoch) return;
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
$("cameraFacing").onchange = changeCamera;
$("processingMode").onchange = changeProcessingMode;
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
  running=false;starting=false;epoch++;
  stopTracker();
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
