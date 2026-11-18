let detector = null;
let canvas = null;
let context = null;
self.onmessage = async ({data}) => {
  try {
    if (data.type === "init") {
      // Classic worker: MediaPipe's WASM loader uses importScripts internally.
      const {FilesetResolver,HandLandmarker} = await import("/vendor/mediapipe/0.10.21/vision_bundle.mjs");
      const files = await FilesetResolver.forVisionTasks("/vendor/mediapipe/0.10.21/wasm");
      detector = await HandLandmarker.createFromOptions(files, {
        baseOptions:{modelAssetPath:"/vendor/mediapipe/0.10.21/hand_landmarker.task",delegate:"CPU"},
        runningMode:"VIDEO",numHands:1,
        minHandDetectionConfidence:0.6,minHandPresenceConfidence:0.5,minTrackingConfidence:0.5
      });
      self.postMessage({type:"ready"});
    } else if (data.type === "frame") {
      const {bitmap,timestamp} = data;
      try {
        if (!detector) throw new Error("Bộ tìm khớp tay chưa sẵn sàng.");
        if (!canvas) {
          canvas = new OffscreenCanvas(1,1);
          context = canvas.getContext("2d");
        }
        const scale = Math.min(1,640/Math.max(bitmap.width,bitmap.height));
        canvas.width = Math.round(bitmap.width*scale);
        canvas.height = Math.round(bitmap.height*scale);
        // Match vsl_app.py and Python fallback: mirror pixels before inference.
        context.setTransform(-1,0,0,1,canvas.width,0);
        context.drawImage(bitmap,0,0,canvas.width,canvas.height);
        const begin = performance.now();
        const result = detector.detectForVideo(canvas,timestamp);
        const landmarks = result.landmarks[0]?.map(p=>[p.x,p.y,p.z]) || null;
        self.postMessage({type:"result",timestamp,landmarks,inferenceMs:performance.now()-begin});
      } finally {
        bitmap.close();
      }
    }
  } catch (error) {
    self.postMessage({type:"error",message:error.message || String(error)});
  }
};
