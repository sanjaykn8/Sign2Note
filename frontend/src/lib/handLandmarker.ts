/**
 * Thin wrapper around @mediapipe/tasks-vision's HandLandmarker for the live
 * webcam demo. Runs entirely in the browser against the live <video>
 * element -- frames are never uploaded anywhere (see PRIVACY.md).
 *
 * NOTE on network use: the WASM runtime and the hand-landmark model file
 * are fetched once from Google's/jsDelivr's CDN (the same way any npm
 * package's binary assets would be fetched, or how a compiled app would
 * bundle a shared library) and then cached by the browser. This is
 * different from "your video is uploaded" -- no camera frames, keypoints,
 * or predictions are ever sent anywhere. See PRIVACY.md for the exact
 * distinction and how to self-host these assets for a fully offline setup.
 */
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let landmarkerPromise: Promise<HandLandmarker> | null = null;

export function loadHandLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
      });
    })();
  }
  return landmarkerPromise;
}

export interface HandDetectionResult {
  /** One array of 21 {x,y,z} normalized landmarks per detected hand, in
   * MediaPipe's detection order (NOT sorted by handedness -- see
   * webcamPipeline.keypointsFromLandmarks for why that matters). */
  hands: { x: number; y: number; z: number }[][];
}

export function detectHands(landmarker: HandLandmarker, video: HTMLVideoElement, timestampMs: number): HandDetectionResult {
  const result = landmarker.detectForVideo(video, timestampMs);
  return { hands: result.landmarks ?? [] };
}
