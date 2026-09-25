/**
 * Thin wrapper around @mediapipe/tasks-vision's HolisticLandmarker for the
 * live webcam demo. Runs entirely in the browser against the live <video>
 * element -- frames are never uploaded anywhere (see PRIVACY.md).
 *
 * This replaces the old HandLandmarker-only wrapper (handLandmarker.ts) as
 * part of the hand+body+face schema v2 migration (see FEATURE_SCHEMA.md).
 * A single HolisticLandmarker call gives hands, pose, and face together,
 * from one model, in one coordinate frame -- and critically, it resolves
 * left/right hand identity itself (`leftHandLandmarks` /
 * `rightHandLandmarks`), so there's no more "first hand MediaPipe detects
 * isn't necessarily the left hand" ambiguity the old wrapper had to carry
 * as a documented limitation.
 *
 * NOTE on network use: the WASM runtime and the holistic-landmark model
 * file are fetched once from Google's/jsDelivr's CDN (the same way any
 * npm package's binary assets would be fetched, or how a compiled app
 * would bundle a shared library) and then cached by the browser. This is
 * different from "your video is uploaded" -- no camera frames, keypoints,
 * or predictions are ever sent anywhere. See PRIVACY.md for the exact
 * distinction and how to self-host these assets for a fully offline setup.
 *
 * The model asset URL below is the exact same model family
 * ml_service/feature_extraction.py's Python HolisticLandmarker downloads
 * (see its MODEL_ASSET_URL) -- both languages run the same underlying
 * detector, which is the whole point (FEATURE_SCHEMA.md "Detector").
 */
import { FilesetResolver, HolisticLandmarker, type HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import type { Landmark } from "./featureSchema";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

let landmarkerPromise: Promise<HolisticLandmarker> | null = null;

export function loadHolisticLandmarker(): Promise<HolisticLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
      return HolisticLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        minHandLandmarksConfidence: 0.5,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
      });
    })();
  }
  return landmarkerPromise;
}

export interface HolisticDetectionResult {
  /** Left hand's 21 landmarks, or null if not detected. Resolved directly
   * by the Holistic model -- no post-hoc "which hand is this" guessing. */
  leftHand: Landmark[] | null;
  /** Right hand's 21 landmarks, or null if not detected. */
  rightHand: Landmark[] | null;
  /** Full 33-point MediaPipe Pose topology, or null if no person detected.
   * featureSchema.buildFeatureVector() subsets this to the 9 canonical
   * landmarks itself -- callers should pass the full array through, not
   * pre-subset it. */
  pose: Landmark[] | null;
  /** Full 468-point FaceMesh topology, or null if no face detected.
   * featureSchema.buildFeatureVector() subsets this to the 41 canonical
   * landmarks itself. */
  face: Landmark[] | null;
}

export function detectHolistic(
  landmarker: HolisticLandmarker,
  video: HTMLVideoElement,
  timestampMs: number
): HolisticDetectionResult {
  const result = landmarker.detectForVideo(video, timestampMs) as HolisticLandmarkerResult;
  return {
    leftHand: result.leftHandLandmarks?.[0] ?? null,
    rightHand: result.rightHandLandmarks?.[0] ?? null,
    pose: result.poseLandmarks?.[0] ?? null,
    face: result.faceLandmarks?.[0] ?? null,
  };
}
