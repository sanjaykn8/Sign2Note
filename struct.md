sign2notes/
├─ data/
│  ├─ wlasl/
│  │  ├─ videos/                 # raw .mp4 files named by video_id (e.g. 0001.mp4)
│  │  └─ WLASL_v0.3.json         # dataset manifest you already have
│  ├─ features/                  # output of feature extraction (.npy per video)
│  └─ index.csv                  # generated: video_id,label,timestamp_start,timestamp_end (optional)
│
├─ models/
│  ├─ sign_recog/
│  │  ├─ checkpoints/            # training checkpoints (best.pt, latest.pt)
│  │  └─ sign_recog.onnx         # exported inference model
│  └─ llama/
│     └─ llama-3.2-3b-instruct.gguf  # local inference model (GGUF) or quantized format
│
├─ backend/
│  ├─ server.js                  # Node/Express API (uploads, job trigger)
│  ├─ package.json
│  └─ uploads/                   # uploaded videos for demo
│
├─ ml_service/                   # Python ML microservice
│  ├─ feature_extraction.py      # extract mediapipe features -> data/features/*.npy
│  ├─ dataset.py                 # SignDataset (used by train.py)
│  ├─ model.py                   # TemporalCNN model definition
│  ├─ train.py                   # training loop
│  ├─ infer.py                   # end-to-end inference (features->frame_probs->HMM->tokens)
│  ├─ inference_viterbi.py       # smoothing / collapse -> token sequence
│  └─ requirements.txt
│
├─ frontend/
│  └─ react-app/                 # React demo to upload video and render Markdown notes
│
├─ utils/
│  ├─ prompts/
│  │  └─ note_prompt_template.txt
│  └─ convert_llama_instructions.txt
│
└─ README.md
