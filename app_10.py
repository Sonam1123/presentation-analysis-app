# Full updated app_4.py
import os
import tempfile
import streamlit as st
import subprocess
import ffmpeg
import librosa
import librosa.display
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd
import cv2
import math
import shutil
import zipfile
import mediapipe as mp

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="🎧 Presentation's StudAssess",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ================== AUDIO PROCESSOR ==================
class AudioProcessor:
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        self.audio_dir = os.path.join(temp_dir, "audio")
        self.noise_dir = os.path.join(temp_dir, "noise_reduced")
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.noise_dir, exist_ok=True)
        self.ffmpeg_path = self.find_ffmpeg()

    def find_ffmpeg(self):
        """Locate FFmpeg executable"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return "ffmpeg"
        except Exception:
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"D:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"D:\M.Tech_3rd sem\MP_1_PRACTICAL\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
            st.error("❌ FFmpeg not found! Please check your installation.")
            return None

    def extract_audio_from_video(self, video_path):
        """Extract audio track as .wav file"""
        if not self.ffmpeg_path:
            st.error("FFmpeg not available")
            return None
        
        audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
        audio_path = os.path.join(self.audio_dir, audio_filename)

        try:
            (ffmpeg
             .input(video_path)
             .output(audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16000")
             .overwrite_output()
             .run(quiet=True, cmd=self.ffmpeg_path))
            return audio_path
        except Exception as e:
            st.error(f"❌ Error extracting audio: {str(e)}")
            return None

    def compute_snr(self, y, sr, silence_thresh=0.01):
        """Compute SNR as speech_power / noise_power"""
        rms = librosa.feature.rms(y=y)[0]
        speech_power = np.mean(rms[rms >= silence_thresh] ** 2) if np.any(rms >= silence_thresh) else 1e-10
        noise_power = np.mean(rms[rms < silence_thresh] ** 2) if np.any(rms < silence_thresh) else 1e-10
        return speech_power / noise_power

    def reduce_noise(self, audio_path):
        """Apply noise reduction and compute SNR improvement"""
        try:
            y_raw, sr = librosa.load(audio_path, sr=None)
            y_clean = nr.reduce_noise(y=y_raw, sr=sr)

            # Compute SNR before and after
            snr_before = self.compute_snr(y_raw, sr)
            snr_after = self.compute_snr(y_clean, sr)
            improvement_db = 10 * np.log10(snr_after / snr_before) if snr_before > 0 else 0

            clean_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_clean.wav"
            clean_path = os.path.join(self.noise_dir, clean_filename)
            sf.write(clean_path, y_clean, sr)

            return clean_path, y_raw, y_clean, sr, snr_before, snr_after, improvement_db

        except Exception as e:
            st.error(f"❌ Noise reduction failed: {str(e)}")
            return None, None, None, None, None, None, None
        

    def remove_silence(self, y, sr, top_db=25):
        """
        Remove silent parts from the cleaned audio signal.
        top_db: threshold (in dB) below reference to consider as silence.
        """
        try:
            intervals = librosa.effects.split(y, top_db=top_db)
            non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
            return non_silent_audio, intervals
        except Exception as e:
            st.error(f"❌ Silence removal failed: {str(e)}")
            return y, []

    def create_spectrograms(self, y_raw, y_clean, sr):
        """Return comparison spectrogram image (before vs after noise reduction)"""
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        # Before
        S_raw = librosa.amplitude_to_db(np.abs(librosa.stft(y_raw)), ref=np.max)
        img1 = librosa.display.specshow(S_raw, sr=sr, x_axis='time', y_axis='hz', ax=axs[0])
        axs[0].set_title("Before Noise Reduction")
        fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")

        # After
        S_clean = librosa.amplitude_to_db(np.abs(librosa.stft(y_clean)), ref=np.max)
        img2 = librosa.display.specshow(S_clean, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
        axs[1].set_title("After Noise Reduction")
        fig.colorbar(img2, ax=axs[1], format="%+2.0f dB")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf    
# =========================
# 🎥 VIDEO FRAME EXTRACTION FUNCTION
# =========================
def extract_frames(video_path, output_dir, frame_rate=1, resize_dim=(224, 224)):
    """
    Extract frames from the given video file.
    Parameters:
        video_path: path to video file
        output_dir: directory to save extracted frames
        frame_rate: frames per second to save
        resize_dim: (width, height) resize each frame
    Returns:
        Folder path containing extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / frame_rate))
    frame_count, saved_count = 0, 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_folder = os.path.join(output_dir, video_name)
    os.makedirs(save_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            resized = cv2.resize(frame, resize_dim)
            frame_filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(save_folder, frame_filename), resized)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"✅ Extracted {saved_count} frames from {video_path} into {save_folder}")
    return save_folder
# =========================
# 🎞️ SPLIT FRAMES INTO CHUNKS FUNCTION
# =========================
def split_into_chunks(input_dir, output_dir, frames_per_chunk=5):
    """
    Split extracted frames into smaller subfolders (chunks).
    Each chunk will contain N frames (frames_per_chunk).
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, video_folder)
        if not os.path.isdir(folder_path):
            continue

        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        total_chunks = math.ceil(len(frame_files) / frames_per_chunk)

        for chunk_idx in range(total_chunks):
            chunk_folder = os.path.join(output_dir, video_folder, f"chunk_{chunk_idx:03d}")
            os.makedirs(chunk_folder, exist_ok=True)

            start_idx = chunk_idx * frames_per_chunk
            end_idx = start_idx + frames_per_chunk
            for i, file in enumerate(frame_files[start_idx:end_idx]):
                src = os.path.join(folder_path, file)
                dst = os.path.join(chunk_folder, f"frame_{i+1:04d}.jpg")
                shutil.copy(src, dst)

        print(f"✅ Split {video_folder} into {total_chunks} chunks")

    return output_dir
# =========================
# 👁️ FACE DETECTION FUNCTION (Using Mediapipe)
# =========================
def detect_and_save_faces(input_dir, output_dir, resize_dim=(224, 224)):
    """
    Detect faces in all frames inside chunk folders and save cropped faces.
    """
    os.makedirs(output_dir, exist_ok=True)
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for video_folder in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, video_folder)
            if not os.path.isdir(folder_path):
                continue

            for chunk in os.listdir(folder_path):
                chunk_path = os.path.join(folder_path, chunk)
                frame_files = sorted([f for f in os.listdir(chunk_path) if f.endswith(".jpg")])

                out_chunk_path = os.path.join(output_dir, video_folder, chunk)
                os.makedirs(out_chunk_path, exist_ok=True)

                for i, file in enumerate(frame_files):
                    img_path = os.path.join(chunk_path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)

                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            h, w, _ = img.shape
                            x = int(bboxC.xmin * w)
                            y = int(bboxC.ymin * h)
                            width = int(bboxC.width * w)
                            height = int(bboxC.height * h)
                            x, y = max(0, x), max(0, y)
                            face = img[y:y + height, x:x + width]
                            if face.size != 0:
                                face_resized = cv2.resize(face, resize_dim)
                                save_path = os.path.join(out_chunk_path, f"face_{i+1:04d}.jpg")
                                cv2.imwrite(save_path, face_resized)

        print(f"✅ Face detection completed. Faces saved in: {output_dir}")
    return output_dir
# =========================
# 🧠 HEAD POSE & GAZE FEATURE EXTRACTION
# =========================
def extract_head_gaze_features(faces_dir):
    """
    Extract average head pose (pitch, yaw, roll) and gaze (horizontal, vertical)
    for each video chunk folder.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    def get_camera_matrix(frame):
        h, w, _ = frame.shape
        focal_length = w
        center = (w/2, h/2)
        return np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double")

    def compute_gaze(landmarks, w, h):
        left_eye_outer = landmarks[33]
        left_eye_inner = landmarks[133]
        left_iris = landmarks[468]
        right_eye_outer = landmarks[362]
        right_eye_inner = landmarks[263]
        right_iris = landmarks[473]

        left_ratio_h = (left_iris.x - left_eye_inner.x) / (left_eye_outer.x - left_eye_inner.x + 1e-6)
        right_ratio_h = (right_iris.x - right_eye_inner.x) / (right_eye_outer.x - right_eye_inner.x + 1e-6)
        horiz_ratio = (left_ratio_h + right_ratio_h) / 2

        left_ratio_v = (left_iris.y - left_eye_inner.y) / (left_eye_outer.y - left_eye_inner.y + 1e-6)
        right_ratio_v = (right_iris.y - right_eye_inner.y) / (right_eye_outer.y - right_eye_inner.y + 1e-6)
        vert_ratio = (left_ratio_v + right_ratio_v) / 2

        return horiz_ratio, vert_ratio

    results_list = []

    for video_folder in os.listdir(faces_dir):
        video_path = os.path.join(faces_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        for chunk_folder in sorted(os.listdir(video_path)):
            chunk_path = os.path.join(video_path, chunk_folder)
            pitch_list, yaw_list, roll_list = [], [], []
            horiz_list, vert_list = [], []

            for frame_file in sorted(os.listdir(chunk_path)):
                frame_path = os.path.join(chunk_path, frame_file)
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                h, w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                faces = face_mesh.process(rgb)
                if faces.multi_face_landmarks:
                    landmarks = faces.multi_face_landmarks[0].landmark
                    if len(landmarks) >= 474:
                        horiz, vert = compute_gaze(landmarks, w, h)
                        horiz_list.append(horiz)
                        vert_list.append(vert)

                        # Head pose estimation
                        image_points = np.array([
                            [landmarks[1].x * w, landmarks[1].y * h],
                            [landmarks[152].x * w, landmarks[152].y * h],
                            [landmarks[263].x * w, landmarks[263].y * h],
                            [landmarks[33].x * w, landmarks[33].y * h],
                            [landmarks[287].x * w, landmarks[287].y * h],
                            [landmarks[57].x * w, landmarks[57].y * h]
                        ], dtype="double")

                        camera_matrix = get_camera_matrix(img)
                        dist_coeffs = np.zeros((4,1))
                        success, rotation_vector, _ = cv2.solvePnP(
                            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                        )

                        if success:
                            rmat, _ = cv2.Rodrigues(rotation_vector)
                            sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                            pitch = np.arctan2(-rmat[2,0], sy) * 180/np.pi
                            yaw = np.arctan2(rmat[1,0], rmat[0,0]) * 180/np.pi
                            roll = np.arctan2(rmat[2,1], rmat[2,2]) * 180/np.pi
                            pitch_list.append(pitch)
                            yaw_list.append(yaw)
                            roll_list.append(roll)

            results_list.append({
                "video_name": video_folder,
                "chunk": chunk_folder,
                "pitch_mean": np.mean(pitch_list) if pitch_list else np.nan,
                "yaw_mean": np.mean(yaw_list) if yaw_list else np.nan,
                "roll_mean": np.mean(roll_list) if roll_list else np.nan,
                "horiz_mean": np.mean(horiz_list) if horiz_list else np.nan,
                "vert_mean": np.mean(vert_list) if vert_list else np.nan
            })

    face_mesh.close()
    print("✅ Head & gaze feature extraction completed.")
    return results_list
# =========================
# 💪 BODY MOVEMENT FEATURE EXTRACTION
# =========================
def extract_body_features(chunks_dir):
    """
    Estimate body movement (average torso displacement) per chunk.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results_list = []

    for video_folder in os.listdir(chunks_dir):
        video_path = os.path.join(chunks_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        for chunk_folder in sorted(os.listdir(video_path)):
            chunk_path = os.path.join(video_path, chunk_folder)
            movements = []
            prev_torso = None

            for frame_file in sorted(os.listdir(chunk_path)):
                frame_path = os.path.join(chunk_path, frame_file)
                img = cv2.imread(frame_path)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    torso_points = [
                        np.array([lm[11].x, lm[11].y]),  # left shoulder
                        np.array([lm[12].x, lm[12].y]),  # right shoulder
                        np.array([lm[23].x, lm[23].y]),  # left hip
                        np.array([lm[24].x, lm[24].y])   # right hip
                    ]

                    if prev_torso is not None:
                        disp = np.mean([
                            np.linalg.norm(tp - pp) for tp, pp in zip(torso_points, prev_torso)
                        ])
                        movements.append(disp)

                    prev_torso = torso_points

            results_list.append({
                "video_name": video_folder,
                "chunk": chunk_folder,
                "body_mean": np.mean(movements) if movements else np.nan
            })

    pose.close()
    print("✅ Body feature extraction completed.")
    return results_list
# =========================
# 🧮 SCORING FUNCTIONS
# =========================

def head_score(pitch, yaw, head_strict=20, head_moderate=10, fluctuation_thresh=5):
    if np.isnan(pitch) or np.isnan(yaw):
        return 0, "No face detected"

    if abs(pitch) > head_strict or abs(yaw) > head_strict:
        return 1, "Excessive head movement"
    elif abs(pitch) > head_moderate or abs(yaw) > head_moderate:
        return 2, "Moderate head movement"
    elif abs(pitch) > fluctuation_thresh or abs(yaw) > fluctuation_thresh:
        return 2, "Unstable head posture"
    else:
        return 3, "Stable head posture"

def gaze_score(horiz, vert, attentive_thresh=0.7, fluctuation_thresh=0.05):
    if np.isnan(horiz) or np.isnan(vert):
        return 0, "No gaze detected"

    attentive = abs(horiz - 0.5) < 0.1 and abs(vert - 0.5) < 0.1
    if attentive and abs(horiz - 0.5) < fluctuation_thresh and abs(vert - 0.5) < fluctuation_thresh:
        return 3, "Consistently attentive"
    elif attentive:
        return 2, "Partially attentive"
    else:
        return 1, "Distracted gaze"

def body_score(body_mean, body_min=0.01, body_max=0.1, fluctuation_thresh=0.05):
    if np.isnan(body_mean):
        return 0, "No body detected"

    if body_mean < body_min:
        return 1, "Too static"
    elif body_mean > body_max:
        return 2, "Excessive movement"
    elif abs(body_mean) > fluctuation_thresh:
        return 2, "Unstable posture"
    else:
        return 3, "Good posture"

def calculate_final_video_score(df, method="simple", weights=None):
    video_scores = {}

    for video_id, group in df.groupby("video_name"):
        head_avg = group["head_score"].replace(0, np.nan).mean()
        gaze_avg = group["gaze_score"].replace(0, np.nan).mean()
        body_avg = group["body_score"].replace(0, np.nan).mean()

        if method == "weighted" and weights:
            final_score = (
                head_avg * weights["head"] +
                gaze_avg * weights["gaze"] +
                body_avg * weights["body"]
            )
        else:
            valid_scores = [v for v in [head_avg, gaze_avg, body_avg] if not np.isnan(v)]
            final_score = np.mean(valid_scores) if valid_scores else 0

        if final_score >= 2.5:
            feedback = "Excellent performance"
        elif final_score >= 2.0:
            feedback = "Good performance"
        elif final_score >= 1.5:
            feedback = "Average — needs improvement"
        else:
            feedback = "Poor — significant improvement needed"

        video_scores[video_id] = {
            "final_score": round(final_score, 2),
            "components": {
                "head": round(head_avg, 2),
                "gaze": round(gaze_avg, 2),
                "body": round(body_avg, 2)
            },
            "feedback": feedback
        }

    return video_scores


def final_ppt_scoring(audio_score, video_score, method="simple", weights=None):
    if method == "simple":
        final_score = (audio_score + video_score) / 2
    elif method == "weighted" and weights:
        total = weights["audio"] + weights["video"]
        final_score = (
            weights["audio"] * audio_score + weights["video"] * video_score
        ) / total
    else:
        final_score = (audio_score + video_score) / 2  # fallback

    percentage = ((final_score - 1) / 2) * 100
    if final_score <= 1.5:
        feedback = "Needs Improvement"
    elif final_score <= 2.3:
        feedback = "Average"
    else:
        feedback = "Excellent"

    return {
        "Final PPT Score (1–3)": round(final_score, 2),
        "Final PPT Performance (%)": round(percentage, 2),
        "Interpretation": feedback
    }

# ================== STREAMLIT APP ==================
class StreamlitApp:
    def __init__(self):
        # self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = tempfile.mkdtemp()

        os.makedirs(self.temp_dir, exist_ok=True)

        self.audio_processor = AudioProcessor(self.temp_dir)

    # def home_page(self):
    #     st.markdown('<div class="main-header">Multimodal Presentation Analysis</div>', unsafe_allow_html=True)
    #     st.markdown("""
    #     ### 👋 Welcome to Multimodal Presentation Analysis

    #     Developed by **<span style='color:#1f77b4;font-weight:700;'>Sonam Singh</span>**  
    #     A smart system that evaluates presentation quality using **Audio**, **Video**, and **Multimodal Fusion**.

    #     ---

    #     ## 🔊 Audio Analysis
    #     Our audio engine extracts meaningful speaking-related features:
    #     - 🎚 **Volume Stability** – detects confidence & clarity  
    #     - ⏸ **Pause Detection** – identifies silence, hesitation & fluency  
    #     - 🎵 **Pitch (YIN Algorithm)** – analyzes voice variation & expressiveness  
    #     - 🗣 **Speaking Rate** – measures delivery speed  
    #     - 📈 **SNR (Signal-to-Noise Ratio)** – checks audio clarity & noise level  

    #     These help assess **speech quality**, **confidence**, and **delivery smoothness**.

    #     ---

    #     ## 🎥 Video Analysis
    #     The video module evaluates non-verbal communication through:
    #     - 🙂 **Face Detection**
    #     - 🧭 **Head Pose (pitch, yaw, roll)**  
    #     - 👀 **Gaze Tracking**  
    #     - 💪 **Body Movement**  
    #     - 🎞 **Frame-based chunk processing**  

    #     These indicate **engagement**, **eye contact**, **posture stability**, and **body language**.

    #     ---

    #     ## 🧠 Multimodal Analysis
    #     Combines audio & video intelligence to generate:
    #     - A unified **presentation score**  
    #     - Weighted or simple fusion options  
    #     - Feedback on both verbal and non-verbal performance  

    #     Offering a **complete, accurate, human-like assessment**.

    #     ---

    #     """, unsafe_allow_html=True)

    def home_page(self):
        st.markdown("""
        <style>
            /* ---- Main Page Styling ---- */
            .premium-header {
                font-size: 3rem;
                text-align: center;
                font-weight: 800;
                color: #ffffff;
                padding: 25px;
                border-radius: 12px;
                background: linear-gradient(135deg, #1f77b4, #6fa8dc);
                animation: fadeIn 1.5s ease;
            }

            .developer-box {
                background: #f0f8ff;
                padding: 12px 20px;
                margin-top: 15px;
                margin-bottom: 25px;
                border-left: 6px solid #1f77b4;
                border-radius: 6px;
                font-size: 1.1rem;
                animation: slideIn 1s ease;
            }

            /* ---- Feature Card ---- */
            .feature-card {
                background: #ffffff;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                transition: transform .3s ease, box-shadow .3s ease;
                animation: fadeUp 1s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
            }

            .feature-title {
                font-size: 1.6rem;
                font-weight: 700;
                color: #1f77b4;
            }

            /* ---- Animations ---- */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            @keyframes fadeUp {
                from { opacity: 0; transform: translateY(15px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            @keyframes slideIn {
                from { opacity: 0; transform: translateX(-15px); }
                to   { opacity: 1; transform: translateX(0); }
            }
        </style>

        <div class="premium-header">✨ Multimodal Presentation Analysis</div>

        <div class="developer-box">
            <strong>Developed by: Sonam Singh</strong><br>
            A smart AI-powered system that evaluates communication skills using Audio, Video, and Multimodal fusion.
        </div>

        """, unsafe_allow_html=True)

    # ---------- FEATURE BOXES ----------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔊 Audio Analysis</div>
                <ul>
                    <li>Volume Stability</li>
                    <li>Pause & Silence Detection</li>
                    <li>Pitch (YIN Algorithm)</li>
                    <li>Speaking Rate</li>
                    <li>SNR for clarity</li>
                </ul>
                <p>Helps evaluate clarity, fluency, and vocal expressiveness.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🎥 Video Analysis</div>
                <ul>
                    <li>Frame & Chunk Processing</li>
                    <li>Face Detection</li>
                    <li>Head Pose Estimation</li>
                    <li>Gaze Tracking</li>
                    <li>Body Movement Analysis</li>
                </ul>
                <p>Assesses engagement, posture, eye contact, and body language.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🧠 Multimodal Fusion</div>
                <ul>
                    <li>Audio + Video Score Fusion</li>
                    <li>Simple & Weighted Methods</li>
                    <li>Final Presentation Score</li>
                    <li>Human-like Performance Feedback</li>
                </ul>
                <p>Provides a complete and balanced evaluation.</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------- Footer ----------
        st.markdown("""
        <br><br>
        <center>
            <p style="opacity:0.7; animation:fadeIn 2s ease;">
                © 2025 Multimodal Presentation Analysis • Created with ❤️ by Sonam Singh
            </p>
        </center>
        """, unsafe_allow_html=True)



    def audio_analysis_page(self):
        st.markdown('<div class="main-header">🎧 Presentation AudioAssess</div>', unsafe_allow_html=True)
        st.write("""
                - Currently helpful for audio parameteres like volume, pause, pitch, and speaking rate.
                - Automatically do every step for audio analysis as you have to select parameters in sidebar then upload the video which support the requirements and then click on run audio analysis.
                """)

        # ---------------- Sidebar: Room area + thresholds (mimic af.py interactive prompts) ----------------
        st.sidebar.markdown("## Select Scoring Parameters ")
        area = st.sidebar.number_input("Enter the room area in sq.ft:", min_value=1.0, value=150.0, step=1.0)

        # Determine defaults based on area (same logic as af.py)
        if area < 100:
            room_type = "small"
            default_min_pause = 0.20  # 200ms for small rooms
            default_long_pause = 0.8  # 800ms for small rooms
            default_opt_min, default_opt_max = 0.5, 1.5
            # default_vol_min, default_vol_max = 55, 70
            default_vol_min, default_vol_max = -70, -55
            default_pitch_min, default_pitch_max = 120, 270
            default_speaking_rate_min, default_speaking_rate_max = 110, 160

        elif 100 <= area <= 400:
            room_type = "medium"
            default_min_pause = 0.25  # 250ms for medium rooms
            default_long_pause = 1.0  # 1 second for medium rooms
            default_opt_min, default_opt_max = 1.0, 2.0
            # default_vol_min, default_vol_max = 65, 80
            default_vol_min, default_vol_max = -80, -65
            default_pitch_min, default_pitch_max = 100, 250
            default_speaking_rate_min, default_speaking_rate_max = 120, 170

        else:
            room_type = "large"
            default_min_pause = 0.30  # 300ms for large rooms
            default_long_pause = 1.2  # 1.2 seconds for large rooms
            default_opt_min, default_opt_max = 1.5, 2.5
            # default_vol_min, default_vol_max = 75, 90
            default_vol_min, default_vol_max = -90, -75
            default_pitch_min, default_pitch_max = 80, 220
            default_speaking_rate_min, default_speaking_rate_max = 130, 180

        st.sidebar.markdown(f"Detected room type: `{room_type}` (area={area} sq.ft)")

        st.sidebar.markdown("### Long pause optimal range (per chunk)")
        optimal_min = st.sidebar.number_input("Optimal minimum (long pauses)", value=float(default_opt_min), step=1.0)
        optimal_max = st.sidebar.number_input("Optimal maximum (long pauses)", value=float(default_opt_max), step=1.0)

        # st.sidebar.markdown("### Pause counting thresholds")
        min_counted_pause = st.sidebar.number_input("Minimum pause duration to count (s)", value=float(default_min_pause), step=0.05)
        long_pause_threshold = st.sidebar.number_input("Long pause threshold (s)", value=float(default_long_pause), step=0.1)

        st.sidebar.markdown("### Volume optimal range (dB)")
        optimal_vol_min = st.sidebar.number_input("Optimal Volume MIN (dB)", value=float(default_vol_min), step=1.0)
        optimal_vol_max = st.sidebar.number_input("Optimal Volume MAX (dB)", value=float(default_vol_max), step=1.0)

        st.sidebar.markdown("### Pitch optimal range (Hz)")   
        optimal_pitch_min = st.sidebar.number_input("Optimal Pitch MIN (Hz)", value=float(default_pitch_min), step=1.0)
        optimal_pitch_max = st.sidebar.number_input("Optimal Pitch MAX (Hz)", value=float(default_pitch_max), step=1.0)

        st.sidebar.markdown("### Speaking Rate optimal range (Hz)")   
        optimal_speaking_rate_min = st.sidebar.number_input("Optimal Speaking Rate MIN (Hz)", value=float(default_speaking_rate_min), step=1.0)
        optimal_speaking_rate_max = st.sidebar.number_input("Optimal  Speaking Rate MAX (Hz)", value=float(default_speaking_rate_max), step=1.0)

        st.sidebar.markdown("### Final scoring options")
        scoring_method = st.sidebar.radio("Final Video Scoring Method:", ["simple", "weighted"])
        if scoring_method == "weighted":
            st.sidebar.markdown("Enter integer weights for features (p, q, r, s)")
            p = st.sidebar.number_input("Weight p (Volume)", min_value=0, value=1, step=1)
            q = st.sidebar.number_input("Weight q (Pause)", min_value=0, value=1, step=1)
            r = st.sidebar.number_input("Weight r (Pitch)", min_value=0, value=1, step=1)
            s = st.sidebar.number_input("Weight s (Speaking Rate)", min_value=0, value=1, step=1)
        else:
            p = q = r = s = 1

        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'], key="audio")

        if uploaded_file is not None:
            st.write({
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB"
            })

            if st.button("🎧 Run Audio Analysis", type="primary", key="extract_audio"):
                with st.spinner("Processing audio (extraction + denoising + SNR + features + scoring)..."):
                    # Save uploaded file
                    video_path = os.path.join(self.temp_dir, uploaded_file.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Step 1: Extract audio
                    audio_path = self.audio_processor.extract_audio_from_video(video_path)
                    if not audio_path or not os.path.exists(audio_path):
                        st.error("❌ Audio extraction failed.")
                        return

                    # Step 2: Noise reduction + SNR
                    clean_path, y_raw, y_clean, sr, snr_before, snr_after, improvement_db = self.audio_processor.reduce_noise(audio_path)
                    if not clean_path or not os.path.exists(clean_path):
                        st.error("❌ Noise reduction failed.")
                        return 

                    # Display basic results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("🎥 Original Video")
                        st.video(video_path)
                        st.markdown("#### 🎧 Original Extracted Audio")
                        with open(audio_path, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/wav")
                        with open(audio_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Original Audio",
                                data=f,
                                file_name=os.path.basename(audio_path),
                                mime="audio/wav"
                            )
                    with col2:
                        st.info("🧼 Noise-Reduced Audio")
                        with open(clean_path, "rb") as clean_file:
                            st.audio(clean_file.read(), format="audio/wav")
                        with open(clean_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Cleaned Audio",
                                data=f,
                                file_name=os.path.basename(clean_path),
                                mime="audio/wav"
                            )

                    # ---- Show SNR metrics ----
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SNR Before", f"{snr_before:.2f}")
                    with col2:
                        st.metric("SNR After", f"{snr_after:.2f}")
                    with col3:
                        st.metric("Improvement (dB)", f"{improvement_db:.2f} dB")

                    # ---- Spectrogram & Waveform (same as before) ----
                    st.markdown("### Spectrogram Comparison (Before vs After)")
                    buf = self.audio_processor.create_spectrograms(y_raw, y_clean, sr)
                    st.image(buf, use_column_width=True, caption="Before & After Noise Reduction Spectrograms")

                    st.markdown("### 📈 Waveform Comparison (Before vs After)")
                    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    axs[0].plot(np.linspace(0, len(y_raw) / sr, len(y_raw)), y_raw, color='red')
                    axs[0].set_title("Before Noise Reduction")
                    axs[0].set_ylabel("Amplitude")
                    axs[1].plot(np.linspace(0, len(y_clean) / sr, len(y_clean)), y_clean, color='green')
                    axs[1].set_title("After Noise Reduction")
                    axs[1].set_xlabel("Time (seconds)")
                    axs[1].set_ylabel("Amplitude")
                    plt.tight_layout()
                    wave_buf = io.BytesIO()
                    plt.savefig(wave_buf, format='png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    wave_buf.seek(0)
                    st.image(wave_buf, use_column_width=True, caption="Waveform Before & After Noise Reduction")

                    # ---- RMS & Centroid visuals omitted for brevity; previous code kept if needed ----

                    # ================== AUDIO CHUNK + FEATURE EXTRACTION (af.py-style) ==================
                    # st.markdown("## 🎚️ Feature Extraction Workflow (Chunk → Pause → Silence Removal → Volume → Pitch → Rate)")

                    # ---- Step 1: Chunk division BEFORE silence removal ----
                    st.markdown("### 🎛️ Audio Chunk Division (5 s fixed, padded last chunk)")
                    chunk_duration = 5  # seconds
                    samples_per_chunk = int(chunk_duration * sr)
                    total_samples = len(y_clean)
                    num_chunks = int(np.ceil(total_samples / samples_per_chunk))
                    st.info(f"Total chunks to be created: {num_chunks}")

                    chunk_dir = os.path.join(self.audio_processor.noise_dir, "chunks_original")
                    os.makedirs(chunk_dir, exist_ok=True)
                    chunk_paths = []

                    for i in range(num_chunks):
                        start = i * samples_per_chunk
                        end = min((i + 1) * samples_per_chunk, total_samples)
                        chunk = y_clean[start:end]

                        # Pad last chunk
                        if len(chunk) < samples_per_chunk:
                            pad_len = samples_per_chunk - len(chunk)
                            chunk = np.pad(chunk, (0, pad_len), mode="constant")

                        chunk_path = os.path.join(chunk_dir, f"chunk_{i+1}.wav")
                        sf.write(chunk_path, chunk, sr)
                        chunk_paths.append(chunk_path)

                    st.success(f"✅ Created {num_chunks} chunks (5 s each, padded).")

                    # ---- Step 2: Per-Chunk Feature Extraction ----
                    st.markdown("### Extracting Features like Pause, Volume, Pitch & Speaking Rate per Chunk")

                    pause_data, volume_data, pitch_data, rate_data = [], [], [], []
                    silence_removed_dir = os.path.join(self.audio_processor.noise_dir, "chunks_silence_removed")
                    os.makedirs(silence_removed_dir, exist_ok=True)

                    silence_thresh = 0.01
                    frame_length = 1024
                    hop_length = 512

                    for i, path in enumerate(chunk_paths):
                        y_chunk, sr = librosa.load(path, sr=None)

                        # ================== PAUSE FEATURES (with silence) ==================
                        rms_full = librosa.feature.rms(y=y_chunk, frame_length=frame_length, hop_length=hop_length)[0]
                        times = librosa.frames_to_time(np.arange(len(rms_full)), sr=sr, hop_length=hop_length)
                        is_silence = rms_full < silence_thresh

                        pauses, start_time = [], None
                        for j, silent in enumerate(is_silence):
                            if silent and start_time is None:
                                start_time = times[j]
                            elif not silent and start_time is not None:
                                end_time = times[j]
                                dur = end_time - start_time
                                if dur >= min_counted_pause:
                                    pauses.append(dur)
                                start_time = None

                        long_pauses = [p for p in pauses if p >= long_pause_threshold]
                        avg_pause_dur = np.mean(pauses) if pauses else 0
                        total_sil = sum(pauses)
                        total_dur = librosa.get_duration(y=y_chunk, sr=sr)
                        silence_pct = (total_sil / total_dur) * 100 if total_dur > 0 else 0

                        pause_data.append({
                            "Chunk": f"chunk_{i+1}",
                            "Total Counted Pauses (>250 ms)": len(pauses),
                            "Long Pauses (>1 s)": len(long_pauses),
                            "Avg Pause Duration (s)": round(avg_pause_dur, 3),
                            "% Silence in Audio": round(silence_pct, 2)
                        })

                        # ================== SILENCE REMOVAL (per chunk) ==================
                        intervals = librosa.effects.split(y_chunk, top_db=25)
                        y_no_sil = np.concatenate([y_chunk[s:e] for s, e in intervals]) if len(intervals) else y_chunk
                        nosil_path = os.path.join(silence_removed_dir, f"chunk_{i+1}_nosil.wav")
                        sf.write(nosil_path, y_no_sil, sr)

                        # ================== VOLUME FEATURES (after silence removal) ==================
                        if len(y_no_sil) == 0:
                            # guard: if silence removal removed everything
                            rms_clean = np.array([0.0])
                        else:
                            rms_clean = librosa.feature.rms(y=y_no_sil, frame_length=frame_length, hop_length=hop_length)[0]

                        avg_rms = float(np.mean(rms_clean)) if rms_clean.size > 0 else 0.0
                        avg_db = 20 * np.log10(avg_rms + 1e-10)

                        volume_data.append({
                            "Chunk": f"chunk_{i+1}",
                            "Avg Volume (RMS)": round(avg_rms, 6),
                            "Avg Volume (dB)": round(avg_db, 2),
                            "Volume Std Dev": round(float(np.std(rms_clean)) if rms_clean.size > 0 else 0.0, 6),
                            "Min Volume (RMS)": round(float(np.min(rms_clean)) if rms_clean.size > 0 else 0.0, 6),
                            "Max Volume (RMS)": round(float(np.max(rms_clean)) if rms_clean.size > 0 else 0.0, 6)
                        })

                        # ================== PITCH FEATURES (using yin) ==================
                        try:
                            # For small chunks, pyin may produce short arrays - using yin to be robust
                            f0 = librosa.yin(y_no_sil, fmin=80, fmax=300, sr=sr)
                            f0_clean = f0[~np.isnan(f0)]
                        except Exception:
                            f0_clean = np.array([])

                        if f0_clean.size > 0:
                            pitch_mean = float(np.mean(f0_clean))
                            pitch_std = float(np.std(f0_clean))
                            pitch_min = float(np.min(f0_clean))
                            pitch_max = float(np.max(f0_clean))
                        else:
                            pitch_mean = pitch_std = pitch_min = pitch_max = 0.0

                        pitch_data.append({
                            "Chunk": f"chunk_{i+1}",
                            "Avg Pitch (Hz)": round(pitch_mean, 2),
                            "Pitch Std Dev": round(pitch_std, 2),
                            "Min Pitch (Hz)": round(pitch_min, 2),
                            "Max Pitch (Hz)": round(pitch_max, 2)
                        })

                        # ================== SPEAKING RATE (onset-based estimation -> WPM) ==================
                        onset_times = librosa.onset.onset_detect(y=y_no_sil, sr=sr, units='time')
                        total_onsets = len(onset_times)
                        chunk_total_dur = total_dur if total_dur > 0 else chunk_duration
                        speaking_rate_wps = total_onsets / chunk_total_dur if chunk_total_dur > 0 else 0
                        speaking_rate_wpm = speaking_rate_wps * 60.0

                        rate_data.append({
                            "Chunk": f"chunk_{i+1}",
                            "Onset Count": total_onsets,
                            "Speaking Rate (words/sec)": round(speaking_rate_wps, 2),
                            "Speaking Rate (words/min)": round(speaking_rate_wpm, 1)
                        })

                    # ---- Step 3: Combine + show ----
                    df_pause = pd.DataFrame(pause_data)
                    df_volume = pd.DataFrame(volume_data)
                    df_pitch = pd.DataFrame(pitch_data)
                    df_rate = pd.DataFrame(rate_data)

                    df_combined = df_volume.merge(df_pause, on="Chunk").merge(df_pitch, on="Chunk").merge(df_rate, on="Chunk")

                    # st.markdown("### 📊 Combined Volume, Pause, Pitch & Speaking Rate Features (per chunk)")
                    st.dataframe(df_combined, use_container_width=True)

                    excel_path = os.path.join(self.audio_processor.noise_dir, "audio_all_features.xlsx")
                    df_combined.to_excel(excel_path, index=False)

                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="📥 Download All Audio Features Excel",
                            data=f,
                            file_name="audio_all_features.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    st.success("All per-chunk features (Pause, Volume, Pitch, Speaking Rate) extracted and saved!")

                    # ================== Scoring Logic (match af.py) ==================
                    st.markdown("## 🧮 Scoring")

                    # -- per-chunk scoring using user parameters (volume/pause/pitch/rate)
                    chunk_scores = []
                    for idx, row in df_combined.iterrows():
                        # Volume score using optimal_vol_min/optimal_vol_max
                        vol_db = float(row["Avg Volume (dB)"])
                        if vol_db < optimal_vol_min:
                            score_vol = 1
                        elif optimal_vol_min <= vol_db <= optimal_vol_max:
                            score_vol = 3
                        else:
                            score_vol = 2

                        # Pause score using long pauses count and optimal_min/optimal_max
                        long_pause_count = int(row.get("Long Pauses (>1 s)", 0))
                        if long_pause_count < optimal_min:
                            score_pause = 1
                        elif optimal_min <= long_pause_count <= optimal_max:
                            score_pause = 3
                        else:
                            score_pause = 2

                        # Pitch score using optimal_pitch_min/optimal_pitch_max
                        avg_pitch = float(row.get("Avg Pitch (Hz)", 0.0))
                        if avg_pitch == 0:
                            score_pitch = 0
                        elif avg_pitch < optimal_pitch_min:
                            score_pitch = 1
                        elif optimal_pitch_min <= avg_pitch <= optimal_pitch_max:
                            score_pitch = 3
                        else:
                            score_pitch = 2

                        # Speaking rate score using room_type specific thresholds (WPM)
                        wpm = float(row.get("Speaking Rate (words/min)", 0.0))                                 
                        if wpm < optimal_speaking_rate_min :
                            score_speaking_rate = 1
                        elif optimal_speaking_rate_min  <= wpm <= optimal_speaking_rate_max :
                            score_speaking_rate = 3
                        else:
                            score_speaking_rate = 2
                        

                        chunk_scores.append({
                            "Chunk": row["Chunk"],
                            "Volume Score (1-3)": score_vol,
                            "Pause Score (1-3)": score_pause,
                            "Pitch Score (1-3)": score_pitch,
                            "Speaking Rate Score (1-3)": score_speaking_rate
                        })

                    df_chunk_scores = pd.DataFrame(chunk_scores)
                    st.markdown("### 🔢 Per-chunk feature scores (1–3)")
                    st.dataframe(df_chunk_scores, use_container_width=True)

                    # -- Video-level averaging of scores per feature
                    video_scores = {
                        "Volume": df_chunk_scores["Volume Score (1-3)"].mean() if not df_chunk_scores.empty else 0,
                        "Pause": df_chunk_scores["Pause Score (1-3)"].mean() if not df_chunk_scores.empty else 0,
                        "Pitch": df_chunk_scores["Pitch Score (1-3)"].mean() if not df_chunk_scores.empty else 0,
                        "Rate": df_chunk_scores["Speaking Rate Score (1-3)"].mean() if not df_chunk_scores.empty else 0
                    }

                    # -- Final video scoring (simple or weighted using p,q,r,s)
                    if scoring_method == "simple":
                        final_score = np.mean(list(video_scores.values()))
                    else:
                        total_w = (p + q + r + s) if (p + q + r + s) > 0 else 1
                        final_score = (
                            p * video_scores["Volume"] + q * video_scores["Pause"] +
                            r * video_scores["Pitch"] + s * video_scores["Rate"]
                        ) / total_w

                    # Percentage mapping similar to af.py
                    percentage = ((final_score - 1) / 2) * 100
                    if final_score <= 1.5:
                        interpretation = "Needs Improvement"
                    elif final_score <= 2.3:
                        interpretation = "Average"
                    else:
                        interpretation = "Excellent"

                    # Show results
                    st.markdown("### 🏁 Final Scoring Result (Video level)")
                    st.write("Per-feature video average scores (1–3):")
                    st.write(pd.DataFrame([video_scores]))
                    st.success(f"Final Video Score (1–3): **{final_score:.2f}** — {interpretation}")
                    st.info(f"Score percentage: {percentage:.2f}%")

                    
                    # Save scoring to excel (append columns)
                    df_final = df_combined.copy()
                    # merge chunk scores to df_final
                    df_final = df_final.merge(df_chunk_scores, on="Chunk", how="left")
                    df_final["Final Chunk Score (avg)"] = df_final[[
                        "Volume Score (1-3)", "Pause Score (1-3)", "Pitch Score (1-3)", "Speaking Rate Score (1-3)"
                    ]].mean(axis=1)

                    score_excel_path = os.path.join(self.audio_processor.noise_dir, "audio_all_features_scored.xlsx")
                    df_final.to_excel(score_excel_path, index=False)

                    with open(score_excel_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Scored Audio Features Excel",
                            data=f,
                            file_name="audio_all_features_scored.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # st.balloons()

                    # ================== PIE CHART OF FEATURE SCORES ==================
                    # st.markdown("### 📊 Feature Contribution Pie Chart")
                    fig, ax = plt.subplots(figsize=(2, 2))
                    labels = list(video_scores.keys())
                    values = list(video_scores.values())
                    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 9})
                    ax.set_title("Average Feature Scores Distribution", fontsize=10)
                    st.pyplot(fig, use_container_width=False)                   

                    st.balloons()

    def video_analysis_page(self):
        st.markdown('<div class="main-header">🎥 Presentation VideoAssess</div>', unsafe_allow_html=True)
        st.write("""
                Automatically do every step for visual analysis as you have to select parameters in sidebar then upload the video which support the requirements and then click on run video analysis.
                """)

        # Sidebar: Frame extraction and chunking settings
        st.sidebar.markdown("### Frame & Chunk Settings")
        frame_rate = st.sidebar.slider("Frames per second to extract", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        resize_w = st.sidebar.number_input("Resize Width", min_value=64, max_value=640, value=224, step=16)
        resize_h = st.sidebar.number_input("Resize Height", min_value=64, max_value=640, value=224, step=16)
        frames_per_chunk = st.sidebar.number_input("Frames per Chunk", min_value=1, max_value=50, value=5, step=1)
        # ----------------------------------
        # USER PARAMETERS FOR SCORING
        # ----------------------------------
        st.sidebar.markdown("### Scoring Parameters")

        head_strict = st.sidebar.number_input("Head strict threshold (°)", min_value=5, max_value=45, value=20, step=1)
        head_moderate = st.sidebar.number_input("Head moderate threshold (°)", min_value=5, max_value=30, value=10, step=1)
        head_fluct = st.sidebar.number_input("Head fluctuation threshold (°)", min_value=1, max_value=10, value=5, step=1)

        attentive_thresh = st.sidebar.slider("Gaze attentive threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.05)
        gaze_fluct = st.sidebar.slider("Gaze fluctuation threshold", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

        body_min = st.sidebar.number_input("Body min movement", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        body_max = st.sidebar.number_input("Body max movement", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        body_fluct = st.sidebar.number_input("Body fluctuation", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

        st.sidebar.markdown("### Final Scoring Method")
        scoring_method = st.sidebar.radio("Choose scoring method:", ["Simple", "Weighted"])
        if scoring_method == "Weighted":
            head_w = st.sidebar.number_input("Weight for Head", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
            gaze_w = st.sidebar.number_input("Weight for Gaze", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            body_w = st.sidebar.number_input("Weight for Body", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        else:
            head_w, gaze_w, body_w = 0.4, 0.3, 0.3


        uploaded_video = st.file_uploader("🎞️ Upload a Video File", type=["mp4", "avi", "mov"], key="video_extract")

        if uploaded_video is not None:
            st.write({
                "Filename": uploaded_video.name,
                "Size (MB)": f"{uploaded_video.size / (1024*1024):.2f}"
            })

            # Save uploaded file temporarily
            video_path = os.path.join(self.temp_dir, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # ✅ Single button for both steps
            if st.button("🚀 Run Video Analysis", type="primary"):
                with st.spinner("Extracting frames, splitting into chunks and give final score... please wait ⏳"):
                    try:
                        # Directories
                        extracted_dir = os.path.join(self.temp_dir, "extracted_frames")
                        chunk_dir = os.path.join(self.temp_dir, "frame_chunks")
                        os.makedirs(extracted_dir, exist_ok=True)
                        os.makedirs(chunk_dir, exist_ok=True)

                        # Step 1: Extract frames
                        frames_folder = extract_frames(
                            video_path=video_path,
                            output_dir=extracted_dir,
                            frame_rate=frame_rate,
                            resize_dim=(resize_w, resize_h)
                        )
                        st.success("✅ Frames extracted successfully!")

                        
                        # Step 2: Split into chunks
                        chunk_folder = split_into_chunks(
                            input_dir=extracted_dir,
                            output_dir=chunk_dir,
                            frames_per_chunk=frames_per_chunk
                        )
                        st.success("✅ Frames successfully split into chunks!")

                        # Step 3: Face detection
                        st.info("🔍 Detecting faces from frame chunks...")
                        faces_dir = os.path.join(self.temp_dir, "detected_faces")
                        os.makedirs(faces_dir, exist_ok=True)

                        faces_folder = detect_and_save_faces(
                            input_dir=chunk_dir,
                            output_dir=faces_dir,
                            resize_dim=(resize_w, resize_h)
                        )
                        st.success("✅ Face detection completed!")


                        # -------------------------------
                        # Show few sample frames
                        # -------------------------------
                        video_subfolder = os.path.join(extracted_dir, os.path.splitext(os.path.basename(video_path))[0])
                        frame_files = sorted([
                            os.path.join(video_subfolder, f)
                            for f in os.listdir(video_subfolder)
                            if f.endswith(".jpg")
                        ])

                        if frame_files:
                            st.markdown("### 📸 Sample Extracted Frames")
                            st.image(frame_files[:5], width=200, caption=[os.path.basename(f) for f in frame_files[:5]])

                        # Count total chunks
                        total_chunks = 0
                        for root, dirs, _ in os.walk(chunk_dir):
                            for d in dirs:
                                if d.startswith("chunk_"):
                                    total_chunks += 1
                        st.info(f"Total Chunks Created: **{total_chunks}**")
                        # -------------------------------
                        # Create ZIP downloads
                        # -------------------------------
                        frames_zip = os.path.join(self.temp_dir, "frames_extracted.zip")
                        with zipfile.ZipFile(frames_zip, "w") as zipf:
                            for root, _, files in os.walk(frames_folder):
                                for file in files:
                                    zipf.write(os.path.join(root, file),
                                               os.path.relpath(os.path.join(root, file), frames_folder))

                        chunks_zip = os.path.join(self.temp_dir, "chunks_output.zip")
                        with zipfile.ZipFile(chunks_zip, "w") as zipf:
                            for root, _, files in os.walk(chunk_folder):
                                for file in files:
                                    zipf.write(os.path.join(root, file),
                                               os.path.relpath(os.path.join(root, file), chunk_folder))
                        # Display download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(frames_zip, "rb") as f:
                                st.download_button(
                                    label="📦 Download All Frames (ZIP)",
                                    data=f,
                                    file_name="frames_extracted.zip",
                                    mime="application/zip"
                                )
                        with col2:
                            with open(chunks_zip, "rb") as f:
                                st.download_button(
                                    label="📦 Download Frame Chunks (ZIP)",
                                    data=f,
                                    file_name="frame_chunks.zip",
                                    mime="application/zip"
                                )

                        # -------------------------------
                        # Show detected faces (samples)
                        # -------------------------------
                        face_samples = []
                        for root, _, files in os.walk(faces_dir):
                            for f in files:
                                if f.endswith(".jpg"):
                                    face_samples.append(os.path.join(root, f))
                            if len(face_samples) >= 5:
                                break

                        if face_samples:
                            st.markdown("### 😃 Sample Detected Faces")
                            st.image(face_samples[:5], width=200, caption=[os.path.basename(f) for f in face_samples[:5]])

                        # Create ZIP for faces
                        faces_zip = os.path.join(self.temp_dir, "detected_faces.zip")
                        with zipfile.ZipFile(faces_zip, "w") as zipf:
                            for root, _, files in os.walk(faces_dir):
                                for file in files:
                                    zipf.write(
                                        os.path.join(root, file),
                                        os.path.relpath(os.path.join(root, file), faces_dir)
                                    )

                        with open(faces_zip, "rb") as f:
                            st.download_button(
                                label="📦 Download All Detected Faces (ZIP)",
                                data=f,
                                file_name="detected_faces.zip",
                                mime="application/zip"
                            )

                        # Step 4: Head pose & gaze feature extraction
                        st.info("🧠 Extracting head pose, gaze & Body features...")
                        try: 
                            head_gaze_features = extract_head_gaze_features(faces_dir)
                            body_features = extract_body_features(chunk_dir)
                            st.success("✅ Head, Gaze & Body feature extraction completed!")

                            # Convert to DataFrame
                            df_head_gaze = pd.DataFrame(head_gaze_features)
                            df_body = pd.DataFrame(body_features)
                            st.markdown("### 📊 Head, Gaze & Body Feature Extraction (Per Chunk)")
                            # st.dataframe(df_head_gaze, df_body, use_container_width=True)


                            # Save to Excel
                            features_excel = os.path.join(self.temp_dir, "head_gaze_features.xlsx")
                            df_head_gaze.to_excel(features_excel, index=False)
                            df_combined = pd.merge(df_head_gaze, df_body, on=["video_name", "chunk"], how="outer")
                            st.dataframe(df_combined, use_container_width=True)
                            # Save combined features to Excel
                            combined_excel = os.path.join(self.temp_dir, "video_features_combined.xlsx")
                            df_combined.to_excel(combined_excel, index=False)

                            with open(combined_excel, "rb") as f:
                                st.download_button(
                                    label="📥 Download Features (Excel)",
                                    data=f,
                                    file_name="video_features_combined.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                        except Exception as e:
                           st.error(f"❌ Body feature extraction failed: {e}")

                           # Step 6: Scoring per chunk + overall video
                        st.info("🧮 Calculating scores for Head, Gaze & Body...")

                        try:
                            # Per-chunk scoring
                            df_combined["head_score"], df_combined["head_feedback"] = zip(*df_combined.apply(
                                lambda row: head_score(row["pitch_mean"], row["yaw_mean"],
                                    head_strict=head_strict,
                                    head_moderate=head_moderate,
                                    fluctuation_thresh=head_fluct),
                                axis=1
                            ))

                            df_combined["gaze_score"], df_combined["gaze_feedback"] = zip(*df_combined.apply(
                                lambda row: gaze_score(row["horiz_mean"], row["vert_mean"],
                                    attentive_thresh=attentive_thresh,
                                    fluctuation_thresh=gaze_fluct),
                                axis=1
                            ))

                            df_combined["body_score"], df_combined["body_feedback"] = zip(*df_combined.apply(
                                lambda row: body_score(row["body_mean"],
                                    body_min=body_min,
                                    body_max=body_max,
                                    fluctuation_thresh=body_fluct),
                                axis=1
                            ))

                            st.success("✅ Per-chunk scoring completed!")

                            # Show per-chunk score table
                            st.markdown("### 🧩 Per-Chunk Scores (1–3)")
                            st.dataframe(df_combined[[
                                "video_name", "chunk", "head_score", "gaze_score", "body_score"
                            ]], use_container_width=True)

                            # Calculate final video-level scores
                            weights = {"head": head_w, "gaze": gaze_w, "body": body_w}
                            final_scores = calculate_final_video_score(
                                df_combined,
                                method="weighted" if scoring_method == "Weighted" else "simple",
                                weights=weights
                            )

                            # Display final result
                            for vid, val in final_scores.items():
                                st.markdown(f"## 🎬 Final Video Score for `{vid}`")
                                st.write(f"**Overall Score:** {val['final_score']}/3 — {val['feedback']}")
                                st.write("**Component Scores:**", val["components"])

                            # Save scored table
                            scored_excel = os.path.join(self.temp_dir, "video_features_scored.xlsx")
                            df_combined.to_excel(scored_excel, index=False)

                            with open(scored_excel, "rb") as f:
                                st.download_button(
                                    label="📥 Download Per Chunk score with feedback (Excel)",
                                    data=f,
                                    file_name="video_features_scored.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                        except Exception as e:
                                st.error(f"❌ Scoring failed: {e}")


                    except Exception as e:
                        st.error(f"❌ Video analysis failed: {e}")

    def multimodal_analysis_page(self):
        st.markdown('<div class="main-header">🧠 Multimodal Analysis</div>', unsafe_allow_html=True)
        st.write("""
        - This module automatically performs **complete preprocessing** of both audio and video:
        - You have to select parameters in sidebar then upload the video which support the requirements and then click on run multimodal analysis.          
        """)
        st.sidebar.markdown("### Audio Scoring Settings")
        area = st.sidebar.number_input("Enter the room area in sq.ft:", min_value=1.0, value=150.0, step=1.0)

        # Determine defaults based on area (same logic as af.py)
        if area < 100:
            room_type = "small"
            default_min_pause = 0.20  # 200ms for small rooms
            default_long_pause = 0.8  # 800ms for small rooms
            default_opt_min, default_opt_max = 0.5, 1.5
            # default_vol_min, default_vol_max = 55, 70
            default_vol_min, default_vol_max = -70, -55
            default_pitch_min, default_pitch_max = 120, 270
            default_speaking_rate_min, default_speaking_rate_max = 110, 160
        elif 100 <= area <= 400:
            room_type = "medium"
            default_min_pause = 0.25  # 250ms for medium rooms
            default_long_pause = 1.0  # 1 second for medium rooms
            default_opt_min, default_opt_max = 1.0, 2.0
            # default_vol_min, default_vol_max = 65, 80
            default_vol_min, default_vol_max = -80, -65
            default_pitch_min, default_pitch_max = 100, 250
            default_speaking_rate_min, default_speaking_rate_max = 120, 170
        else:
            room_type = "large"
            default_min_pause = 0.30  # 300ms for large rooms
            default_long_pause = 1.2  # 1.2 seconds for large rooms
            default_opt_min, default_opt_max = 1.5, 2.5
            # default_vol_min, default_vol_max = 75, 90
            default_vol_min, default_vol_max = -90, -75
            default_pitch_min, default_pitch_max = 80, 220
            default_speaking_rate_min, default_speaking_rate_max = 130, 180

        st.sidebar.markdown(f"Detected room type: `{room_type}` (area={area} sq.ft)")
        optimal_min = st.sidebar.number_input("Optimal minimum (long pauses)", value=float(default_opt_min), step=1.0)
        optimal_max = st.sidebar.number_input("Optimal maximum (long pauses)", value=float(default_opt_max), step=1.0)
        min_counted_pause = st.sidebar.number_input("Minimum pause duration to count (s)", value=float(default_min_pause), step=0.05)
        long_pause_threshold = st.sidebar.number_input("Long pause threshold (s)", value=float(default_long_pause), step=0.1)
        optimal_vol_min = st.sidebar.number_input("Optimal Volume MIN (dB)", value=float(default_vol_min), step=1.0)
        optimal_vol_max = st.sidebar.number_input("Optimal Volume MAX (dB)", value=float(default_vol_max), step=1.0)   
        optimal_pitch_min = st.sidebar.number_input("Optimal Pitch MIN (Hz)", value=float(default_pitch_min), step=1.0)
        optimal_pitch_max = st.sidebar.number_input("Optimal Pitch MAX (Hz)", value=float(default_pitch_max), step=1.0)  
        optimal_speaking_rate_min = st.sidebar.number_input("Optimal Speaking Rate MIN (Hz)", value=float(default_speaking_rate_min), step=1.0)
        optimal_speaking_rate_max = st.sidebar.number_input("Optimal  Speaking Rate MAX (Hz)", value=float(default_speaking_rate_max), step=1.0)

        st.sidebar.markdown("### Video Parameters Settings")
       
        frame_rate = st.sidebar.slider("Frames per second to extract", min_value=0.5, max_value=5.0, value=1.0, step=0.5)                
        resize_w = st.sidebar.number_input("Resize Width", min_value=64, max_value=640, value=224, step=16)
        resize_h = st.sidebar.number_input("Resize Height", min_value=64, max_value=640, value=224, step=16)
        frames_per_chunk = st.sidebar.number_input("Frames per Chunk", min_value=1, max_value=50, value=5, step=1)

        head_strict = st.sidebar.number_input("Head strict threshold (°)", min_value=5, max_value=45, value=20, step=1)
        head_moderate = st.sidebar.number_input("Head moderate threshold (°)", min_value=5, max_value=30, value=10, step=1)
        head_fluct = st.sidebar.number_input("Head fluctuation threshold (°)", min_value=1, max_value=10, value=5, step=1)

        attentive_thresh = st.sidebar.slider("Gaze attentive threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.05)
        gaze_fluct = st.sidebar.slider("Gaze fluctuation threshold", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

        body_min = st.sidebar.number_input("Body min movement", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        body_max = st.sidebar.number_input("Body max movement", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        body_fluct = st.sidebar.number_input("Body fluctuation", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

        st.sidebar.markdown("### Audio scoring Method")
        scoring_method = st.sidebar.radio("Final Video Scoring Method:", ["simple", "weighted"])
        if scoring_method == "weighted":
            st.sidebar.markdown("Enter integer weights for features (p, q, r, s)")
            p = st.sidebar.number_input("Weight p (Volume)", min_value=0, value=1, step=1)
            q = st.sidebar.number_input("Weight q (Pause)", min_value=0, value=1, step=1)
            r = st.sidebar.number_input("Weight r (Pitch)", min_value=0, value=1, step=1)
            s = st.sidebar.number_input("Weight s (Speaking Rate)", min_value=0, value=1, step=1)
        else:
            p = q = r = s = 1

        st.sidebar.markdown("### Video Scoring Method")
        scoring_method = st.sidebar.radio("Choose scoring method:", ["Simple", "Weighted"])
        if scoring_method == "Weighted":
            head_w = st.sidebar.number_input("Weight for Head", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
            gaze_w = st.sidebar.number_input("Weight for Gaze", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            body_w = st.sidebar.number_input("Weight for Body", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        else:
            head_w, gaze_w, body_w = 0.4, 0.3, 0.3

        st.sidebar.markdown("### Multimodal Scoring Method")
        scoring_mode = st.sidebar.radio("Choose Final PPT Scoring Method:", ["simple", "weighted"], key="ppt_score_method")
        if scoring_mode == "weighted":
            audio_w = st.sidebar.number_input("Weight for Audio", min_value=0, value=1, step=1)
            video_w = st.sidebar.number_input("Weight for Video", min_value=0, value=1, step=1)
            weights = {"audio": audio_w, "video": video_w}
        else:
            weights = None

        uploaded_file = st.file_uploader("📤 Upload your presentation video", type=["mp4", "mov", "avi"], key="multimodal")

        if uploaded_file is not None:
            st.write({
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB"
            })

        # Save uploaded video
            video_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("🚀 Run Multimodal Analysis", type="primary"):
                with st.spinner("Running preprocessing pipeline... please wait ⏳"):
                    try:
#=================================================================================================================================================================================================
# ===='''''''''''''''''''''''''''''''...............................................🎧 AUDIO PREPROCESSING..........................''''''''''''''''=============================================
# ==================================================================================================================================================================================================
                        st.subheader("🎧 Step 1: Audio Preprocessing")
                        st.info("Extracting audio from video...")

                        audio_path = self.audio_processor.extract_audio_from_video(video_path)
                        if not audio_path or not os.path.exists(audio_path):
                            st.error("❌ Audio extraction failed.")
                            return

                        st.info("Applying noise reduction...")
                        clean_path, y_raw, y_clean, sr, snr_before, snr_after, improvement_db = self.audio_processor.reduce_noise(audio_path)
                        if not clean_path or not os.path.exists(clean_path):
                            st.error("❌ Noise reduction failed.")
                            return 
                        
                        # Display basic results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info("🎥 Original Video")
                            st.video(video_path)
                            st.markdown("#### 🎧 Original Extracted Audio")
                            with open(audio_path, "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/wav")
                            with open(audio_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Original Audio",
                                    data=f,
                                    file_name=os.path.basename(audio_path),
                                    mime="audio/wav"
                                )
                        with col2:
                            st.info("🧼 Noise-Reduced Audio")
                            with open(clean_path, "rb") as clean_file:
                                st.audio(clean_file.read(), format="audio/wav")
                            with open(clean_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Cleaned Audio",
                                    data=f,
                                    file_name=os.path.basename(clean_path),
                                    mime="audio/wav"
                                )
                        # ---- Show SNR metrics ----
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("SNR Before", f"{snr_before:.2f}")
                        with col2:
                            st.metric("SNR After", f"{snr_after:.2f}")
                        with col3:
                            st.metric("Improvement (dB)", f"{improvement_db:.2f} dB")

                        # ---- Spectrogram & Waveform (same as before) ----
                        st.markdown("### Spectrogram Comparison (Before vs After)")
                        buf = self.audio_processor.create_spectrograms(y_raw, y_clean, sr)
                        st.image(buf, use_column_width=True, caption="Before & After Noise Reduction Spectrograms")

                        st.markdown("### 📈 Waveform Comparison (Before vs After)")
                        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                        axs[0].plot(np.linspace(0, len(y_raw) / sr, len(y_raw)), y_raw, color='red')
                        axs[0].set_title("Before Noise Reduction")
                        axs[0].set_ylabel("Amplitude")
                        axs[1].plot(np.linspace(0, len(y_clean) / sr, len(y_clean)), y_clean, color='green')
                        axs[1].set_title("After Noise Reduction")
                        axs[1].set_xlabel("Time (seconds)")
                        axs[1].set_ylabel("Amplitude")
                        plt.tight_layout()
                        wave_buf = io.BytesIO()
                        plt.savefig(wave_buf, format='png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        wave_buf.seek(0)
                        st.image(wave_buf, use_column_width=True, caption="Waveform Before & After Noise Reduction")

                        # ---- Split into 5-second chunks ----
                        st.info("Splitting cleaned audio into 5-second chunks...")
                        chunk_duration = 5  # seconds
                        samples_per_chunk = int(chunk_duration * sr)
                        total_samples = len(y_clean)
                        num_chunks = int(np.ceil(total_samples / samples_per_chunk))
                        st.info(f"Total chunks to be created: {num_chunks}")

                        chunk_dir = os.path.join(self.audio_processor.noise_dir, "chunks_for_multimodal")
                        os.makedirs(chunk_dir, exist_ok=True)
                        chunk_paths = []

                        for i in range(num_chunks):
                            start = i * samples_per_chunk
                            end = min((i + 1) * samples_per_chunk, total_samples)
                            chunk = y_clean[start:end]

                            if len(chunk) < samples_per_chunk:
                                pad_len = samples_per_chunk - len(chunk)
                                chunk = np.pad(chunk, (0, pad_len), mode="constant")

                            chunk_path = os.path.join(chunk_dir, f"chunk_{i+1:03d}.wav")
                            sf.write(chunk_path, chunk, sr)
                            chunk_paths.append(chunk_path)
# ================================================== ==========================================================================================================
# '''''''''''''''''''''''''''''''''''.................................🎥 VIDEO PREPROCESSING...........................'''''''''''''''''''''''''''''''''''''''
# ===============================================================================================================================================                       
                        st.subheader("🎥 Step 2: Video Preprocessing")
                        st.info("Extracting frames from video...")

                        frame_dir = os.path.join(self.temp_dir, "frames_mm")
                        chunked_frame_dir = os.path.join(self.temp_dir, "video_chunks_mm")
                        faces_dir = os.path.join(self.temp_dir, "faces_mm")
                        os.makedirs(frame_dir, exist_ok=True)
                        os.makedirs(chunked_frame_dir, exist_ok=True)
                        os.makedirs(faces_dir, exist_ok=True)

                        # ---- Frame extraction ----
                        frames_path = extract_frames(video_path, frame_dir, frame_rate=frame_rate, resize_dim=(resize_w, resize_h))
                        
                        # ---- Split frames into chunks ----
                        st.info("Splitting frames into chunks ...")
                        chunks_path = split_into_chunks(frame_dir, chunked_frame_dir, frames_per_chunk=frames_per_chunk)   

                        # ---- Face detection ----
                        st.info("Detecting faces in chunks...")
                        faces_path = detect_and_save_faces(chunked_frame_dir, faces_dir, resize_dim=(resize_w, resize_h))

                        m_video_subfolder = os.path.join(frame_dir, os.path.splitext(os.path.basename(video_path))[0])
                        m_frame_files = sorted([
                            os.path.join(m_video_subfolder, f)
                            for f in os.listdir(m_video_subfolder)
                            if f.endswith(".jpg")
                        ])

                        if m_frame_files:
                            st.markdown("### 📸 Sample Extracted Frames")
                            st.image(m_frame_files[:5], width=200, caption=[os.path.basename(f) for f in m_frame_files[:5]])

                        # Count total chunks
                        total_chunks = 0
                        for root, dirs, _ in os.walk( chunked_frame_dir):
                            for d in dirs:
                                if d.startswith("chunk_"):
                                    total_chunks += 1
                        st.info(f"Total Chunks Created: **{total_chunks}**")

                        # Create ZIP downloads
                        m_frames_zip = os.path.join(self.temp_dir, "frames_extracted.zip")
                        with zipfile.ZipFile(m_frames_zip, "w") as zipf:
                            for root, _, files in os.walk(frames_path):
                                for file in files:
                                    zipf.write(os.path.join(root, file),
                                               os.path.relpath(os.path.join(root, file), frames_path))

                        chunks_zip = os.path.join(self.temp_dir, "chunks_output.zip")
                        with zipfile.ZipFile(chunks_zip, "w") as zipf:
                            for root, _, files in os.walk(chunks_path):
                                for file in files:
                                    zipf.write(os.path.join(root, file),
                                               os.path.relpath(os.path.join(root, file), chunks_path))

                        # Display download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(m_frames_zip, "rb") as f:
                                st.download_button(
                                    label="📦 Download All Frames (ZIP)",
                                    data=f,
                                    file_name="frames_extracted.zip",
                                    mime="application/zip"
                                )
                        with col2:
                            with open(chunks_zip, "rb") as f:
                                st.download_button(
                                    label="📦 Download Frame Chunks (ZIP)",
                                    data=f,
                                    file_name="frame_chunks.zip",
                                    mime="application/zip"
                                )

                        # Show detected faces (samples)
                        m_face_samples = []
                        for root, _, files in os.walk(faces_dir):
                            for f in files:
                                if f.endswith(".jpg"):
                                    m_face_samples.append(os.path.join(root, f))
                            if len(m_face_samples) >= 5:
                                break

                        if m_face_samples:
                            st.markdown("### 😃 Sample Detected Faces")
                            st.image(m_face_samples[:5], width=200, caption=[os.path.basename(f) for f in m_face_samples[:5]])
                        # Create ZIP for faces
                        m_faces_zip = os.path.join(self.temp_dir, "detected_faces.zip")
                        with zipfile.ZipFile(m_faces_zip, "w") as zipf:
                            for root, _, files in os.walk(faces_dir):
                                for file in files:
                                    zipf.write(
                                        os.path.join(root, file),
                                        os.path.relpath(os.path.join(root, file), faces_dir)
                                    )

                        with open(m_faces_zip, "rb") as f:
                            st.download_button(
                                label="📦 Download All Detected Faces (ZIP)",
                                data=f,
                                file_name="detected_faces.zip",
                                mime="application/zip"
                            )
# ======================================================================================================================================================
# '''''''''''''''''''''''''''''''''''''''''''''...........................Audio Feature.............................'''''''''''''''''''''''''''''''''
# =================================================================================================================================================
                        st.info("🧠 Extracting Audio features...")
                        m_pause_data, m_volume_data, m_pitch_data, m_rate_data = [], [], [], []
                        m_silence_removed_dir = os.path.join(self.audio_processor.noise_dir, "chunks_silence_removed")
                        os.makedirs(m_silence_removed_dir, exist_ok=True)

                        silence_thresh = 0.01
                        frame_length = 1024
                        hop_length = 512

                        for i, path in enumerate(chunk_paths):
                            y_chunk, sr = librosa.load(path, sr=None)

                        # ================== PAUSE FEATURES (with silence) ==================
                            rms_full = librosa.feature.rms(y=y_chunk, frame_length=frame_length, hop_length=hop_length)[0]
                            times = librosa.frames_to_time(np.arange(len(rms_full)), sr=sr, hop_length=hop_length)
                            is_silence = rms_full < silence_thresh

                            pauses, start_time = [], None
                            for j, silent in enumerate(is_silence):
                                if silent and start_time is None:
                                    start_time = times[j]
                                elif not silent and start_time is not None:
                                    end_time = times[j]
                                    dur = end_time - start_time
                                    if dur >= min_counted_pause:
                                        pauses.append(dur)
                                    start_time = None

                            long_pauses = [p for p in pauses if p >= long_pause_threshold]
                            avg_pause_dur = np.mean(pauses) if pauses else 0
                            total_sil = sum(pauses)
                            total_dur = librosa.get_duration(y=y_chunk, sr=sr)
                            silence_pct = (total_sil / total_dur) * 100 if total_dur > 0 else 0

                            m_pause_data.append({
                                "Chunk": f"chunk_{i+1}",
                                "Total Counted Pauses (>250 ms)": len(pauses),
                                "Long Pauses (>1 s)": len(long_pauses),
                                "Avg Pause Duration (s)": round(avg_pause_dur, 3),
                                "% Silence in Audio": round(silence_pct, 2)
                            })

                            # ================== SILENCE REMOVAL (per chunk) ==================
                            intervals = librosa.effects.split(y_chunk, top_db=25)
                            y_no_sil = np.concatenate([y_chunk[s:e] for s, e in intervals]) if len(intervals) else y_chunk
                            nosil_path = os.path.join(m_silence_removed_dir, f"chunk_{i+1}_nosil.wav")
                            sf.write(nosil_path, y_no_sil, sr)

                            # ================== VOLUME FEATURES (after silence removal) ==================
                            if len(y_no_sil) == 0:
                            # guard: if silence removal removed everything
                                rms_clean = np.array([0.0])
                            else:
                                rms_clean = librosa.feature.rms(y=y_no_sil, frame_length=frame_length, hop_length=hop_length)[0]

                            avg_rms = float(np.mean(rms_clean)) if rms_clean.size > 0 else 0.0
                            avg_db = 20 * np.log10(avg_rms + 1e-10)

                            m_volume_data.append({
                                "Chunk": f"chunk_{i+1}",
                                "Avg Volume (RMS)": round(avg_rms, 6),
                                "Avg Volume (dB)": round(avg_db, 2),
                                "Volume Std Dev": round(float(np.std(rms_clean)) if rms_clean.size > 0 else 0.0, 6),
                                "Min Volume (RMS)": round(float(np.min(rms_clean)) if rms_clean.size > 0 else 0.0, 6),
                                "Max Volume (RMS)": round(float(np.max(rms_clean)) if rms_clean.size > 0 else 0.0, 6)
                            })

                            # ================== PITCH FEATURES (using yin) ==================
                            try:
                                f0 = librosa.yin(y_no_sil, fmin=80, fmax=300, sr=sr)
                                f0_clean = f0[~np.isnan(f0)]
                            except Exception:
                                f0_clean = np.array([])

                            if f0_clean.size > 0:
                                pitch_mean = float(np.mean(f0_clean))
                                pitch_std = float(np.std(f0_clean))
                                pitch_min = float(np.min(f0_clean))
                                pitch_max = float(np.max(f0_clean))
                            else:
                                pitch_mean = pitch_std = pitch_min = pitch_max = 0.0

                            m_pitch_data.append({
                                "Chunk": f"chunk_{i+1}",
                                "Avg Pitch (Hz)": round(pitch_mean, 2),
                                "Pitch Std Dev": round(pitch_std, 2),
                                "Min Pitch (Hz)": round(pitch_min, 2),
                                "Max Pitch (Hz)": round(pitch_max, 2)
                            })

                            # ================== SPEAKING RATE (onset-based estimation -> WPM) ==================
                            onset_times = librosa.onset.onset_detect(y=y_no_sil, sr=sr, units='time')
                            total_onsets = len(onset_times)
                            chunk_total_dur = total_dur if total_dur > 0 else chunk_duration
                            speaking_rate_wps = total_onsets / chunk_total_dur if chunk_total_dur > 0 else 0
                            speaking_rate_wpm = speaking_rate_wps * 60.0

                            m_rate_data.append({
                                "Chunk": f"chunk_{i+1}",
                                "Onset Count": total_onsets,
                                "Speaking Rate (words/sec)": round(speaking_rate_wps, 2),
                                "Speaking Rate (words/min)": round(speaking_rate_wpm, 1)
                            })

                        # ---- Step 3: Combine + show ----
                        df_pause = pd.DataFrame(m_pause_data)
                        df_volume = pd.DataFrame(m_volume_data)
                        df_pitch = pd.DataFrame(m_pitch_data)
                        df_rate = pd.DataFrame(m_rate_data)

                        df_combined = df_volume.merge(df_pause, on="Chunk").merge(df_pitch, on="Chunk").merge(df_rate, on="Chunk")

                        # st.markdown("### 📊 Combined Volume, Pause, Pitch & Speaking Rate Features (per chunk)")
                        st.dataframe(df_combined, use_container_width=True)

                        excel_path = os.path.join(self.audio_processor.noise_dir, "audio_all_features.xlsx")
                        df_combined.to_excel(excel_path, index=False)

                        with open(excel_path, "rb") as f:
                            st.download_button(
                                label="📥 Download All Audio Features Excel",
                                data=f,
                                file_name="audio_all_features.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
# ================================================================================================================================================
# ''''''''''''''''''''.................................'video feature extraction with scoring.................................'''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ============================================================================================================================================================
                        st.info("🧠 Extracting Video features...")
                        # try: 
                        m_head_gaze_features = extract_head_gaze_features(faces_dir)
                        m_body_features = extract_body_features(chunked_frame_dir)
                        # Convert to DataFrame
                        df_head_gaze = pd.DataFrame(m_head_gaze_features)
                        df_body = pd.DataFrame(m_body_features)
                        if df_head_gaze.empty:
                            st.warning("⚠️ No head/gaze features extracted — likely no faces detected.")
                            df_head_gaze = pd.DataFrame(columns=["video_name", "chunk", "pitch_mean", "yaw_mean", "roll_mean", "horiz_mean", "vert_mean"])

                        if df_body.empty:
                            st.warning("⚠️ No body features extracted — likely no human body detected.")
                            df_body = pd.DataFrame(columns=["video_name", "chunk", "body_mean"])
                        # Save to Excel
                        features_excel = os.path.join(self.temp_dir, "head_gaze_features.xlsx")
                        df_head_gaze.to_excel(features_excel, index=False)
                        df_body.to_excel(features_excel, index=False)
                        m_df_combined = pd.merge(df_head_gaze, df_body, on=["video_name", "chunk"], how="outer")
                        st.dataframe(m_df_combined, use_container_width=True)
                        # Save combined features to Excel
                        combined_excel = os.path.join(self.temp_dir, "video_features_combined.xlsx")
                        m_df_combined.to_excel(combined_excel, index=False)
                        with open(combined_excel, "rb") as f:
                            st.download_button(
                                label="📥 Download Features (Excel)",
                                data=f,
                                file_name="video_features_combined.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

# ===================================================================================================================================================================================================
# ''''''''''''''''''''''''''''''''''''''''''''''''''.........................AUDIO SCORING......................................'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ===================================================================================================================================================================================================
                        st.markdown("## 🧮 Audio Features Scoring Per Chunk")
                        chunk_scores = []
                        for idx, row in df_combined.iterrows():
                            # Volume score using optimal_vol_min/optimal_vol_max
                            vol_db = float(row["Avg Volume (dB)"])
                            if vol_db < optimal_vol_min:
                                score_vol = 1
                            elif optimal_vol_min <= vol_db <= optimal_vol_max:
                                score_vol = 3
                            else:
                                score_vol = 2
                            # Pause score using long pauses count and optimal_min/optimal_max
                            long_pause_count = int(row.get("Long Pauses (>1 s)", 0))
                            if long_pause_count < optimal_min:
                                score_pause = 1
                            elif optimal_min <= long_pause_count <= optimal_max:
                                score_pause = 3
                            else:
                                score_pause = 2
                            # Pitch score using optimal_pitch_min/optimal_pitch_max
                            avg_pitch = float(row.get("Avg Pitch (Hz)", 0.0))
                            if avg_pitch == 0:
                                score_pitch = 0
                            elif avg_pitch < optimal_pitch_min:
                                score_pitch = 1
                            elif optimal_pitch_min <= avg_pitch <= optimal_pitch_max:
                                score_pitch = 3
                            else:
                                score_pitch = 2
                            # Speaking rate score using room_type specific thresholds (WPM)
                            wpm = float(row.get("Speaking Rate (words/min)", 0.0))                                 
                            if wpm < optimal_speaking_rate_min :
                                score_speaking_rate = 1
                            elif optimal_speaking_rate_min  <= wpm <= optimal_speaking_rate_max :
                                score_speaking_rate = 3
                            else:
                                score_speaking_rate = 2

                            chunk_scores.append({
                                "Chunk": row["Chunk"],
                                "Volume Score (1-3)": score_vol,
                                "Pause Score (1-3)": score_pause,
                                "Pitch Score (1-3)": score_pitch,
                                "Speaking Rate Score (1-3)": score_speaking_rate
                            })

                        df_chunk_scores = pd.DataFrame(chunk_scores)
                        st.dataframe(df_chunk_scores, use_container_width=True)
                        # -- Video-level averaging of scores per feature
                        video_scores = {
                            "Volume": df_chunk_scores["Volume Score (1-3)"].mean() if not df_chunk_scores.empty else 0,
                            "Pause": df_chunk_scores["Pause Score (1-3)"].mean() if not df_chunk_scores.empty else 0,
                            "Pitch": df_chunk_scores["Pitch Score (1-3)"].mean() if not df_chunk_scores.empty else 0,
                            "Rate": df_chunk_scores["Speaking Rate Score (1-3)"].mean() if not df_chunk_scores.empty else 0
                        }
                        # -- Final video scoring (simple or weighted using p,q,r,s)
                        if scoring_method == "simple":
                            final_score = np.mean(list(video_scores.values()))
                        else:
                            total_w = (p + q + r + s) if (p + q + r + s) > 0 else 1
                            final_score = (
                                p * video_scores["Volume"] + q * video_scores["Pause"] +
                                r * video_scores["Pitch"] + s * video_scores["Rate"]
                            ) / total_w
                        # Percentage mapping 
                        percentage = ((final_score - 1) / 2) * 100
                        if final_score <= 1.5:
                            interpretation = "Needs Improvement"
                        elif final_score <= 2.3:
                            interpretation = "Average"
                        else:
                            interpretation = "Excellent"
                        # Show results
                        st.write("Per-feature audio average scores (1–3):")
                        st.write(pd.DataFrame([video_scores]))

                        # Save scoring to excel (append columns)
                        df_final = df_combined.copy()
                            # merge chunk scores to df_final
                        df_final = df_final.merge(df_chunk_scores, on="Chunk", how="left")
                        df_final["Final Chunk Score (avg)"] = df_final[[
                            "Volume Score (1-3)", "Pause Score (1-3)", "Pitch Score (1-3)", "Speaking Rate Score (1-3)"
                        ]].mean(axis=1)

                        score_excel_path = os.path.join(self.audio_processor.noise_dir, "audio_all_features_scored.xlsx")
                        df_final.to_excel(score_excel_path, index=False)
                        with open(score_excel_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Scored Audio Features Excel",
                                data=f,
                                file_name="audio_all_features_scored.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

# ===================================================================================================================================================================================================
# ''''''''''''''''''''''''''''''''''''''''''''''''''.........................VIDEO SCORING......................................'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ===================================================================================================================================================================================================
                        try:
                            # Per-chunk scoring
                            m_df_combined["head_score"], m_df_combined["head_feedback"] = zip(*m_df_combined.apply(
                                lambda row: head_score(row["pitch_mean"], row["yaw_mean"],
                                    head_strict=head_strict,
                                    head_moderate=head_moderate,
                                    fluctuation_thresh=head_fluct),
                                axis=1
                            ))

                            m_df_combined["gaze_score"], m_df_combined["gaze_feedback"] = zip(*m_df_combined.apply(
                                lambda row: gaze_score(row["horiz_mean"], row["vert_mean"],
                                    attentive_thresh=attentive_thresh,
                                    fluctuation_thresh=gaze_fluct),
                                axis=1
                            ))

                            m_df_combined["body_score"], m_df_combined["body_feedback"] = zip(*m_df_combined.apply(
                                lambda row: body_score(row["body_mean"],
                                    body_min=body_min,
                                    body_max=body_max,
                                    fluctuation_thresh=body_fluct),
                                axis=1
                            ))

                            # Show per-chunk score table
                            st.markdown("### Video Feature Scoring per chunk (1–3)")
                            st.dataframe(m_df_combined[[
                                "video_name", "chunk", "head_score", "gaze_score", "body_score"
                            ]], use_container_width=True)

                            # Calculate final video-level scores
                            weights = {"head": head_w, "gaze": gaze_w, "body": body_w}
                            final_scores = calculate_final_video_score(
                                m_df_combined,
                                method="weighted" if scoring_method == "Weighted" else "simple",
                                weights=weights
                            )
                            # Save scored table
                            scored_excel = os.path.join(self.temp_dir, "video_features_scored.xlsx")
                            m_df_combined.to_excel(scored_excel, index=False)

                            with open(scored_excel, "rb") as f:
                                st.download_button(
                                    label="📥 Download Per Chunk score with feedback (Excel)",
                                    data=f,
                                    file_name="video_features_scored.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            video_final_score = list(final_scores.values())[0]["final_score"]

# ============================..................audio final score represent.................========================================================================
                            st.success(f"Final Audio Modality Score (1–3): **{final_score:.2f}/3** — {interpretation}")
#============================================...........Display final Video result.............=============================================================
                            for vid, val in final_scores.items():
                                # st.write("**Component Scores:**", val["components"])
                                st.success(f"Final Video Modality Score (1–3): **{val['final_score']}/3** — {val['feedback']}")                                
                        except Exception as e:
                                st.error(f"❌ Scoring failed: {e}")


# ===================================================================================================================================================================================
#=============================================================🏁 FINAL PPT SCORING==================================================================================================
#===================================================================================================================================================================================
                        st.markdown("## 🏁 Final PPT Scoring Summary")
                        result = final_ppt_scoring(
                            audio_score=final_score,
                            video_score=video_final_score,
                            method=scoring_mode,
                            weights=weights
                        )
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Audio Final Score", f"{final_score:.2f}")
                        with col2:
                            st.metric("Video Final Score", f"{video_final_score:.2f}")
                        with col3:
                            st.metric("PPT Final Score", f"{result['Final PPT Score (1–3)']:.2f}")

                        st.progress(result["Final PPT Performance (%)"] / 100)
                        st.success(f"🏆 Overall Performance: **{result['Interpretation']}** ({result['Final PPT Performance (%)']}%)")

                        # ---- PIE CHART (Audio vs Video Contribution) ----
                        audio_value = final_score
                        video_value = video_final_score
                        ppt_value = result["Final PPT Score (1–3)"]

                        col1, col2 = st.columns(2)
                        with col1:
                            fig1, ax1 = plt.subplots(figsize=(3, 3))
                            ax1.pie(
                                [audio_value, video_value],
                                labels=["Audio", "Video"],
                                autopct="%1.1f%%",
                                startangle=90,
                                textprops={'fontsize': 8},
                                shadow=True
                            )
                            ax1.set_title("Audio vs Video Contribution", fontsize=9, pad=8)
                            plt.tight_layout()
                            st.pyplot(fig1, use_container_width=False)

                        # ---- BAR CHART (Score Comparison) ----
                        with col2:
                            fig2, ax2 = plt.subplots(figsize=(3.2, 2.2))
                            bars = ax2.bar(["Audio", "Video", "Final PPT"], [audio_value, video_value, ppt_value], width=0.5)
                            ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
                            ax2.set_ylim(0, 3)
                            ax2.set_ylabel("Score (1–3)", fontsize=8)
                            ax2.set_title("Audio vs Video vs Final PPT Score", fontsize=9, pad=6)
                            ax2.tick_params(axis='x', labelsize=8)
                            ax2.tick_params(axis='y', labelsize=8)
                            plt.tight_layout()
                            st.pyplot(fig2, use_container_width=False)

                        # ==============================
                        # 💬 QUALITATIVE FEEDBACK
                        # ==============================
                        feedback_lines = []
                        if audio_value > video_value + 0.3:
                            feedback_lines.append("🎧 Your **audio delivery** is stronger than your visual engagement. Great clarity and tone!")
                        elif video_value > audio_value + 0.3:
                            feedback_lines.append("🎥 Your **visual engagement** is impressive — maintain that energy! Work slightly on vocal modulation.")
                        else:
                            feedback_lines.append("🗣️ Balanced performance between audio and visual delivery — well done!")

                        # Based on overall PPT interpretation
                        interpretation = result["Interpretation"]
                        if interpretation == "Excellent":
                            feedback_lines.append("🌟 Excellent overall presentation — confident, clear, and well-coordinated.")
                        elif interpretation == "Average":
                            feedback_lines.append("🙂 Decent presentation — with a bit more practice, you can make it more engaging.")
                        else:
                            feedback_lines.append("🚀 Needs more rehearsal — focus on voice modulation, pacing, and maintaining eye contact.")

                        st.info("\n".join(feedback_lines))

                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error during multimodal preprocessing: {str(e)}")

    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Page", ["Home", "Audio Analysis", "Video Analysis", "Multimodal Analysis"])
        if page == "Home":
            self.home_page()
        elif page == "Audio Analysis":
            self.audio_analysis_page()
        elif page == "Video Analysis":
            self.video_analysis_page()
        elif page == "Multimodal Analysis":
            self.multimodal_analysis_page()

# ================== RUN APP ==================
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
