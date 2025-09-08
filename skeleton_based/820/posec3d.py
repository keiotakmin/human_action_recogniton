import cv2
import numpy as np
import torch
from ultralytics import YOLO
from mmaction.apis import init_recognizer, inference_skeleton
import mmengine
import collections
import time
import os
import urllib.request

class RealTimeActionRecognition:
    """
    YOLOv8-PoseとMMAction2 PoseC3Dを統合したリアルタイム行動認識システム。
    オリジナルの60クラスNTU行動認識を使用。
    """

    def __init__(self, device=None):
        # Auto-detect device: use CUDA if available, otherwise CPU
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print("CUDA is available. Using GPU.")
            else:
                self.device = 'cpu'
                print("CUDA not available. Using CPU.")
        else:
            # Check if the specified device is actually available
            import torch
            if device.startswith('cuda') and not torch.cuda.is_available():
                print(f"Warning: {device} requested but CUDA not available. Falling back to CPU.")
                self.device = 'cpu'
            else:
                self.device = device
                
        self.pose_model = None
        self.action_model = None
        self.action_labels = None
        self.use_posec3d = False  # Initialize use_posec3d flag
        
        # パラメータ設定
        self.pose_model_path = 'yolov8x-pose.pt'
        # Use PoseC3D model with the provided checkpoint and config
        self.action_checkpoint_path = 'C:/Users/takmi/RA/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth'
        self.action_config_path = 'C:/Users/takmi/RA/mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'
        self.action_label_path = 'label_map_ntu60.txt'
        
        self.WINDOW_SIZE = 48  # 行動認識に使用するフレーム数
        self.CONF_THRESHOLD = 0.5 # 人物検出の信頼度閾値
        self.ACTION_UPDATE_INTERVAL = 5 # 行動認識を更新するフレーム間隔

        # データバッファの初期化
        self.pose_results_buffer = collections.defaultdict(
            lambda: collections.deque(maxlen=self.WINDOW_SIZE)
        )
        self.action_predictions = {}
        self.frame_counter = collections.defaultdict(int)

    def download_file(self, url, filepath):
        """
        指定されたURLからファイルをダウンロードするヘルパー関数。
        """
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading {filepath}: {e}")
                raise

    def load_models(self):
        """
        YOLOv8-PoseモデルとMMAction2 PoseC3Dモデルをロードする。
        """
        print("Loading models...")
        
        # Apply NumPy compatibility fix BEFORE loading MMAction2
        self.fix_numpy_compatibility()
        
        # YOLOv8-Poseモデルのロード
        self.pose_model = YOLO(self.pose_model_path)
        
        # Try to load PoseC3D model
        try:
            # --- MMAction2 PoseC3Dモデルのロード ---
            # Load the config file directly
            if os.path.exists(self.action_config_path):
                config = mmengine.Config.fromfile(self.action_config_path)
                print(f"Loaded config from: {self.action_config_path}")
            else:
                print(f"Config file not found at: {self.action_config_path}")
                # Fallback to a basic PoseC3D config if file not found
                config = self.create_fallback_posec3d_config()
            
            # Check if checkpoint exists
            if not os.path.exists(self.action_checkpoint_path):
                print(f"Checkpoint not found at: {self.action_checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint file not found: {self.action_checkpoint_path}")
            
            self.action_model = init_recognizer(config, self.action_checkpoint_path, device=self.device)
            
            # ラベルマップファイルのダウンロードとロード
            label_url = 'https://raw.githubusercontent.com/open-mmlab/mmaction/main/tools/data/skeleton/label_map_ntu60.txt'
            self.download_file(label_url, self.action_label_path)
            with open(self.action_label_path, 'r') as f:
                # Load original 60 NTU action classes without mapping
                self.action_labels = [line.strip() for line in f]
                
            print("PoseC3D model loaded successfully.")
            self.use_posec3d = True
            
        except Exception as e:
            print(f"Failed to load PoseC3D model: {e}")
            print("PoseC3D model is required for this system.")
            raise e
            
        print("Models loaded successfully.")

    def create_fallback_posec3d_config(self):
        """
        Create a fallback PoseC3D config if the config file is not found
        """
        print("Creating fallback PoseC3D config...")
        
        posec3d_config_dict = dict(
            model=dict(
                type='Recognizer3D',
                backbone=dict(
                    type='ResNet3dSlowOnly',
                    depth=50,
                    pretrained=None,
                    lateral=False,
                    conv1_kernel=(1, 7, 7),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    inflate=(0, 0, 1, 1),
                    norm_eval=False),
                cls_head=dict(
                    type='I3DHead',
                    num_classes=60,
                    in_channels=2048,
                    spatial_type='avg',
                    dropout_ratio=0.5,
                    init_std=0.01)
            ),
            # Test pipeline for skeleton data
            test_pipeline=[
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=1, mode='zero'),
                dict(type='PackActionInputs')
            ]
        )
        
        return mmengine.Config(cfg_dict=posec3d_config_dict)

    def process_video(self, input_video_path, output_video_path):
        """
        ビデオファイルを処理し、行動認識結果を付与したビデオを保存する。
        """
        # Fix the model loading check - pose_model is always loaded, action_model might be None for fallback
        if not self.pose_model:
            print("Pose model is not loaded. Please call load_models() first.")
            return

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
            return

        # ビデオライターの設定
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            start_time = time.time()

            # Stage 1: YOLOv8による追跡と姿勢推定
            results = self.pose_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.CONF_THRESHOLD, verbose=False)

            # Stage 2 & 3: データ変換と行動認識
            self.update_buffers_and_recognize_actions(results, (height, width))
            
            # Stage 4: 結果の可視化
            annotated_frame = self.visualize_results(results, frame)

            # パフォーマンス表示
            end_time = time.time()
            processing_fps = 1 / (end_time - start_time)
            cv2.putText(annotated_frame, f"FPS: {processing_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(annotated_frame)
            frame_idx += 1
            print(f"Processing frame {frame_idx}...", end='\r')

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing complete. Output saved to {output_video_path}")

    def update_buffers_and_recognize_actions(self, results, img_shape):
        """
        YOLOの結果をバッファに格納し、一定間隔で行動認識を実行する。
        """
        # Handle results as a list (newer YOLO versions)
        if isinstance(results, list):
            if len(results) == 0:
                return
            result = results[0]
        else:
            result = results
        
        # Safety checks for boxes and keypoints
        if not hasattr(result, 'boxes') or result.boxes is None:
            return
        
        if not hasattr(result.boxes, 'id') or result.boxes.id is None:
            return
        
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return

        track_ids = result.boxes.id.int().cpu().tolist()
        #.data 属性からキーポイントのテンソル (x, y, conf) を取得
        keypoints_data = result.keypoints.data.cpu().numpy()

        for i, track_id in enumerate(track_ids):
            # キーポイントデータをバッファに追加
            person_kpts = keypoints_data[i]
            self.pose_results_buffer[track_id].append(person_kpts)
            
            self.frame_counter[track_id] += 1

            # バッファが満たされ、更新間隔に達した場合に行動認識を実行
            if len(self.pose_results_buffer[track_id]) == self.WINDOW_SIZE and \
               self.frame_counter[track_id] % self.ACTION_UPDATE_INTERVAL == 0:
                
                buffer = self.pose_results_buffer[track_id]
                
                # Stage 3: 行動認識
                try:
                    if self.use_posec3d and self.action_model:
                        # Stage 2: データ変換 - PoseC3Dの正しい入力形式に整形
                        # PoseC3D expects 2D keypoints in a specific format
                        pose_sequence_2d = np.array([kpt[:, :2] for kpt in buffer])  # (T, V, 2)
                        pose_scores = np.array([kpt[:, 2] for kpt in buffer])   # (T, V)

                        # PoseC3Dが期待する形式: リスト形式のpose_results (2D keypoints)
                        pose_results = []
                        for t in range(len(buffer)):
                            pose_result = {
                                'keypoints': pose_sequence_2d[t:t+1],  # Add person dimension: (1, V, 2)
                                'keypoint_scores': pose_scores[t:t+1],  # Add person dimension: (1, V)
                                'total_frames': self.WINDOW_SIZE
                            }
                            pose_results.append(pose_result)
                        
                        # Use PoseC3D model
                        action_result = inference_skeleton(self.action_model, pose_results, img_shape)
                        
                        # Better error handling for prediction results
                        if hasattr(action_result, 'pred_score'):
                            pred_scores = action_result.pred_score
                            
                            if hasattr(pred_scores, 'tolist'):
                                pred_scores = pred_scores.tolist()
                            elif hasattr(pred_scores, 'cpu'):
                                pred_scores = pred_scores.cpu().numpy().tolist()
                            elif hasattr(pred_scores, 'numpy'):
                                pred_scores = pred_scores.numpy().tolist()
                            
                            if self.action_labels and len(pred_scores) > 0:
                                action_id = np.argmax(pred_scores)
                                
                                if action_id < len(self.action_labels):
                                    action_label = self.action_labels[action_id]
                                    confidence = pred_scores[action_id] if isinstance(pred_scores, list) else pred_scores[action_id].item()
                                    self.action_predictions[track_id] = f"{action_label} ({confidence:.2f})"
                                else:
                                    self.action_predictions[track_id] = f"Unknown action {action_id}"
                            else:
                                self.action_predictions[track_id] = "No valid predictions"
                        else:
                            # Try alternative attributes
                            for attr in ['pred_scores', 'predictions', 'logits', 'outputs']:
                                if hasattr(action_result, attr):
                                    pred_scores = getattr(action_result, attr)
                                    action_id = np.argmax(pred_scores)
                                    if self.action_labels and action_id < len(self.action_labels):
                                        action_label = self.action_labels[action_id]
                                        self.action_predictions[track_id] = action_label
                                    break
                            else:
                                self.action_predictions[track_id] = "Cannot extract predictions"
                    else:
                        # PoseC3D model is required
                        self.action_predictions[track_id] = "PoseC3D model not available"
                        
                except Exception as e:
                    # PoseC3D failed - no fallback available
                    self.action_predictions[track_id] = f"Error: {str(e)[:50]}"


    def visualize_results(self, results, frame):
        """
        検出、追跡、行動認識の結果をフレーム上に描画する。
        """
        # Handle results as a list (newer YOLO versions)
        if isinstance(results, list):
            if len(results) == 0:
                return frame
            result = results[0]
        else:
            result = results
            
        annotated_frame = result.plot() # YOLOv8のデフォルト描画（BBox, Pose）

        # Safety checks
        if not hasattr(result, 'boxes') or result.boxes is None:
            return annotated_frame
            
        if not hasattr(result.boxes, 'id') or result.boxes.id is None:
            return annotated_frame

        track_ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        
        # 画面のサイズを取得
        frame_height, frame_width = annotated_frame.shape[:2]

        for track_id, box in zip(track_ids, boxes):
            x1, y1, _, _ = map(int, box)
            
            # 行動ラベルを描画
            action_label = self.action_predictions.get(track_id, "Recognizing...")
            text = f"ID: {track_id} - {action_label}"
            
            # テキストのサイズを取得（1.2倍のサイズと太さ）
            font_scale = 0.6 * 1.2  # 1.2倍のサイズ
            thickness = 2  # 太さを2に設定
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # ラベルの位置を画面内に収まるように調整
            label_x = x1
            label_y = y1 - 5
            
            # 右端がはみ出る場合、左に移動
            if label_x + text_w > frame_width:
                label_x = frame_width - text_w - 5
            
            # 左端がはみ出る場合、右に移動
            if label_x < 0:
                label_x = 5
            
            # 上端がはみ出る場合、下に移動
            if label_y - text_h < 0:
                label_y = y1 + text_h + 5
            
            # テキストの背景を描画
            cv2.rectangle(annotated_frame, 
                         (label_x, label_y - text_h - 8), 
                         (label_x + text_w, label_y), 
                         (0, 0, 0), -1)
            
            # テキストを描画（1.2倍のサイズと太さ）
            cv2.putText(annotated_frame, text, 
                       (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return annotated_frame

    def fix_numpy_compatibility(self):
        """
        Fix NumPy 2.0 compatibility issues in MMAction2
        """
        try:
            # Patch the problematic file
            import mmaction.datasets.transforms.pose_transforms as pose_transforms
            
            # Get the original transform method
            original_transform = pose_transforms.PoseCompact.transform
            
            def patched_transform(self, results):
                # Monkey patch np.Inf to np.inf before calling original method
                import numpy as np
                if not hasattr(np, 'Inf'):
                    np.Inf = np.inf
                return original_transform(self, results)
            
            # Apply the patch
            pose_transforms.PoseCompact.transform = patched_transform
            print("Applied NumPy 2.0 compatibility patch")
            
        except Exception as e:
            print(f"Could not apply NumPy patch: {e}")
            print("Will use simplified pipeline instead")

if __name__ == '__main__':
    # 使用例
    input_video = 'C:/Users/takmi/RA/input_video/n60_baseline1.mp4'
    output_video = 'C:/Users/takmi/RA/output_video/n60_baseline1/posec3d60_base2.mp4'
    
    # システムの初期化（デバイス自動検出）
    recognizer = RealTimeActionRecognition()
    recognizer.load_models()
    recognizer.process_video(input_video, output_video) 