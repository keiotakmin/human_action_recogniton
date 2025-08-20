import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
from torchvision.transforms import Compose, Lambda, functional, transforms
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.models.head import ResNetBasicHead
from tqdm import tqdm
import time
import datetime
import argparse
import copy
import torch.nn.functional as F
from collections import defaultdict
from ultralytics import YOLO

# 環境設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows環境でのエラー回避

# AVAデータセットのアクションクラス（80クラス）
AVA_ACTIONS = [
    "bend/bow", "crawl", "crouch/kneel", "dance", "fall down", "get up", "jump/leap", "lie/sleep", "martial art", "run/jog",
    "sit", "stand", "swim", "walk", "answer phone", "brush teeth", "carry/hold", "catch", "chop", "climb",
    "clink glass", "close", "cook", "cut", "dig", "dress/put on", "drink", "drive", "eat", "enter",
    "exit", "extract", "fishing", "hit", "kick", "lift/pick up", "listen", "open", "paint", "play board game",
    "play instrument", "play sports", "point", "pour", "press", "pull", "push", "put down", "read", "ride",
    "row boat", "sail boat", "shoot", "shovel", "smoke", "stir", "take photo", "text on phone", "throw", "touch",
    "turn", "watch", "work on computer", "write", "fight/hit", "give/serve", "grab", "hand clap", "hand shake", "hand wave",
    "hug", "kiss", "lift person", "listen to person", "play with kids", "push person", "sing to", "take from person", "talk to", "watch person"
]

# ターゲットとなる行動クラス（選択したサブセット）
TARGET_ACTIONS = [
    "stand", "sit", "walk", "run/jog", "carry/hold", "fight/hit", "fall down", "lie/sleep", 
    "dance", "talk to", "eat", "work on computer", "read", "write", "hand wave"
]

# インデックスマッピングの作成
TARGET_ACTION_INDICES = [AVA_ACTIONS.index(action) for action in TARGET_ACTIONS]

# ハイパーパラメータ設定
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
TRAIN_VAL_SPLIT = 0.8  # 訓練:検証の比率

# SlowFast用の設定
SIDE_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CROP_SIZE = 256
NUM_FRAMES = 32 # 変更不可
SAMPLING_RATE = 4
FRAMES_PER_SECOND = 30
SLOWFAST_ALPHA = 4 # 変更不可
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FRAMES_PER_SECOND

# 検出関連の設定
PERSON_THRESHOLD = 0.3  # 人物検出の閾値
ACTION_THRESHOLD = 0.4  # 行動検出の閾値
NUM_PERSON_CLASSES = 1  # AVAモデルは人物クラスのみ検出
NUM_ACTION_CLASSES = 80  # AVAデータセットの行動クラス数
DETECTION_INTERVAL = 30  # 検出間隔（約1秒）

class PackPathway(torch.nn.Module):
    """
    動画フレームをSlowFastモデル用に変換するトランスフォーム
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
            
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Slow Pathway
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames).float() if isinstance(frames, np.ndarray) else frames
            
        slow_pathway = torch.index_select(
            frames,
            1, # Time dimension
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def create_detection_transform(alpha=SLOWFAST_ALPHA):
    """SlowFast検出モデル用の変換を作成"""
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(NUM_FRAMES),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(MEAN, STD),
                ShortSideScale(size=SIDE_SIZE),
                CenterCropVideo(CROP_SIZE),
                PackPathway(alpha=alpha)
            ]
        ),
    )
    return transform

class SlowFastActionDetectionModel(nn.Module):
    """
    SlowFast行動検出モデル用のラッパークラス
    """
    def __init__(self, target_actions=None):
        super().__init__()
        
        self.target_actions = target_actions
        self.action_indices = None
        if target_actions:
            # ターゲットのアクションインデックスを作成
            self.action_indices = [AVA_ACTIONS.index(action) for action in target_actions]
        
        # SlowFast検出モデルのロード
        print("SlowFast検出モデルを読み込み中...")
        try:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50_detection', pretrained=True)
            print("SlowFast R50検出モデルを読み込みました")
            self.model_name = 'slowfast_r50_detection'
        except Exception as e:
            print(f"R50検出モデル読み込み中にエラー: {e}")
            try:
                print("SlowFast R101検出モデルを試みます...")
                self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101_detection', pretrained=True)
                print("SlowFast R101検出モデルを読み込みました")
                self.model_name = 'slowfast_r101_detection'
            except Exception as e2:
                print(f"R101検出モデル読み込み中にエラー: {e2}")
                raise RuntimeError("利用可能なSlowFast検出モデルが見つかりません")
        
        # モデル構造を表示
        self._analyze_model_structure()
    
    def _analyze_model_structure(self):
        """モデルの構造を分析して表示"""
        print("\nモデル構造の分析:")
        
        # 主要コンポーネントの確認
        if hasattr(self.model, 'detection_head'):
            print("検出ヘッド:", type(self.model.detection_head))
            
            if hasattr(self.model.detection_head, 'bbox_head'):
                print("バウンディングボックスヘッド:", type(self.model.detection_head.bbox_head))
                if isinstance(self.model.detection_head.bbox_head, ResNetBasicHead):
                    print(f"バウンディングボックスヘッド出力サイズ: {self.model.detection_head.bbox_head.proj.out_features}")
            
            if hasattr(self.model.detection_head, 'cls_head'):
                print("分類ヘッド:", type(self.model.detection_head.cls_head))
                if isinstance(self.model.detection_head.cls_head, ResNetBasicHead):
                    print(f"分類ヘッド出力サイズ: {self.model.detection_head.cls_head.proj.out_features}")
        else:
            print("検出ヘッドが見つかりません。詳細な構造を表示します:")
            self._print_model_structure()
    
    def _print_model_structure(self):
        """モデル構造を詳細に出力する補助関数"""
        print("詳細なモデル構造:")
        for name, module in self.model.named_children():
            print(f"Module '{name}':", module)
    
    def forward(self, x, bboxes=None):
        """
        モデルの順伝播
        入力: 
        - x: [slow_pathway, fast_pathway]
        - bboxes: バウンディングボックス [N, 4] (オプション)
        出力: 検出結果（行動クラスのロジット）
        """
        # bboxesがNoneの場合は、ダミーのバウンディングボックスを生成
        if bboxes is None:
            device = x[0].device if isinstance(x, list) else x.device
            bboxes = torch.tensor([[64, 64, 192, 192]], dtype=torch.float32).to(device)
        
        # 入力の形状を確認・修正
        if isinstance(x, list) and len(x) == 2:
            slow_pathway, fast_pathway = x
            
            # 5D形状チェック (batch_size, channels, temporal, height, width)
            if slow_pathway.dim() == 4:
                slow_pathway = slow_pathway.unsqueeze(0)  # バッチ次元を追加
                print("Slow pathwayにバッチ次元を追加しました")
            if fast_pathway.dim() == 4:
                fast_pathway = fast_pathway.unsqueeze(0)  # バッチ次元を追加
                print("Fast pathwayにバッチ次元を追加しました")
            
            x = [slow_pathway, fast_pathway]
            
            # 形状を確認
            print(f"Slow pathway 形状: {slow_pathway.shape}")
            print(f"Fast pathway 形状: {fast_pathway.shape}")
            print(f"Bounding boxes 形状: {bboxes.shape}")
        
        # バウンディングボックスの形状を修正 [N, 4] -> [N, 5]
        # AVAモデルは [batch_index, x1, y1, x2, y2] 形式を期待
        if bboxes.shape[1] == 4:
            batch_indices = torch.zeros((bboxes.shape[0], 1), device=bboxes.device)
            bboxes = torch.cat([batch_indices, bboxes], dim=1)
            print(f"バウンディングボックスをAVA形式に変換: {bboxes.shape}")
        
        try:
            # AVA検出モデル用の正しい呼び出し方法
            if hasattr(self.model, 'extract_features'):
                # 特徴抽出
                print("特徴抽出を実行中...")
                features = self.model.extract_features(x)
                print(f"抽出された特徴の形状: {[f.shape for f in features] if isinstance(features, list) else features.shape}")
                
                # 検出ヘッドに特徴とバウンディングボックスを渡す
                print("検出ヘッドを実行中...")
                preds = self.model.detection_head(features, bboxes)
                return preds
            else:
                # バウンディングボックスを含めて直接呼び出し
                print("直接モデル呼び出しを実行中...")
                return self.model(x, bboxes)
        except Exception as e:
            print(f"モデル内部で呼び出しエラー: {e}")
            print(f"入力形状の詳細 - Slow: {x[0].shape}, Fast: {x[1].shape}, BBoxes: {bboxes.shape}")
            
            # より詳細なエラー情報
            import traceback
            traceback.print_exc()
            
            # フォールバック: ダミーの出力を返す
            batch_size = bboxes.shape[0] if bboxes is not None else 1
            device = x[0].device if isinstance(x, list) else x.device
            print(f"フォールバック: ダミー出力を生成中（バッチサイズ: {batch_size}）")
            return torch.randn(batch_size, 80, device=device)  # 80は行動クラス数

def preprocess_video_for_detection(video_clip):
    """
    動画クリップをSlowFast検出モデル用に前処理
    """
    # NumPy配列をテンソルに変換（[T, H, W, C] -> [C, T, H, W]）
    video_tensor = torch.from_numpy(video_clip.transpose(3, 0, 1, 2)).float()
    
    # 正規化
    video_tensor = video_tensor / 255.0
    
    # 平均と標準偏差で正規化
    for i in range(3):
        video_tensor[i] = (video_tensor[i] - MEAN[i]) / STD[i]
    
    # SlowFastの2つのパスウェイに変換
    # Fast pathway は全フレーム
    fast_pathway = video_tensor
    
    # Slow pathway は間引いたフレーム
    slow_indices = torch.linspace(
        0, video_tensor.shape[1] - 1, video_tensor.shape[1] // SLOWFAST_ALPHA
    ).long()
    slow_pathway = torch.index_select(video_tensor, 1, slow_indices)
    
    # 重要: バッチ次元を追加して5Dテンソルにする
    # [C, T, H, W] -> [1, C, T, H, W]
    slow_pathway = slow_pathway.unsqueeze(0)
    fast_pathway = fast_pathway.unsqueeze(0)
    
    return [slow_pathway, fast_pathway]

def process_detections(pred_boxes, pred_scores, action_scores, image_size, 
                      score_threshold=PERSON_THRESHOLD, action_threshold=ACTION_THRESHOLD):
    """
    検出結果を処理してバウンディングボックスと行動ラベルのリストを返す
    """
    results = []
    if pred_boxes is None or len(pred_boxes) == 0:
        print("⚠️ pred_boxes が空です")
        return results
    
    print(f"🔍 検出処理開始:")
    print(f"  - pred_boxes: {pred_boxes.shape}")
    print(f"  - pred_scores: {pred_scores.shape}")  
    print(f"  - action_scores: {action_scores.shape}")
    
    # 全ての検出を処理
    for i in range(len(pred_boxes)):
        print(f"📦 検出 {i+1}/{len(pred_boxes)} を処理中...")
        
        # バウンディングボックスの取得
        box = pred_boxes[i].cpu().numpy() if torch.is_tensor(pred_boxes[i]) else pred_boxes[i]
        print(f"   SlowFastボックス座標: {box}")
        
        # 座標を表示ウィンドウサイズに直接変換
        # CROP_SIZE (256x256) -> 表示ウィンドウ (960x720)
        display_x1 = int(box[0] * 960 / CROP_SIZE)
        display_y1 = int(box[1] * 720 / CROP_SIZE)  
        display_x2 = int(box[2] * 960 / CROP_SIZE)
        display_y2 = int(box[3] * 720 / CROP_SIZE)
        
        # 座標の境界チェック
        display_x1 = max(0, min(display_x1, 960))
        display_y1 = max(0, min(display_y1, 720))
        display_x2 = max(0, min(display_x2, 960))
        display_y2 = max(0, min(display_y2, 720))
        
        display_box = [display_x1, display_y1, display_x2, display_y2]
        print(f"   表示ボックス座標: {display_box}")
        
        # 人物スコアの取得
        person_score = pred_scores[i].item() if i < len(pred_scores) else 0.9
        print(f"   人物スコア: {person_score}")
        
        # 行動スコアの処理
        if i < action_scores.shape[0]:
            action_logits = action_scores[i]
        else:
            action_logits = action_scores[0]  # 最初のスコアを使用
        
        # シグモイド関数を適用して確率に変換
        action_probs = torch.sigmoid(action_logits)
        
        # 閾値を超える行動を選択、なければ上位3つを選択
        high_confidence_actions = []
        for j, prob in enumerate(action_probs):
            if prob.item() > action_threshold and j < len(AVA_ACTIONS):
                high_confidence_actions.append((j, prob.item()))
        
        # 閾値を超える行動がない場合は上位3つを選択
        if not high_confidence_actions:
            top_k = 3
            top_values, top_indices = torch.topk(action_probs, top_k)
            actions = []
            for j in range(top_k):
                action_id = top_indices[j].item()
                action_score = top_values[j].item()
                if action_id < len(AVA_ACTIONS):
                    action_name = AVA_ACTIONS[action_id]
                    actions.append((action_name, action_score))
                    print(f"   行動 {j+1}: {action_name} (スコア: {action_score:.3f})")
        else:
            # 閾値を超える行動を使用
            actions = []
            high_confidence_actions.sort(key=lambda x: x[1], reverse=True)  # スコア順でソート
            for action_id, action_score in high_confidence_actions[:5]:  # 上位5つまで
                action_name = AVA_ACTIONS[action_id]
                actions.append((action_name, action_score))
                print(f"   高信頼度行動: {action_name} (スコア: {action_score:.3f})")
        
        # 結果の追加
        result = {
            "box": display_box,  # 表示用に変換済みの座標
            "person_score": person_score,
            "actions": actions
        }
        results.append(result)
        print(f"   ✅ 検出結果 {i+1} を追加")
    
    print(f"🎯 処理完了: {len(results)}個の検出結果")
    return results

def visualize_detections(frame, detections, target_actions=None):
    """
    検出結果を可視化して描画済みフレームを返す
    """
    vis_frame = frame.copy()
    
    if not detections:
        return vis_frame
    
    print(f"🎨 可視化開始: {len(detections)}個の検出結果を処理")
    
    # 各検出に対して処理
    for i, det in enumerate(detections):
        try:
            box = det["box"]
            x1, y1, x2, y2 = [max(0, min(int(coord), frame.shape[1] if coord in [box[0], box[2]] else frame.shape[0])) for coord in box]
            
            print(f"  検出{i+1}: ボックス({x1}, {y1}, {x2}, {y2})")
            
            # バウンディングボックスを描画（太い緑線）
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # 人物スコアを描画
            person_text = f"Person: {det['person_score']:.2f}"
            cv2.putText(vis_frame, person_text, (x1, max(y1 - 15, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            
            # 行動ラベルを描画（上位3つまで）
            y_offset = 30
            for j, (action, score) in enumerate(det["actions"][:3]):
                action_text = f"{action}: {score:.2f}"
                text_color = (0, 0, 255) if score > 0.5 else (0, 255, 255)
                
                text_y = min(y1 + y_offset, frame.shape[0] - 15)
                cv2.putText(vis_frame, action_text, (x1, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                y_offset += 30
                print(f"    行動{j+1}: {action} ({score:.2f})")
            
            print(f"  ✅ 検出{i+1}の描画完了")
            
        except Exception as e:
            print(f"  ❌ 検出{i+1}の描画エラー: {e}")
            # エラーが発生しても、画面中央に基本的な情報を表示
            cv2.rectangle(vis_frame, (100, 100), (500, 300), (0, 255, 0), 4)
            cv2.putText(vis_frame, "Detection Error", (120, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    print(f"🎨 可視化完了")
    return vis_frame

def realtime_action_detection(model, device=None):
    """
    YOLOv8を使用したリアルタイム行動検出
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # YOLOv8による人物検出器の初期化
    try:
        yolo_detector = YOLO("yolov8x.pt")  # 最新のYOLOv8 extralargeモデル
        print("YOLOv8x人物検出器を読み込みました")
    except Exception as e:
        print(f"YOLOv8xの読み込みエラー、yolov8nを試します: {e}")
        try:
            yolo_detector = YOLO("yolov8n.pt")  # より軽量なnanoモデル
            print("YOLOv8n人物検出器を読み込みました")
        except Exception as e2:
            print(f"YOLOv8の読み込みエラー: {e2}")
            print("システムを終了します")
            return

    # カメラを開く
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    frame_buffer = []
    last_detections = []
    last_detection_frame = -DETECTION_INTERVAL
    window_width, window_height = 960, 720
    cv2.namedWindow('SlowFast + YOLOv8 Action Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SlowFast + YOLOv8 Action Detection', window_width, window_height)

    frame_count = 0
    start_time = time.time()
    detection_time = 0

    print("SlowFast + YOLOv8による行動検出を開始します。'q'キーで終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレーム取得に失敗")
            break

        display_frame = cv2.resize(frame.copy(), (window_width, window_height))
        proc_frame = cv2.resize(frame, (CROP_SIZE, CROP_SIZE))
        frame_buffer.append(proc_frame)
        if len(frame_buffer) > NUM_FRAMES:
            frame_buffer.pop(0)

        should_detect = len(frame_buffer) == NUM_FRAMES and (frame_count - last_detection_frame) >= DETECTION_INTERVAL

        if should_detect:
            print(f"🔍 フレーム {frame_count} で検出を実行中...")
            det_start = time.time()
            current_frame = cv2.resize(frame, (640, 480))

            # YOLOv8による人物検出
            try:
                results = yolo_detector.predict(
                    current_frame, 
                    imgsz=640, 
                    conf=PERSON_THRESHOLD,
                    classes=[0],  # 人物クラスのみ
                    verbose=False
                )
                
                pred_boxes = []
                pred_scores = []
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        # YOLOv8の出力形式に対応
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        
                        if conf >= PERSON_THRESHOLD:  # 信頼度閾値チェック
                            # 座標をCROP_SIZEに合わせてスケーリング
                            x1 = x1 * CROP_SIZE / current_frame.shape[1]
                            y1 = y1 * CROP_SIZE / current_frame.shape[0]
                            x2 = x2 * CROP_SIZE / current_frame.shape[1]
                            y2 = y2 * CROP_SIZE / current_frame.shape[0]
                            pred_boxes.append([x1, y1, x2, y2])
                            pred_scores.append(conf)

                if pred_boxes:
                    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32).to(device)
                    pred_scores = torch.tensor(pred_scores, dtype=torch.float32).to(device)
                    print(f"✅ YOLOv8で {len(pred_boxes)} 人を検出")
                else:
                    pred_boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                    pred_scores = torch.zeros(0, dtype=torch.float32).to(device)
                    print("⚠️ YOLOv8で人物検出されませんでした")
                    
            except Exception as e:
                print(f"YOLOv8検出エラー: {e}")
                pred_boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                pred_scores = torch.zeros(0, dtype=torch.float32).to(device)

            # 人物が検出された場合のみ行動認識を実行
            if len(pred_boxes) > 0:
                try:
                    # フレームバッファをSlowFast検出モデル用に前処理
                    frames_array = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_buffer])
                    
                    # 前処理と変換
                    inputs = preprocess_video_for_detection(frames_array)
                    inputs = [x.to(device) for x in inputs]
                    
                    # 行動予測
                    with torch.no_grad():
                        action_preds = model(inputs, pred_boxes)
                    
                    # 検出結果を処理
                    image_size = (CROP_SIZE, CROP_SIZE)
                    new_detections = process_detections(
                        pred_boxes, pred_scores, action_preds,
                        image_size, PERSON_THRESHOLD, ACTION_THRESHOLD
                    )
                    
                    # 検出結果を更新
                    if new_detections:
                        last_detections = new_detections
                        last_detection_frame = frame_count
                        print(f"✅ 新しい検出結果を保存: {len(new_detections)}個")
                        
                except Exception as e:
                    print(f"行動認識エラー: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ 人物が検出されませんでした")

            det_end = time.time()
            detection_time += (det_end - det_start)

        # 最後の検出結果を表示（持続表示）
        if last_detections:
            display_frame = visualize_detections(display_frame, last_detections, TARGET_ACTIONS)
            
            # 検出結果の年齢を表示
            frames_since_detection = frame_count - last_detection_frame
            cv2.putText(display_frame, f"Detection age: {frames_since_detection} frames", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # システム情報の表示
        cv2.putText(display_frame, "SlowFast + YOLOv8 Action Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # バッファ状態の表示
        cv2.putText(display_frame, f"Frame buffer: {len(frame_buffer)}/{NUM_FRAMES}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 検出FPSを表示
        detection_count = (frame_count // DETECTION_INTERVAL) + 1
        avg_detection_time = detection_time / detection_count if detection_count > 0 else 0
        detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        cv2.putText(display_frame, f"Detection FPS: {detection_fps:.2f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 全体のFPSを表示
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(display_frame, f"Display FPS: {fps:.2f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # ステータス表示（右上）
        persistent_countdown = max(0, DETECTION_INTERVAL - (frame_count - last_detection_frame))
        status_color = (0, 255, 0) if persistent_countdown > 0 else (0, 0, 255)
        cv2.rectangle(display_frame, (window_width-150, 10), (window_width-10, 130), status_color, 3)
        
        # ステータステキスト
        status_text = "DETECTING" if should_detect else "WAITING"
        cv2.putText(display_frame, status_text, (window_width-140, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # フレーム番号
        cv2.putText(display_frame, f"F:{frame_count}", (window_width-140, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # 次の検出までのカウントダウン
        cv2.putText(display_frame, f"Next:{persistent_countdown}", (window_width-140, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # 検出された人数
        num_persons = len(last_detections) if last_detections else 0
        cv2.putText(display_frame, f"Persons:{num_persons}", (window_width-140, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # フレーム表示
        cv2.imshow('SlowFast + YOLOv8 Action Detection', display_frame)
        
        # フレームカウンター更新
        frame_count += 1
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # リソース解放
    cap.release()
    cv2.destroyAllWindows()
    
    # 最終統計の表示
    total_time = time.time() - start_time
    print(f"\n=== 実行統計 ===")
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"処理フレーム数: {frame_count}")
    print(f"平均FPS: {frame_count / total_time:.2f}")
    print(f"検出実行回数: {detection_count}")
    print(f"平均検出時間: {avg_detection_time:.3f}秒")

def main():
    """メイン処理"""
    # パラメータの設定
    parser = argparse.ArgumentParser(description="SlowFast + YOLOv8 Action Detection System")
    parser.add_argument("--output_dir", type=str, default="./output_slowfast_yolo8_detection",
                        help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # モデルの読み込み
        print("SlowFast行動検出モデルを読み込み中...")
        model = SlowFastActionDetectionModel(target_actions=TARGET_ACTIONS)
        
        # リアルタイム検出の実行
        print("YOLOv8による人物検出とSlowFastによる行動認識を開始します...")
        realtime_action_detection(model, device)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print("\n推奨される対処法:")
        print("1. PyTorchとtorchvisionが正しくインストールされているか確認")
        print("2. pytorchvideoライブラリがインストールされているか確認")
        print("3. ultralyticsライブラリがインストールされているか確認 (pip install ultralytics)")
        print("4. インターネット接続を確認（初回実行時はモデルをダウンロードします）")

if __name__ == "__main__":
    print("=" * 60)
    print("SlowFast + YOLOv8 リアルタイム行動検出システム")
    print("=" * 60)
    print("このシステムは最新技術を組み合わせます:")
    print("- YOLOv8: 最新かつ高精度な人物検出")
    print("- SlowFast R50: AVAデータセットで事前訓練された行動認識")
    print("- リアルタイム処理: Webカメラからの映像をリアルタイムで解析")
    print()
    print("検出可能な行動:")
    for i, action in enumerate(TARGET_ACTIONS):
        print(f"  {i+1:2d}. {action}")
    print()
    print("システム要件:")
    print("- Python 3.8+")
    print("- PyTorch")
    print("- pytorchvideo")
    print("- ultralytics (YOLOv8)")
    print("- OpenCV")
    print("- Webカメラ")
    print()
    print("準備が完了したら何かキーを押してください...")
    input()
    
    main()