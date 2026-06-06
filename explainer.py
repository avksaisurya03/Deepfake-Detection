import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from mtcnn import MTCNN
import timm

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_local_model(model_path):
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def detect_and_crop_face(image_pil):
    image_np = np.array(image_pil.convert('RGB'))
    results  = detector.detect_faces(image_np)

    if not results:
        return image_pil.resize((224, 224)), False

    largest = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = largest['box']
    pad = int(0.2 * max(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(image_np.shape[1], x + w + pad)
    y2  = min(image_np.shape[0], y + h + pad)
    face_crop = image_np[y1:y2, x1:x2]
    face_pil  = Image.fromarray(face_crop).resize((224, 224))
    return face_pil, True


def predict_single_frame(image_pil, model):
    face_image, face_found = detect_and_crop_face(image_pil)
    input_tensor = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs    = model(input_tensor)
        probs      = torch.softmax(outputs, dim=1)
        fake_prob  = probs[0][1].item()
        real_prob  = probs[0][0].item()
        pred_class = 1 if fake_prob > 0.5 else 0

    return {
        'label'      : 'FAKE' if pred_class == 1 else 'REAL',
        'fake_prob'  : round(fake_prob * 100, 2),
        'real_prob'  : round(real_prob * 100, 2),
        'face_image' : face_image,
        'face_found' : face_found
    }


def predict_and_explain(image_pil, model):
    """
    Full prediction + Grad-CAM for single frame.
    Used for the visual analysis section.
    """
    face_image, face_found = detect_and_crop_face(image_pil)
    original_np  = np.array(face_image) / 255.0
    input_tensor = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs    = model(input_tensor)
        probs      = torch.softmax(outputs, dim=1)
        fake_prob  = probs[0][1].item()
        real_prob  = probs[0][0].item()
        pred_class = 1 if fake_prob > 0.5 else 0

    label      = 'FAKE' if pred_class == 1 else 'REAL'
    confidence = fake_prob if pred_class == 1 else real_prob

    # Grad-CAM
    target_layers   = [model.conv_head]
    cam             = GradCAM(model=model, target_layers=target_layers)
    targets         = [ClassifierOutputTarget(pred_class)]
    cam_mask        = cam(input_tensor=input_tensor, targets=targets)[0]

    heatmap_overlay = show_cam_on_image(
        original_np.astype(np.float32),
        cam_mask,
        use_rgb=True
    )

    return {
        'label'      : label,
        'confidence' : round(confidence * 100, 2),
        'fake_prob'  : round(fake_prob * 100, 2),
        'real_prob'  : round(real_prob * 100, 2),
        'heatmap'    : heatmap_overlay,
        'face_image' : face_image,
        'face_found' : face_found
    }


def extract_frames_from_video(video_path):
    """
    Extracts 1 frame per second based on video duration.
    Returns frames, timestamps, duration, fps.
    """
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration  = round(total / fps, 2) if fps > 0 else 0

    step      = max(1, int(fps))
    frames    = []
    timestamps = []

    frame_idx = 0
    second    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            timestamps.append(round(frame_idx / fps, 1))
            second += 1
        if second >= 30:
            break
        frame_idx += 1

    cap.release()
    return frames, timestamps, round(duration, 1), round(fps, 1)


def compute_facial_motion_analysis(frames):
    """
    Computes dense optical flow on the full detected face region
    between consecutive frame pairs.

    For each pair of consecutive frames:
      1. Detect and crop the face using MTCNN
      2. Convert to grayscale
      3. Compute dense optical flow (Farneback) on the full face crop
      4. Compute mean magnitude (how much the face moved overall)
      5. Compute std deviation of magnitude (how irregular the motion is)

    Real faces move smoothly and consistently.
    Deepfake faces exhibit jittery, irregular motion across the full face.

    Returns:
      flow_results  - list of per-pair dicts with scores and timestamps
      overall_score - 0 to 100 score (higher = more irregular = more suspicious)
    """
    if len(frames) < 2:
        return [], 0.0

    flow_results = []
    prev_gray    = None
    prev_ts      = 0

    for i, frame in enumerate(frames):
        # Detect and crop full face
        face_pil, face_found = detect_and_crop_face(frame)
        face_np = np.array(face_pil)

        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            # Dense optical flow on full face region 224x224
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                pyr_scale  = 0.5,
                levels     = 3,
                winsize    = 15,
                iterations = 3,
                poly_n     = 5,
                poly_sigma = 1.2,
                flags      = 0
            )

            # Motion magnitude at each pixel
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            mean_motion  = float(np.mean(magnitude))
            irregularity = float(np.std(magnitude))

            # Irregularity score normalized to 0-100
            # Threshold calibrated: real faces typically < 2.0 std, fakes > 3.5
            norm_score = min(irregularity / 8.0 * 100, 100.0)

            flow_results.append({
                'pair'         : f"{i-1}s → {i}s",
                'mean_motion'  : round(mean_motion, 3),
                'irregularity' : round(irregularity, 3),
                'norm_score'   : round(norm_score, 1),
                'face_found'   : face_found
            })

        prev_gray = gray

    if not flow_results:
        return flow_results, 0.0

    overall_score = round(
        float(np.mean([r['norm_score'] for r in flow_results])), 2
    )

    return flow_results, overall_score


def get_motion_verdict(overall_score):
    if overall_score >= 50:
        return "HIGH — Strong facial motion irregularity detected (Deepfake Likely)"
    elif overall_score >= 30:
        return "MEDIUM — Some irregular motion detected"
    else:
        return "LOW — Natural facial motion (Likely Real)"