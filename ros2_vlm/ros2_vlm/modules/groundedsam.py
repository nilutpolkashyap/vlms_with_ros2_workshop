from pathlib import Path
import sys
import os
import cv2
import torch
import numpy as np
import openvino as ov
from PIL import Image
import supervision as sv
import transformers
from typing import Union, List
from torchvision.transforms.functional import resize, InterpolationMode

current_dir = os.getcwd()

additional_path = '/src/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules'
updated_path = os.path.join(current_dir + additional_path)

sys.path.append(os.path.join(current_dir + '/src/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules/GroundingDINO')) 
sys.path.append(os.path.join(current_dir + '/src/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules/GroundingDINO')) 
sys.path.append(os.path.join(current_dir + '/src/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules/checkpoints')) 
sys.path.append(os.path.join(current_dir + '/src/vlms_with_ros2_workshop/ros2_vlm/ros2_vlm/modules/openvino_irs')) 

ground_dino_dir =  updated_path / Path("GroundingDINO")
efficient_sam_dir =  updated_path / Path("EfficientSAM")

from .GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from .GroundingDINO.groundingdino.models import build_model
from .GroundingDINO.groundingdino.util.slconfig import SLConfig
from .GroundingDINO.groundingdino.util.utils import clean_state_dict
from .GroundingDINO.groundingdino.util import get_tokenlizer
from .GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from .GroundingDINO.groundingdino.util.inference import Model
from .GroundingDINO.groundingdino.datasets import transforms as T

class ImageProcessor:
    def __init__(self, core, device):
        self.core = core
        self.irs_path = Path("openvino_irs")
        self.ov_dino_name = "openvino_grounding_dino"
        self.ov_dino_path = updated_path / self.irs_path / f"{self.ov_dino_name}.xml"

        self.ov_dino_model = self.core.read_model(self.ov_dino_path)
        self.device = device    #"AUTO"

        self.ground_dino_img_size = (1024, 1280)

        self.pt_device = "cpu"
        self.ckpt_base_path = Path("checkpoints")

        self.grounding_dino_config_path = f"{ground_dino_dir}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounding_dino_checkpoint_path = updated_path / self.ckpt_base_path / "groundingdino_swint_ogc.pth"

        self.ov_compiled_grounded_dino = self.core.compile_model(self.ov_dino_model, self.device)

        self.box_threshold = 0.3
        self.text_threshold = 0.25

        self.ov_efficient_sam_name = "openvino_efficient_sam"
        self.ov_efficient_sam_path = updated_path / self.irs_path / f"{self.ov_efficient_sam_name}.xml"

        self.ov_efficient_sam = core.read_model(self.ov_efficient_sam_path)

        self.ov_compiled_efficient_sam = core.compile_model(self.ov_efficient_sam, device_name=self.device)

        self.model, self.max_text_len, self.dino_tokenizer, *_ = self.load_pt_grounding_dino(self.grounding_dino_config_path, self.grounding_dino_checkpoint_path)

    def sig(self, x):
        return 1 / (1 + np.exp(-x))
    
    def load_pt_grounding_dino(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)

        args.device = self.pt_device
        args.use_checkpoint = False
        args.use_transformer_ckpt = False

        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=self.pt_device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()

        return (
            model,
            args.max_text_len,
            get_tokenlizer.get_tokenlizer(args.text_encoder_type),
        )

    def transform_image(self, pil_image: Image.Image) -> torch.Tensor:
        
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(pil_image, None)  # 3, h, w
        return image
    
    def get_ov_grounding_output(
        self,
        model: ov.CompiledModel,
        pil_image: Image.Image,
        caption: Union[str, List[str]],
        box_threshold: float,
        text_threshold: float,
        dino_tokenizer: transformers.PreTrainedTokenizerBase,
        max_text_len: int) -> (torch.Tensor, List[str], torch.Tensor):

        if isinstance(caption, list):
            caption = ". ".join(caption)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions = [caption]

        tokenized = dino_tokenizer(captions, padding="longest", return_tensors="pt")
        specical_tokens = dino_tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, specical_tokens, dino_tokenizer)

        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]

            position_ids = position_ids[:, :max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

        inputs = {}
        inputs["attention_mask.1"] = tokenized["attention_mask"]
        inputs["text_self_attention_masks"] = text_self_attention_masks
        inputs["input_ids"] = tokenized["input_ids"]
        inputs["position_ids"] = position_ids
        inputs["token_type_ids"] = tokenized["token_type_ids"]


        input_img = resize(
            self.transform_image(pil_image),
            self.ground_dino_img_size,
            interpolation=InterpolationMode.BICUBIC,
        )[None, ...]
        inputs["samples"] = input_img

        request = model.create_infer_request()
        request.start_async(inputs, share_inputs=False)
        request.wait()

        logits = torch.from_numpy(self.sig(np.squeeze(request.get_tensor("pred_logits").data, 0)))
        boxes = torch.from_numpy(np.squeeze(request.get_tensor("pred_boxes").data, 0))

        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits, boxes = logits[filt_mask], boxes[filt_mask]

        tokenized = dino_tokenizer(caption)
        pred_phrases = []
        for logit in logits:
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, dino_tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        return boxes, pred_phrases, logits.max(dim=1)[0]
    
    def predict_efficient_sam_mask(self, compiled_efficient_sam: ov.CompiledModel, image: Image.Image, bbox: torch.Tensor):
        input_size = 1024
        w, h = image.size[:2]
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))

        numpy_image = np.array(image, dtype=np.float32) / 255.0
        numpy_image = np.transpose(numpy_image, (2, 0, 1))[None, ...]

        scaled_points = bbox * scale

        bounding_box = scaled_points.reshape([1, 1, 2, 2])
        bbox_labels = np.reshape(np.array([2, 3]), [1, 1, 2])

        res = compiled_efficient_sam((numpy_image, bounding_box, bbox_labels))

        predicted_logits, predicted_iou = res[0], res[1]

        all_masks = torch.ge(torch.sigmoid(torch.from_numpy(predicted_logits[0, 0, :, :, :])), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...]

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if curr_predicted_iou > max_predicted_iou or selected_mask_using_predicted_iou is None:
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]
        return selected_mask_using_predicted_iou

    def predict_efficient_sam_masks(self, compiled_efficient_sam: ov.CompiledModel, pil_image: Image.Image, transformed_boxes) -> torch.Tensor:
        masks = []
        for bbox in transformed_boxes:
            mask = self.predict_efficient_sam_mask(compiled_efficient_sam, pil_image, bbox)
            mask = Image.fromarray(mask).resize(pil_image.size)
            masks.append(np.array(mask))
        masks = torch.from_numpy(np.array(masks))
        return masks

    def process_image(self, pil_image: Image.Image, classes_prompt: List[str], use_segment: bool = False) -> np.ndarray:
        boxes, pred_phrases, logits = self.get_ov_grounding_output(self.ov_compiled_grounded_dino, pil_image, classes_prompt, self.box_threshold, self.text_threshold, self.dino_tokenizer, self.max_text_len)

        source_w, source_h = pil_image.size
        detections = Model.post_process_result(source_h=source_h, source_w=source_w, boxes=boxes, logits=logits)

        class_id = Model.phrases2classes(phrases=pred_phrases, classes=list(map(str.lower, classes_prompt)))
        detections.class_id = class_id

        if use_segment:
            masks = self.predict_efficient_sam_masks(self.ov_compiled_efficient_sam, pil_image, detections.xyxy)
            detections.mask = masks.numpy()

            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()

            labels = [f"{classes_prompt[class_id] if class_id is not None else 'None'} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]

            annotated_image = np.array(pil_image)
            annotated_image = mask_annotator.annotate(scene=np.array(pil_image).copy(), detections=detections)
            mask_annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            annotated_frame_bgr = cv2.cvtColor(mask_annotated_image, cv2.COLOR_RGB2BGR)
        else:
            box_annotator = sv.BoxAnnotator()
            box_annotated_image = box_annotator.annotate(scene=np.array(pil_image).copy(), detections=detections)

            annotated_frame_bgr = cv2.cvtColor(box_annotated_image, cv2.COLOR_RGB2BGR)

        return annotated_frame_bgr

