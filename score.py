import torch
from PIL import Image
from src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
parser.add_argument('--checkpoint', type=str, default='./HPS_v2.pt', help='Path to the model checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess_train, preprocess_val = create_model_and_transforms(
    'ViT-H-14',
    'laion2B-s32B-b79K',
    precision='amp',
    device=device,
    jit=False,
    force_quick_gelu=False,
    force_custom_text=False,
    force_patch_dropout=False,
    force_image_size=None,
    pretrained_image=False,
    image_mean=None,
    image_std=None,
    light_augmentation=True,
    aug_cfg={},
    output_dict=True,
    with_score_predictor=False,
    with_region_predictor=False
)

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'])
tokenizer = get_tokenizer('ViT-H-14')
model.eval()

# Load your image and prompt
with torch.no_grad():
    # Process the image
    image = preprocess_val(Image.open(args.image_path)).unsqueeze(0).to(device=device, non_blocking=True)
    # Process the prompt
    text = tokenizer([args.prompt]).to(device=device, non_blocking=True)
    # Calculate the HPS
    with torch.cuda.amp.autocast():
        outputs = model(image, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T

        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
print('HPSv2 score:', hps_score[0])
