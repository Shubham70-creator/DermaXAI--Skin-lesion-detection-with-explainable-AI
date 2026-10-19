import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

IMG_SIZE = 224
NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_INFO = {
    'nv': {
        'full_name': 'Melanocytic Nevus (Common Mole)',
        'description': 'This is a common, ordinary mole. Most people have several of these on their skin. It is made of pigment cells and is typically harmless.',
        'risk': 'Low Risk',
        'recommendation': 'No immediate action needed. Keep an eye on it for any changes in size, shape, or color over time.'
    },
    'mel': {
        'full_name': 'Melanoma',
        'description': 'This is the most dangerous form of skin cancer. It develops in the pigment-producing cells and can spread to other parts of the body if not treated early.',
        'risk': 'High Risk',
        'recommendation': 'Consult a dermatologist immediately for a biopsy and further evaluation. Early detection significantly improves treatment outcomes.'
    },
    'bkl': {
        'full_name': 'Benign Keratosis',
        'description': 'A non-cancerous, rough, wart-like growth that commonly appears with age. It is not related to skin cancer.',
        'risk': 'Low Risk',
        'recommendation': 'Generally no treatment is required. See a doctor only if it becomes itchy, irritated, or changes appearance.'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'description': 'A type of skin cancer that grows slowly and rarely spreads to other body parts, but can damage surrounding skin and tissue if untreated.',
        'risk': 'Moderate-High Risk',
        'recommendation': 'Schedule a dermatologist visit soon for proper diagnosis and treatment, such as minor surgical removal.'
    },
    'akiec': {
        'full_name': 'Actinic Keratosis',
        'description': 'A rough, scaly patch caused by long-term sun damage. It is considered a pre-cancerous condition.',
        'risk': 'Moderate Risk',
        'recommendation': 'Get it checked by a dermatologist. If left untreated, it can potentially develop into skin cancer over time.'
    },
    'vasc': {
        'full_name': 'Vascular Lesion',
        'description': 'A growth made of blood vessels, such as a birthmark or small red/purple spot. Usually present from birth or appears later in life.',
        'risk': 'Low Risk',
        'recommendation': 'Typically no treatment needed. Consult a doctor only for cosmetic concerns or if it changes suddenly.'
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'description': 'A small, firm, benign skin nodule, often caused by minor skin injury such as an insect bite. Completely non-cancerous.',
        'risk': 'Low Risk',
        'recommendation': 'No treatment necessary. It can be removed for cosmetic reasons if desired, but this is optional.'
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path='models/best_model.pth'):
    model = models.densenet169(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_explanation(model, image_path, save_heatmap_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_idx = probs.max(1)

        top3_probs, top3_idx = torch.topk(probs, 3, dim=1)
        top3_list = []
        for i in range(3):
            cls_name = CLASS_NAMES[top3_idx[0][i].item()]
            cls_prob = round(top3_probs[0][i].item() * 100, 2)
            top3_list.append({
                'class_code': cls_name,
                'full_name': CLASS_INFO[cls_name]['full_name'],
                'probability': cls_prob
            })

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()

    target_layer = model.features.norm5
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(img_resized) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    heatmap_img = Image.fromarray(visualization)
    heatmap_img.save(save_heatmap_path)

    class_details = CLASS_INFO[predicted_class]
    return predicted_class, confidence_score, save_heatmap_path, class_details, top3_list