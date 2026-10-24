# Explore different representation learning algorithms for text, image and biological
# datasets.
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torchvision import models, transforms
class TextRestoration(nn.Module):
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(max_features=max_features,stop_words='english')
    def fit_transform(self, texts):
	    vectors = self.vectorizer.fit_transform(texts)
	    return vectors.toarray()
    def get_feature_names(self):
       return self.vectorizer.get_feature_names_out()
class ImageRestoration(nn.Module):
    def __init__(self, device=None):
	    self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
	    self.model = models.resnet18(weights=models.ResNet50_Weights.DEFAULT)
	    self.model = nn.Sequential(*list(self.model.children())[:-1])
	    self.model.to(self.device)
	    self.model.eval()
	    self.transform = transforms.Compose([transforms.Resize(224,224),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	def extract_features_from_image(self, image_path):
	    image = Image.open(image_path).convert('RGB')
	    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
	    with torch.no_grad():
		    features = self.model(image)
		features = features.view(features.size(0), -1)
	    return features.cpu().numpy()[0]
	def extract_features_from_folder(self, folder_path):
		features =[]
		image_names = []
		valid_ext= ['.jpg','.jpeg','.png']
		
	    for file in os.listdir(folder_path):
			if file.lower().endwith(valid_ext):