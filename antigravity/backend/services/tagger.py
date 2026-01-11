from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List

# Use a smaller model for reasonable performance on CPU/Laptop GPU
MODEL_NAME = "openai/clip-vit-base-patch32"

class ClipTagger:
    def __init__(self):
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("CLIP model loaded.")

    def generate_tags(self, image_path: str, candidate_labels: List[str] = None) -> List[tuple]:
        """
        Classifies the image against a list of candidate labels.
        Returns a list of (label, score) tuples, sorted by score.
        """
        if candidate_labels is None:
            candidate_labels = [
                "screenshot", "receipt", "document",
                "landscape", "city", "beach", "forest", "mountain",
                "food", "coffee", "restaurant",
                "cat", "dog", "pet",
                "selfish", "group photo", "portrait",
                "car", "architecture", "flower"
            ]

        try:
            image = Image.open(image_path)
            inputs = self.processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # This logic works for zero-shot classification
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

            # Convert to list
            scores = probs[0].tolist()
            result = list(zip(candidate_labels, scores))
            
            # Sort by score desc
            result.sort(key=lambda x: x[1], reverse=True)
            
            # Return top tags (> 0.05 confidence maybe? or just top 3)
            return [r for r in result if r[1] > 0.05]

        except Exception as e:
            print(f"Error tagging {image_path}: {e}")
            return []

# Singleton instance for simple usage
tagger_instance = None

def get_tagger():
    global tagger_instance
    if tagger_instance is None:
        tagger_instance = ClipTagger()
    return tagger_instance
