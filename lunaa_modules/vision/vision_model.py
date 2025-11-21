"""Vision model integration for Lunaa AI"""
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

class VisionModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load vision model (BLIP for image captioning)"""
        if not _VISION_AVAILABLE:
            raise ImportError("Vision dependencies not installed")
        
        if self.model is None:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
    
    def analyze_image(self, image_path: str) -> str:
        """Analyze image and return description"""
        if not _VISION_AVAILABLE:
            return "Vision dependencies not installed"
        
        self.load_model()
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=100)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error analyzing image: {e}"
    
    def analyze_image_with_question(self, image_path: str, question: str) -> str:
        """Answer questions about an image"""
        if not _VISION_AVAILABLE:
            return "Vision dependencies not installed"
        
        self.load_model()
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=100)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            return f"Error analyzing image: {e}"
