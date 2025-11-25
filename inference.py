import re

import torch
from PIL import Image
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string


class NanoVLMInference:
    """
    A wrapper class for nanoVLM inference that encapsulates model loading,
    image processing, prompt formatting, and text generation.
    """
    
    def __init__(self, model_name="lusxvr/nanoVLM", device="mps"):
        """
        Initialize the nanoVLM inference wrapper.
        
        Args:
            model_name: HuggingFace model name or path to local checkpoint
            device: Device to run inference on ("cuda", "mps", or "cpu")
        """
        self.device = device
        self.model_name = model_name
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = VisionLanguageModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Get tokenizer from model's configuration
        self.tokenizer = get_tokenizer(
            self.model.cfg.lm_tokenizer,
            self.model.cfg.vlm_extra_tokens,
            self.model.cfg.lm_chat_template
        )
        
        # Get image processor from model's configuration
        resize_to_max_side_len = getattr(self.model.cfg, "resize_to_max_side_len", False)
        self.image_processor = get_image_processor(
            self.model.cfg.max_img_size,
            self.model.cfg.vit_img_size,
            resize_to_max_side_len
        )
        print(f"Model loaded successfully on {device}")
    
    def process_image(self, image):
        """
        Process an image into tensors ready for model input.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            tuple: (image_tensor, splitted_image_ratio)
                - image_tensor: Processed image tensor on device
                - splitted_image_ratio: Grid structure of the image
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        # Process the image
        processed_image, splitted_image_ratio = self.image_processor(img)
        
        # Handle models that don't have a global image token
        if not hasattr(self.tokenizer, "global_image_token") and \
           splitted_image_ratio[0] * splitted_image_ratio[1] == len(processed_image) - 1:
            processed_image = processed_image[1:]
        
        # Move to device
        image_tensor = processed_image.to(self.device)
        
        return image_tensor, splitted_image_ratio
    
    def format_prompt(self, prompt, splitted_image_ratio):
        """
        Format a text prompt with image tokens for the model.
        
        Args:
            prompt: Text prompt/question
            splitted_image_ratio: Grid structure from process_image()
            
        Returns:
            torch.Tensor: Tokenized prompt tensor on device
        """
        # Get the special image string (e.g., "<image><image>...<image>")
        image_string = get_image_string(
            self.tokenizer,
            [splitted_image_ratio],
            self.model.cfg.mp_image_token_length
        )
        
        # Format the prompt using the model's chat template
        messages = [{"role": "user", "content": image_string + prompt}]
        encoded_prompt = self.tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True
        )
        
        # Convert to tensor on device
        prompt_tokens = torch.tensor(encoded_prompt).to(self.device)
        
        return prompt_tokens
    
    def generate(
        self,
        prompt_tokens,
        image_tensor,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False
    ):
        """
        Generate text response from prompt tokens and image tensor.
        
        Args:
            prompt_tokens: Tokenized prompt tensor
            image_tensor: Processed image tensor
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text response
        """
        with torch.no_grad():
            generated_ids = self.model.generate(
                prompt_tokens,
                image_tensor,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                greedy=greedy
            )
        
        out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out
    
    def generate_from_image_and_prompt(
        self,
        image,
        prompt,
        max_new_tokens=10,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False
    ):
        """
        Convenience method that processes image, formats prompt, and generates response.
        
        Args:
            image: PIL Image or path to image file
            prompt: Text prompt/question
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text response
        """
        # Process image
        image_tensor, splitted_image_ratio = self.process_image(image)
        
        # Format prompt
        prompt_tokens = self.format_prompt(prompt, splitted_image_ratio)
        
        # Generate response
        return self.generate(prompt_tokens, image_tensor, max_new_tokens, top_k, top_p, temperature, greedy)


def get_enclosed_letter(text: str):
    """
    Finds the first single alphabetical letter that is enclosed by non-alphabetical
    characters or the start/end of the string.

    Args:
        text: The input string to search.

    Returns:
        The single alphabetical letter (str) if found, otherwise None.
    """
    # re.search looks for the pattern anywhere in the string.
    pattern = r"(?:^|[^a-zA-Z])([a-zA-Z])(?:$|[^a-zA-Z])"
    match = re.search(pattern, text)

    if match:
        # The letter is in capturing Group 1 of the regex.
        return match.group(1)
    return None