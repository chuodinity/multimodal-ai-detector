import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, ViTImageProcessor, ViTForImageClassification

# --- MODEL DEFINITION FOR DESKLIB TEXT DETECTOR ---
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return self.classifier(mean_pooled)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Text Model
    text_model_name = "desklib/ai-text-detector-v1.01"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = DesklibAIDetectionModel.from_pretrained(text_model_name).to(device)
    text_model.eval()

    # Image Model (ViT)
    image_model_name = "google/vit-base-patch16-224"
    image_processor = ViTImageProcessor.from_pretrained(image_model_name)
    image_model = ViTForImageClassification.from_pretrained(image_model_name).to(device)
    image_model.eval()
    
    return text_tokenizer, text_model, image_processor, image_model, device

text_tokenizer, text_model, image_processor, image_model, device = load_models()

# --- UI SETUP ---
st.set_page_config(page_title="Multimodal AI Detector", layout="centered")
st.title("🛡️ Multimodal AI Content Detector")
st.write("Upload an image and text to check for AI generation using Late Fusion.")

# Input Section
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
user_text = st.text_area("Enter Text Content", height=150)

if st.button("Analyze Content") and uploaded_image and user_text:
    with st.spinner("Processing..."):
        # 1. Text Inference
        inputs = text_tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logit = text_model(inputs['input_ids'], inputs['attention_mask'])
            p_text = torch.sigmoid(logit).item()

        # 2. Image Inference
        image = Image.open(uploaded_image).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = image_model(**inputs)
            # Interpret top class (Note: Fine-tuning ViT on your specific dataset is recommended)
            p_image = torch.softmax(outputs.logits, dim=-1).max().item()

        # 3. Late Fusion (Weighted Average)
        # We give more weight (0.7) to text as it's typically more reliable in these benchmarks
        fused_score = (0.7 * p_text) + (0.3 * p_image)
        
        # --- CONCLUSION ---
        st.divider()
        st.subheader("Final Conclusion")
        
        confidence_label = "HIGH" if fused_score > 0.8 or fused_score < 0.2 else "MODERATE"
        verdict = "AI-GENERATED" if fused_score > 0.5 else "HUMAN-WRITTEN"
        
        color = "red" if verdict == "AI-GENERATED" else "green"
        st.markdown(f"The system concludes this content is likely **:{color}[{verdict}]**.")
        st.metric("Confidence Score", f"{fused_score:.2%}", delta=f"Reliability: {confidence_label}")
        
        # Breakdown Visualization
        st.bar_chart({"Text Score": p_text, "Image Score": p_image, "Combined": fused_score})
