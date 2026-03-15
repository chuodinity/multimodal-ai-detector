import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, 
    pipeline, ViTImageProcessor, ViTForImageClassification
)

# --- DESKLIB TEXT DETECTOR ARCHITECTURE ---
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    # NEW: Add this line to satisfy the latest Transformers internal checks
    _tied_weights_keys = [] 

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # NEW: Always call post_init at the end of __init__
        self._tied_weights_keys = []
        if not hasattr(self, "_keys_to_ignore_on_save"):
            self._keys_to_ignore_on_save = []
            
        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        mean_pooled = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return self.classifier(mean_pooled)

# --- LOAD SPECIALIZED MODELS ---
@st.cache_resource
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text Model (Desklib)
    text_model_id = "desklib/ai-text-detector-v1.01"
    t_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    t_model = DesklibAIDetectionModel.from_pretrained(text_model_id).to(device)
    
    # Image Model (Specialized ViT for AIGC)
    img_model_id = "capcheck/ai-image-detection"
    img_pipe = pipeline("image-classification", model=img_model_id, device=0 if device == "cuda" else -1)
    
    return t_tokenizer, t_model, img_pipe, device

tokenizer, text_model, img_pipeline, device = load_assets()

# --- UI INTERFACE ---
st.set_page_config(page_title="AIGC Late Fusion Detector", layout="wide")
st.title("🛡️ Specialized Multimodal AIGC Detector")

col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("Input Content")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    user_text = st.text_area("Input Text", placeholder="Paste article or caption...", height=200)
    
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

# --- PROCESSING ---
if st.button("Run Multi-Modal Detection") and uploaded_file and user_text:
    with st.spinner("Analyzing artifacts in text and pixels..."):
        # 1. Text Score (Logit -> Sigmoid)
        t_inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            t_logit = text_model(t_inputs['input_ids'], t_inputs['attention_mask'])
            p_text = torch.sigmoid(t_logit).item()

        # 2. Image Score (AIGC ViT)
        img_results = img_pipeline(Image.open(uploaded_file))
        # Find the score for 'Fake' (AI-generated)
        p_image = next(item['score'] for item in img_results if item['label'] == 'Fake')

        # 3. Late Fusion (Weighted Average)
        # Using 0.5/0.5 for balanced multimodal detection
        fused_score = (0.5 * p_text) + (0.5 * p_image)

        with col_out:
            st.subheader("System Verdict")
            
            # Classification logic
            verdict = "AI-GENERATED" if fused_score > 0.5 else "HUMAN-ORIGIN"
            color = "red" if verdict == "AI-GENERATED" else "green"
            
            st.markdown(f"### Result: :{color}[{verdict}]")
            st.metric("Aggregate Confidence", f"{fused_score:.2%}")
            
            # Visual Breakdown
            st.write("**Modality Breakdown:**")
            st.progress(p_text, text=f"Text AI Probability: {p_text:.1%}")
            st.progress(p_image, text=f"Image AI Probability: {p_image:.1%}")
            
            # Brief Forensic Note
            if fused_score > 0.5:
                st.warning("Conclusion: High cross-modal artifact detection. The content shows patterns consistent with synthetic generation.")
            else:
                st.success("Conclusion: Low probability of AI generation. Features align with natural human patterns.")
