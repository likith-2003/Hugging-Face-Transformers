import streamlit as st
from PIL import Image
import pandas as pd
import plotly
import plotly.express as px
import json
import io
import time
import random
import hashlib

TORCH_AVAILABLE = None
TORCH_IMPORT_ERROR = None

def ensure_torch_available():
    global TORCH_AVAILABLE, TORCH_IMPORT_ERROR, torch
    if TORCH_AVAILABLE is not None:
        return TORCH_AVAILABLE
    try:
        import torch
        TORCH_AVAILABLE = True
        return True
    except Exception as e:
        TORCH_AVAILABLE = False
        TORCH_IMPORT_ERROR = e
        torch = None
        return False

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (Professional Theme)
# ==========================================
st.set_page_config(page_title="NutriVision AI", page_icon="🍏", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    .main-header {
        color: #264653;
        font-weight: 700;
        border-bottom: 3px solid #2a9d8f;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 6px solid #2a9d8f;
        transition: transform 0.2s;
    }
    .prediction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-top: 4px solid #e9c46a;
    }
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #21867a;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        color: #264653;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. NUTRITION DATABASE (Food-101 Mapping)
# ==========================================
NUTRITION_DB = {
    "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
    "hamburger": {"calories": 350, "protein": 20, "carbs": 30, "fat": 18},
    "sushi": {"calories": 150, "protein": 8, "carbs": 25, "fat": 2},
    "chocolate_cake": {"calories": 400, "protein": 5, "carbs": 55, "fat": 18},
    "caesar_salad": {"calories": 180, "protein": 6, "carbs": 12, "fat": 14},
    "french_fries": {"calories": 320, "protein": 4, "carbs": 42, "fat": 15},
    "chicken_curry": {"calories": 250, "protein": 22, "carbs": 10, "fat": 14},
    "ice_cream": {"calories": 200, "protein": 4, "carbs": 24, "fat": 10},
    "spaghetti_bolognese": {"calories": 380, "protein": 18, "carbs": 45, "fat": 12},
    "apple_pie": {"calories": 350, "protein": 3, "carbs": 50, "fat": 15}
}
# Default fallback for other Food-101 classes
DEFAULT_NUTRITION = {"calories": 250, "protein": 10, "carbs": 30, "fat": 12}

FOOD101_CLASSES = list(NUTRITION_DB.keys()) + ["ramen", "tacos", "steak", "pad_thai", "donuts"]

# ==========================================
# 3. MODEL INFERENCE ENGINE
# ==========================================
@st.cache_resource
def load_model():
    if not ensure_torch_available():
        return None, None, False, f"Torch import failed: {TORCH_IMPORT_ERROR}"
    try:
        from transformers import ViTImageProcessor, ViTForImageClassification
    except Exception as e:
        return None, None, False, f"Transformers import failed: {e}"

    try:
        # Replace with your Hugging Face repo after fine-tuning
        model_id = "google/vit-base-patch16-224"
        processor = ViTImageProcessor.from_pretrained(model_id)
        model = ViTForImageClassification.from_pretrained(model_id, num_labels=101, ignore_mismatched_sizes=True)
        model.eval()
        return processor, model, True, None
    except Exception as e:
        return None, None, False, str(e)

def mock_predict(image_bytes):
    """Smart mock inference for UI demonstration"""
    seed = int(hashlib.md5(image_bytes).hexdigest(), 16)
    rng = random.Random(seed)
    num_classes = rng.randint(1, 3)
    classes = rng.sample(FOOD101_CLASSES, num_classes)
    scores = [rng.random() for _ in range(num_classes)]
    total = sum(scores)
    scores = [s / total for s in scores]
    return [{"label": cls, "score": score} for cls, score in zip(classes, scores)]

def predict_food(image_bytes, processor, model, use_real_model):
    if use_real_model and model is not None:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            top5 = torch.topk(probs, 3)
            return [{"label": model.config.id2label.get(idx, f"class_{idx}").replace("_", " "), 
                     "score": val.item()} for idx, val in zip(top5.indices, top5.values)]
    else:
        return mock_predict(image_bytes)

# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================
def main():
    st.markdown("<h1 class='main-header'>🍏 NutriVision AI</h1>", unsafe_allow_html=True)
    st.markdown("### *Advanced Food Recognition & Nutritional Analytics powered by Vision Transformers*")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Control Panel")
        torch_ok = ensure_torch_available()
        if torch_ok:
            use_real_model = st.toggle("Use Real ViT Model (Requires Download)", value=False)
        else:
            use_real_model = st.toggle(
                "Use Real ViT Model (Requires Download)",
                value=False,
                disabled=True
            )
            st.error("❌ Torch is unavailable on this machine. The app will run in Demo Mode.")
            if TORCH_IMPORT_ERROR is not None:
                st.write(f"Error: {TORCH_IMPORT_ERROR}")

        if use_real_model:
            with st.spinner("Loading ViT Model..."):
                processor, model, loaded, load_error = load_model()
            if loaded:
                st.success("✅ ViT Model Loaded")
            else:
                st.error("❌ Model Load Failed. Using Demo Mode.")
                processor, model = None, None
                if load_error:
                    st.write(f"Load error: {load_error}")
        else:
            processor, model = None, None
            st.info("💡 Running in Demo Mode (Mock Inference)")

        st.markdown("---")
        st.markdown("### 📊 Session Summary")
        if 'history' in st.session_state and len(st.session_state.history) > 0:
            total_cals = sum(item['calories'] for item in st.session_state.history)
            st.metric("Total Calories Tracked", f"{total_cals} kcal")
            st.metric("Items Analyzed", len(st.session_state.history))
        else:
            st.write("No items tracked yet.")

    # Initialize Session State
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Recognition", "🥗 Nutrition & Analytics", "📜 History & Export"])

    # TAB 1: IMAGE RECOGNITION
    with tab1:
        st.subheader("Upload Food Images")
        with st.form("image_upload_form", clear_on_submit=False):
            uploaded_files = st.file_uploader(
                "Choose an image or batch of images...",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Supports single and batch processing. Max 10MB per image.",
                key="image_uploader"
            )
            process_images = st.form_submit_button("Analyze Images")
            if uploaded_files is not None and len(uploaded_files) > 0:
                st.info(f"{len(uploaded_files)} image(s) selected. Click Analyze to process them.")

        if process_images and uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {uploaded_file.name}...")
                try:
                    image_bytes = uploaded_file.read()
                    predictions = predict_food(image_bytes, processor, model, use_real_model)
                    top_pred = predictions[0]
                    food_name = top_pred['label'].replace("_", " ").title()
                    confidence = top_pred['score']

                    # Fetch Nutrition
                    nutrition = NUTRITION_DB.get(top_pred['label'].lower(), DEFAULT_NUTRITION)

                    # Save to History
                    st.session_state.history.append({
                        "filename": uploaded_file.name,
                        "food": food_name,
                        "confidence": round(confidence * 100, 2),
                        "calories": nutrition['calories'],
                        "protein": nutrition['protein'],
                        "carbs": nutrition['carbs'],
                        "fat": nutrition['fat']
                    })

                    # Display Results
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)
                    with col2:
                        st.markdown(f"""
                        <div class='prediction-card'>
                            <h3 style="margin:0; color:#264653;">{food_name}</h3>
                            <p style="color:#6c757d; margin:5px 0;">Confidence: <b>{confidence*100:.1f}%</b></p>
                            <div style="display: flex; gap: 15px; margin-top: 10px;">
                                <div><b>🔥 {nutrition['calories']}</b> kcal</div>
                                <div><b>🥩 {nutrition['protein']}g</b> Protein</div>
                                <div><b>🍞 {nutrition['carbs']}g</b> Carbs</div>
                                <div><b>🥑 {nutrition['fat']}g</b> Fat</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Unable to process {uploaded_file.name}: {e}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text("✅ Analysis Complete!")
            st.balloons()

    # TAB 2: NUTRITION & ANALYTICS
    with tab2:
        st.subheader("Nutritional Breakdown")
        if not st.session_state.history:
            st.warning("No data available. Please upload images in the 'Image Recognition' tab.")
        else:
            df = pd.DataFrame(st.session_state.history)
            
            # Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Calories", f"{df['calories'].sum()} kcal")
            c2.metric("Total Protein", f"{df['protein'].sum()} g")
            c3.metric("Total Carbs", f"{df['carbs'].sum()} g")
            c4.metric("Total Fat", f"{df['fat'].sum()} g")

            # Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### Macronutrient Distribution")
                macros = df[['protein', 'carbs', 'fat']].sum()
                # Convert to calories (1g protein=4cal, 1g carb=4cal, 1g fat=9cal)
                macro_cals = [macros['protein']*4, macros['carbs']*4, macros['fat']*9]
                fig_pie = px.pie(
                    values=macro_cals, 
                    names=['Protein', 'Carbs', 'Fat'],
                    color_discrete_sequence=['#2a9d8f', '#e9c46a', '#e76f51'],
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                st.markdown("#### Calorie Breakdown by Item")
                fig_bar = px.bar(
                    df, x='food', y='calories', 
                    color='calories',
                    color_continuous_scale='Tealgrn',
                    labels={'food': 'Food Item', 'calories': 'Calories (kcal)'},
                    text_auto='.0f'
                )
                fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 3: HISTORY & EXPORT
    with tab3:
        st.subheader("Upload History & Logs")
        if not st.session_state.history:
            st.info("Your tracking history will appear here.")
        else:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Export Buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                csv = df_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export CSV",
                    data=csv,
                    file_name="nutrivision_history.csv",
                    mime="text/csv",
                )
            
            with col2:
                json_data = df_history.to_json(orient="records", indent=4)
                st.download_button(
                    label="📥 Export JSON",
                    data=json_data,
                    file_name="nutrivision_history.json",
                    mime="application/json",
                )
                
            with col3:
                if st.button("🗑️ Clear History", type="secondary"):
                    st.session_state.history = []
                    st.rerun()

if __name__ == "__main__":
    main()
