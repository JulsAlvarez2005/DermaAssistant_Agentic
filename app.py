import gradio as gr
import os
import easyocr
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

client = Groq()
reader = easyocr.Reader(['en']) 
vector_db = None 

def initialize_brain():
    global vector_db
    
    all_documents = [] 
    

    pdf_path = "data/medical_knowledge.pdf"
    if os.path.exists(pdf_path):
        print(f"📖 System: Reading clinical guidelines from '{pdf_path}'...")
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            all_documents.extend(pdf_loader.load())
            print("✅ PDF loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
    else:
        print(f"⚠️ SYSTEM WARNING: '{pdf_path}' not found!")


    print("📝 System: Searching for Markdown (.md) knowledge files...")
    md_files_found = 0
    if os.path.exists("data"):
     for filename in os.listdir("data"):
        if filename.endswith(".md"):
            print(f"   -> Digesting: {filename}...")
            try:
                md_loader = TextLoader(os.path.join("data", filename), encoding="utf-8") 
                all_documents.extend(md_loader.load())
                md_files_found += 1
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {e}")
                
    if md_files_found == 0:
        print("⚠️ SYSTEM WARNING: No .md files found in the folder!")
    else:
        print(f"✅ Successfully loaded {md_files_found} Markdown file(s).")

   
    if not all_documents:
        print("🛑 FATAL ERROR: No knowledge files found. The AI has an empty brain!")
        return

    print("🧠 System: Digesting combined medical knowledge base...")
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vector_db = FAISS.from_documents(docs, embeddings)
        
        print(f"✅ SYSTEM READY: Learned {len(docs)} medical chunks from all sources.")
    except Exception as e:
        print(f"❌ ERROR CREATING VECTOR DB: {e}")

initialize_brain()

def smart_scanner(image_path, history):
    if not image_path: return "Please upload an image first."
    print("👀 Scanning image...")
    try:
        results = reader.readtext(image_path, detail=0)
        raw_ingredients = ", ".join(results)
    except:
        return "Error reading image."
        
    user_complaint = "General Skin Sensitivity"
    
    if history:
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_complaint = msg.get("content", "General Skin Sensitivity")
                break
            elif isinstance(msg, (list, tuple)) and len(msg) > 0:
                user_complaint = msg[0] 
                break
        
    prompt = f"""
    TASK: Identify Dermatological Triggers.
    PATIENT COMPLAINT: "{user_complaint}"
    PRODUCT INGREDIENTS FOUND: "{raw_ingredients}"
    
    INSTRUCTIONS:
    1. Analyze the ingredients based on the patient's complaint.
    2. If you find ingredients known to irritate that specific condition, list them using clean bullet points.
    3. Keep the explanations very brief (1 sentence max per ingredient).
    4. STRICT OUTPUT FORMAT:
       ⚠️ POTENTIAL TRIGGERS DETECTED:
       • [Ingredient Name]: [Brief reason]
       • [Ingredient Name]: [Brief reason]
    5. If no specific triggers are found, output exactly: "✅ No common triggers found for this condition."
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Analysis Failed: {e}"

def doctor_agent(message, history, image_path):
    visual_context = "No scan provided."
    if image_path:
        try: visual_context = ", ".join(reader.readtext(image_path, detail=0))
        except: pass
    
    rag_context = "General Knowledge."
    if vector_db:
        docs = vector_db.similarity_search(message, k=3)
        rag_context = "\n".join([d.page_content for d in docs])
    
    system_prompt = f"""
    You are 'Derma-Agent', an expert Agentic Dermatological Assistant.
    
    MEDICAL KNOWLEDGE RETRIEVED (Based on current query): 
    {rag_context}
    
    CURRENT VISUAL EVIDENCE (OCR Scan): {visual_context}
    
    INSTRUCTIONS (THINK STEP-BY-STEP):
    1. First, analyze the Patient Symptoms in their latest message and compare them strictly to the Medical Knowledge Retrieved.
    2. Second, if the user's description is vague (e.g., "I have a bump"), DO NOT give a remedy yet. Instead, ask 2-3 clarifying questions about the SYMPTOMS (color, texture, itchiness, duration).
    3. Third, evaluate the Visual Evidence to see if any ingredients are known triggers for the suspected condition.
    4. Fourth, formulate a professional, empathetic triage response.
    5. IF the answer is not in your Medical Knowledge, you MUST state: "I do not have enough clinical data to assess this." Do not guess.
    6. STRICT NEGATIVE CONSTRAINT: Answer ONLY the specific question asked by the patient. Do NOT volunteer remedies, foods to avoid, or lifestyle changes UNLESS the patient explicitly asks for them in their message. Keep it concise.
    """

    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("content"):
                conversation.append({"role": msg.get("role", "user"), "content": msg.get("content")})
 
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_past, ai_past = msg
            if user_past:
                conversation.append({"role": "user", "content": user_past})
            if ai_past:
                conversation.append({"role": "assistant", "content": ai_past})

    conversation.append({"role": "user", "content": message})
    
    try:
        completion = client.chat.completions.create(
            messages=conversation,
            model="llama-3.3-70b-versatile",
            temperature=0.1 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

custom_css = """
body { background-color: #f0f4f8; }
.header-text { text-align: center; margin-bottom: 10px; }
.header-text h1 { color: #0f766e; font-weight: 700; }
.trigger-box textarea { 
    color: #b91c1c !important; 
    font-weight: bold; 
    background-color: #fef2f2 !important; 
    border: 1px solid #f87171 !important;
}
"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=custom_css, title="Derma-Agent") as app:
    with gr.Column(elem_classes="header-text"):
        gr.Markdown("# 🩺 Derma Companion")
        gr.Markdown("### Symptom Triage & Hazard Detection")
        gr.Markdown("---")
    
    with gr.Row():
        # Left Panel: Chat Interface
        with gr.Column(scale=7, variant="panel"):
            gr.Markdown("### 💬 Consultation Room")
            img_input = gr.Image(visible=False, type="filepath") 
            chatbot = gr.ChatInterface(
                fn=doctor_agent,
                additional_inputs=[img_input],
                description="**Step 1:** Describe your skin condition below.\n **Step 2:** Upload a product label on the right to scan for triggers.",
                examples=[["I have Rosacea and sensitive skin. What should I look out for?", None]]
            )
            # Disclaimer
            gr.HTML(
                """
                <p style='text-align: center; font-size: 12px; color: #888888; margin-top: 10px;'>
                ⚠️ <b>Disclaimer:</b> Derma-Agent is an AI tool and can make mistakes. It is designed for informational triage and educational purposes only. Always verify ingredient safety and consult a licensed dermatologist or healthcare provider before making medical decisions.
                </p>
                """
            )
            
        #Right Panel: Smart Scanner
        with gr.Column(scale=3, variant="panel"):
            gr.Markdown("### 📷 Smart Trigger Detector")
            gr.Markdown("<span style='color: gray; font-size: 0.9em;'>Upload a product label. The AI will cross-reference the ingredients with your chat symptoms.</span>")
            
            scanner_ui = gr.Image(type="filepath", label="Upload Product Label", height=280)
            scan_btn = gr.Button("🔍 Scan for Triggers", variant="primary", size="lg")
            
            gr.Markdown("<br> 📊 Risk Analysis Result")
            scanner_output = gr.Textbox(
                label="", 
                show_label=False,
                interactive=False, 
                lines=5, 
                max_lines=5,
                elem_classes="trigger-box",
                placeholder="Scan results will appear here..."
            )
            
            scan_btn.click(
                fn=smart_scanner, 
                inputs=[scanner_ui, chatbot.chatbot], 
                outputs=[scanner_output]
            )
            scanner_ui.change(lambda x: x, inputs=scanner_ui, outputs=img_input)

# NOTE: share=False for local development is usually faster.
app.launch(share=False, debug=True)
#.\venv\Scripts\activate