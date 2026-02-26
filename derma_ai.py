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

# .env
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

client = Groq()
reader = easyocr.Reader(['en']) 
vector_db = None 

def initialize_brain():
    global vector_db
    
    all_documents = [] 
    
    pdf_path = "medical_knowledge.pdf"
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
    
    for filename in os.listdir("."):
        if filename.endswith(".md"):
            print(f"   -> Digesting: {filename}...")
            try:
                md_loader = TextLoader(filename, encoding="utf-8") 
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

def doctor_agent(message, history):
    # Grab the text and files from the multimodal dictionary
    user_text = message.get("text", "")
    files = message.get("files", [])
    
    visual_context = ""
    combined_user_input = user_text # Start with just the text
    
    # 1. Handle the Image conditional logic securely
    if files:
        image_path = files[0] # Grab the first uploaded file
        print("👀 Scanning uploaded image in chat...")
        try:
            results = reader.readtext(image_path, detail=0)
            if results:
                visual_context = "RAW INGREDIENTS DETECTED FROM IMAGE: " + ", ".join(results)
            else:
                visual_context = "RAW INGREDIENTS DETECTED FROM IMAGE: [No readable text found]"
            print(f"🔍 Found Ingredients: {visual_context}")
            
            # ONLY attach the system note if an image was actually uploaded!
            combined_user_input += f"\n\n[SYSTEM NOTE: {visual_context}]"
        except Exception as e:
            combined_user_input += f"\n\n[SYSTEM NOTE: Error reading image: {e}]"
    
    # 2. Search the RAG database based on the user's text
    rag_context = "General Knowledge."
    if vector_db and user_text.strip():
        docs = vector_db.similarity_search(user_text, k=3)
        rag_context = "\n".join([d.page_content for d in docs])
    
    # 3. The Anti-Hallucination Prompt
    system_prompt = f"""
    You are 'Derma-Agent', an expert Agentic Dermatological Assistant.
    
    MEDICAL KNOWLEDGE RETRIEVED: 
    {rag_context}
    
    INSTRUCTIONS (STRICT COMPLIANCE REQUIRED):
    1. CONVERSATIONAL PACING: Act like a real doctor. Address the patient's current message with empathy. Ask ONLY ONE logical follow-up question at a time.
    2. TRIAGE FIRST: If the user describes a new symptom, ask ONE clarifying question (e.g., location, duration, or pain level) to narrow down the condition.
    3. PRODUCT INQUIRY: ONLY AFTER you understand the basic symptoms (usually by the second or third message), gently ask: "Have you started using any new products recently? If so, you can upload a picture of the ingredient label using the 📎 button."
    4. IMAGE AWARENESS & OCR: You can "see" images ONLY IF the user uploads one. If they do, the backend system will append a [SYSTEM NOTE] to the user's message containing the extracted OCR text.
    5. STRICT ANTI-HALLUCINATION RULE: DO NOT acknowledge, mention, or assume an image upload UNLESS you explicitly see a [SYSTEM NOTE] at the end of the user's message. If there is no [SYSTEM NOTE], treat it as a pure text conversation. DO NOT say "I see you uploaded an image" if there is no system note.
    6. HAZARD DETECTION: If you do receive a [SYSTEM NOTE], cross-reference the ingredients with your Medical Knowledge to find triggers and warn the user.
    """

    conversation = [{"role": "system", "content": system_prompt}]
    
    # Safely format the Gradio history for the Groq API (Gradio 6.0+ Compatible)
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if isinstance(content, str) and content.strip():
                conversation.append({"role": role, "content": content})
            elif isinstance(content, (list, tuple)):
                conversation.append({"role": role, "content": "[User uploaded an image previously]"})
                
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_past, ai_past = msg
            if isinstance(user_past, str) and user_past.strip():
                conversation.append({"role": "user", "content": user_past})
            if isinstance(ai_past, str) and ai_past.strip():
                conversation.append({"role": "assistant", "content": ai_past})

    # Pass the perfectly formatted input to the AI
    if combined_user_input.strip():
        conversation.append({"role": "user", "content": combined_user_input})
    else:
        # Fallback if the user sends an empty message
        conversation.append({"role": "user", "content": "[Empty message]"})
    
    try:
        completion = client.chat.completions.create(
            messages=conversation,
            model="llama-3.3-70b-versatile",
            temperature=0.1 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

    conversation = [{"role": "system", "content": system_prompt}]
    
    # Safely format the Gradio history for the Groq API (Gradio 6.0+ Compatible)
    for msg in history:
        # Gradio 6 stores history as a list of dictionaries
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # If the content is a string, add it to the memory safely
            if isinstance(content, str) and content.strip():
                conversation.append({"role": role, "content": content})
            # If the user uploaded an image in the past, Gradio stores it as a list/tuple
            elif isinstance(content, (list, tuple)):
                conversation.append({"role": role, "content": "[User uploaded an image]"})
                
        # Fallback just in case Gradio reverts to the old list format
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_past, ai_past = msg
            if isinstance(user_past, str) and user_past.strip():
                conversation.append({"role": "user", "content": user_past})
            if isinstance(ai_past, str) and ai_past.strip():
                conversation.append({"role": "assistant", "content": ai_past})

    # Add the current user message and inject the OCR data secretly so the AI sees it
    combined_user_input = user_text + f"\n\n[SYSTEM NOTE: {visual_context}]"
    conversation.append({"role": "user", "content": combined_user_input})
    
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
.gradio-container {background-color: #f8f9fa;}
"""

# --- UPDATED UI DESIGN ---
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Derma-Agent") as app:
    gr.Markdown("<h1 style='text-align: center;'>🩺 Derma-Agent AI</h1>")
    gr.Markdown("<h3 style='text-align: center;'>Context-Aware Symptom Triage & Hazard Detection</h3>")
    
    # Notice multimodal=True! This adds the paperclip icon to the chatbox.
    chatbot = gr.ChatInterface(
        fn=doctor_agent,
        multimodal=True,
        title="Consultation Room",
        description="Describe your symptoms below. If asked, you can attach an ingredient label using the 📎 icon.",
        examples=[
            {"text": "I have a red, itchy rash on my cheeks.", "files": []},
            {"text": "My acne is flaring up. I use this product.", "files": []}
        ]
    )

app.launch(share=False, debug=True)
#.\venv\Scripts\activate