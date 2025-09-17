# Fix for ChromaDB SQLite compatibility on Streamlit Cloud only
try:
    import streamlit.web.cli as stcli
    # Only apply fix when running on Streamlit Cloud
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, ModuleNotFoundError):
    # Running locally or pysqlite3 not available, use regular sqlite3
    pass

import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import speech_recognition as sr

# Load environment variables
load_dotenv()

st.title("My Content Creator AI Assistant")
st.write("Welcome to your AI-powered content creation tool!")

# Get API keys from environment variables
openai_key = os.getenv("OPENAI_API_KEY")
claude_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_key or not claude_key:
    st.error("Please set both OPENAI_API_KEY and ANTHROPIC_API_KEY in your .env file!")
else:
    # Model selection dropdown
    st.subheader("🤖 Choose Your AI Model")
    model_choice = st.selectbox(
        "Select AI model:",
        [ 
            "Claude Sonnet 4",
            "GPT-5"
        ]
    )
    
    # Map display names to actual model names
    model_mapping = {
    "Claude Sonnet 4": {"provider": "claude", "model": "claude-sonnet-4-20250514"},
    "GPT-5": {"provider": "openai", "model": "gpt-5"}
    }
    
    selected_model = model_mapping[model_choice]

    # Initialize ChromaDB
    @st.cache_resource
    def init_vector_databases():
        client = chromadb.PersistentClient(path="./content_memory")
        ef = embedding_functions.DefaultEmbeddingFunction()
        
        # My content style collection
        my_style_collection = client.get_or_create_collection(
            name="my_content_style",
            embedding_function=ef
        )
        
        # Favorite creators collection
        creators_collection = client.get_or_create_collection(
            name="favorite_creators",
            embedding_function=ef
        )
        
        return my_style_collection, creators_collection

    my_style_collection, creators_collection = init_vector_databases()
    
    # Show model info
    st.info(f"Selected: {model_choice}")
    
    # Function to make AI call based on selected model
    def make_ai_call(prompt):
        if selected_model["provider"] == "openai":
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=selected_model["model"],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:  # Claude
            client = Anthropic(api_key=claude_key)
            response = client.messages.create(
                model=selected_model["model"],
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
# Enhanced Script Editor Section
st.subheader("Script Editor & Improvement")
st.write("Transform your rough draft into a polished script using inspiration from your style and favorite creators.")

# Script input
user_script = st.text_area(
    "Your rough draft:",
    placeholder="Paste your script draft, outline, or rough ideas here...",
    height=250
)

if user_script:
    # Inspiration source selection
    st.markdown("### Choose Your Inspiration Sources")
    
    inspiration_source = st.radio(
        "What should inspire the improvement?",
        ["My personal style only", "Favorite creators only", "Both my style and favorite creators"]
    )
    
    # If choosing favorite creators, allow specific selection
    creator_selection = None
    if inspiration_source in ["Favorite creators only", "Both my style and favorite creators"]:
        creator_stored = creators_collection.get()
        if creator_stored['documents']:
            # Get unique creator names
            creator_names = list(set([meta['creator_name'] for meta in creator_stored['metadatas']]))
            
            creator_selection = st.multiselect(
                "Select specific creators (leave empty for all):",
                options=creator_names,
                help="Choose specific creators to draw inspiration from, or leave empty to use all stored creator content"
            )
            
            # Show selected creator content count
            if creator_selection:
                selected_count = sum(1 for meta in creator_stored['metadatas'] 
                                   if meta['creator_name'] in creator_selection)
                st.info(f"Using {selected_count} pieces of content from {len(creator_selection)} creators")
        else:
            st.warning("No creator content stored yet. Add some in the Favorite Creators tab.")
    
    # Improvement focus
    improvement_focus = st.selectbox(
        "Focus area for improvement:",
        [
            "Overall storytelling and flow", 
            "Hook and opening strength", 
            "Call-to-action effectiveness", 
            "Clarity and structure",
            "Emotional engagement",
            "Match my personal voice"
        ]
    )
    
if st.button("Improve My Script"):
    with st.spinner(f"Improving your script using {inspiration_source.lower()}..."):
        try:
            # Keep all your existing context building code here
            context_parts = []
            
            if inspiration_source in ["My personal style only", "Both my style and favorite creators"]:
                my_examples = my_style_collection.query(
                    query_texts=[user_script[:500]],
                    n_results=2
                )
                if my_examples['documents'][0]:
                    context_parts.append("Your personal writing style examples:")
                    for i, (doc, meta) in enumerate(zip(my_examples['documents'][0], my_examples['metadatas'][0]), 1):
                        context_parts.append(f"Your Style Example {i} - '{meta['title']}':\n{doc[:400]}...")
            
            if inspiration_source in ["Favorite creators only", "Both my style and favorite creators"]:
                where_filter = None
                if creator_selection:
                    where_filter = {"creator_name": {"$in": creator_selection}}
                
                creator_examples = creators_collection.query(
                    query_texts=[user_script[:500]],
                    n_results=3,
                    where=where_filter
                )
                if creator_examples['documents'][0]:
                    context_parts.append("\nSuccessful creator examples for inspiration:")
                    for i, (doc, meta) in enumerate(zip(creator_examples['documents'][0], creator_examples['metadatas'][0]), 1):
                        context_parts.append(f"Creator Example {i} - {meta['creator_name']}: '{meta['content_title']}':\n{doc[:400]}...")
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant examples found."
            
            enhancement_prompt = f"""You are an expert script editor specializing in content creation and storytelling.

                ORIGINAL SCRIPT TO IMPROVE:
                "{user_script}"

                FOCUS AREA: {improvement_focus}

                INSPIRATION SOURCES:
                {context}

                Your task:
                1. Rewrite and improve the original script, focusing specifically on {improvement_focus.lower()}
                2. Draw inspiration from the provided examples while maintaining the user's authentic voice
                3. Enhance storytelling elements, structure, and engagement
                4. Keep the core message but elevate the execution

                Provide the improved script.

                Make it compelling, authentic, and ready to use."""
            
            improved_script = make_ai_call(enhancement_prompt)
            
            # Store in session state
            st.session_state.enhanced_script = improved_script
            
            st.success("Script improved successfully!")
            
        except Exception as e:
            st.error(f"Error improving script: {str(e)}")

# Display the enhanced script if it exists
if 'enhanced_script' in st.session_state:
    st.markdown("### Your Enhanced Script")
    st.write(st.session_state.enhanced_script)
    
    st.download_button(
        label="Download Enhanced Script",
        data=st.session_state.enhanced_script,
        file_name="enhanced_script.txt",
        mime="text/plain"
    )

# Content Memory Banks
st.markdown("---")
st.subheader("Content Memory Banks")

# Two tabs: My Style, Favorite Creators
memory_tab1, memory_tab2 = st.tabs(["My Style Library", "Favorite Creators"])

# Tab 1: My Style
with memory_tab1:
    style_subtab1, style_subtab2 = st.tabs(["Add Style Example", "View My Content"])
    
    with style_subtab1:
        st.write("Add examples of your successful content:")
        
        example_title = st.text_input("Content title/topic:")
        example_script = st.text_area(
            "Your script or content:",
            height=200,
            placeholder="Paste your script, video transcript, or content..."
        )
        example_notes = st.text_input(
            "Style notes (optional):",
            placeholder="e.g., casual tone, uses humor, includes personal stories"
        )
        
        if st.button("Save My Style Example"):
            if example_title and example_script:
                my_style_collection.add(
                    documents=[example_script],
                    metadatas=[{
                        "title": example_title,
                        "notes": example_notes,
                        "timestamp": str(datetime.now())
                    }],
                    ids=[f"my_style_{len(my_style_collection.get()['ids']) + 1}"]
                )
                st.success(f"Saved: {example_title}")
    
    with style_subtab2:
        my_stored_examples = my_style_collection.get()
        if my_stored_examples['documents']:
            st.write(f"You have {len(my_stored_examples['documents'])} style examples:")
            
            for i, (doc, metadata, doc_id) in enumerate(zip(
                my_stored_examples['documents'], 
                my_stored_examples['metadatas'], 
                my_stored_examples['ids']
            )):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    with st.expander(f"My Content: {metadata['title']}"):
                        st.write(f"**Notes:** {metadata.get('notes', 'None')}")
                        st.write(f"**Content:** {doc[:300]}...")
                
                with col2:
                    if st.button("Delete", key=f"delete_my_{doc_id}"):
                        my_style_collection.delete(ids=[doc_id])
                        st.rerun()
        else:
            st.info("No style examples stored yet.")

# Tab 2: Favorite Creators
with memory_tab2:
    creator_subtab1, creator_subtab2 = st.tabs(["Add Creator Content", "Browse Creator Library"])
    
    with creator_subtab1:
        st.write("Add content from creators you admire:")
        
        creator_name = st.text_input("Creator name:")
        content_title = st.text_input("Video/content title:")
        creator_content = st.text_area(
            "Creator's script/transcript:",
            height=200,
            placeholder="Paste the script or transcript from content you want to learn from..."
        )
        content_notes = st.text_input(
            "What you like about this content:",
            placeholder="e.g., great storytelling, strong hook, clear structure"
        )
        
        if st.button("Save Creator Content"):
            if creator_name and content_title and creator_content:
                creators_collection.add(
                    documents=[creator_content],
                    metadatas=[{
                        "creator_name": creator_name,
                        "content_title": content_title,
                        "notes": content_notes,
                        "timestamp": str(datetime.now())
                    }],
                    ids=[f"creator_{len(creators_collection.get()['ids']) + 1}"]
                )
                st.success(f"Saved content from {creator_name}: {content_title}")
    
    with creator_subtab2:
        creator_stored = creators_collection.get()
        if creator_stored['documents']:
            # Group by creator
            creators_dict = {}
            for doc, metadata, doc_id in zip(
                creator_stored['documents'], 
                creator_stored['metadatas'], 
                creator_stored['ids']
            ):
                creator = metadata['creator_name']
                if creator not in creators_dict:
                    creators_dict[creator] = []
                creators_dict[creator].append((doc, metadata, doc_id))
            
            st.write(f"Content from {len(creators_dict)} creators:")
            
            for creator_name, content_list in creators_dict.items():
                with st.expander(f"Content from {creator_name} ({len(content_list)} videos)"):
                    for doc, metadata, doc_id in content_list:
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.write(f"**{metadata['content_title']}**")
                            st.write(f"Notes: {metadata.get('notes', 'None')}")
                            st.write(f"Content: {doc[:200]}...")
                        
                        with col2:
                            if st.button("Delete", key=f"delete_creator_{doc_id}"):
                                creators_collection.delete(ids=[doc_id])
                                st.rerun()
        else:
            st.info("No creator content stored yet.")