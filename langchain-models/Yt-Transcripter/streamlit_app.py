"""
Streamlit frontend for the YouTube Transcript Q&A tool.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from app import extract_video_id, fetch_transcript, build_vector_store, build_rag_chain

st.set_page_config(page_title="YouTube Transcript Q&A", page_icon="🎬")
st.title("🎬 YouTube Transcript Q&A")
st.caption("Paste a YouTube URL, then ask questions answered only from that video's transcript.")

if "video_id" not in st.session_state:
    st.session_state.video_id = None
    st.session_state.chain = None

url = st.text_input("YouTube URL or video ID", placeholder="https://youtu.be/MIlDK1qQLaI")

if st.button("Load video", type="primary", disabled=not url):
    try:
        video_id = extract_video_id(url)
        with st.spinner("Fetching transcript and building index..."):
            transcript = fetch_transcript(video_id)
            vector_store = build_vector_store(transcript)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            st.session_state.chain = build_rag_chain(retriever)
            st.session_state.video_id = video_id
        st.success("Video loaded. Ask a question below.")
    except (ValueError, RuntimeError) as e:
        st.error(str(e))

if st.session_state.video_id:
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    question = st.text_input("Your question", placeholder="Can you summarize the video?")
    if st.button("Ask", disabled=not question):
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
        st.markdown("### Answer")
        st.write(answer)
