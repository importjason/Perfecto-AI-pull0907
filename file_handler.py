# 파일 처리 모듈
# LlamaParse를 사용하여 업로드된 파일을 분석하고, LangChain Document 객체로 변환하는 역할을 전담합니다.

import os
import tempfile
import asyncio
import streamlit as st
from llama_parse import LlamaParse
from langchain_core.documents import Document as LangChainDocument

async def parse_files_with_llamaparse(files):
    """LlamaParse를 사용하여 파일들을 비동기적으로 파싱합니다."""
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True,
    )
    parsed_data = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            documents = await parser.aload_data(tmp_file_path)
            parsed_data.extend(documents)
        except Exception as e:
            st.error(f"LlamaParse 처리 중 오류 발생: {e}")
        finally:
            os.remove(tmp_file_path)
    return parsed_data

def get_documents_from_files(uploaded_files):
    """
    업로드된 파일들을 LlamaParse로 처리하고 LangChain Document 객체로 변환합니다.
    """
    # LlamaParse는 LlamaIndex 형식의 문서를 반환합니다.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    with st.spinner("LlamaParse로 문서를 분석하고 있습니다..."):
        llama_index_documents = loop.run_until_complete(parse_files_with_llamaparse(uploaded_files))
    
    if not llama_index_documents:
        return []

    # LlamaIndex 문서를 LangChain 문서로 변환합니다.
    langchain_documents = [
        LangChainDocument(page_content=doc.text, metadata=doc.metadata)
        for doc in llama_index_documents
    ]
    return langchain_documents