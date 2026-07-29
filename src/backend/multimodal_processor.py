"""
Multimodal document processing with EasyOCR
"""
import os
import tempfile
from typing import Dict, Any

import easyocr
from fastapi import UploadFile

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

async def process_file(file: UploadFile) -> Dict[str, Any]:
    """Process a file using EasyOCR for image text extraction."""
    try:
        # Read file content
        content = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process with EasyOCR
            result = reader.readtext(temp_file_path)
            extracted_text = "\n".join([text for _, text, _ in result])
            
            # Return the result
            return {
                "filename": file.filename,
                "processed_text": extracted_text,
                "error": None
            }
        except Exception as e:
            return {
                "filename": file.filename,
                "processed_text": "",
                "error": f"Error processing image: {str(e)}"
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    except Exception as e:
        return {
            "filename": file.filename,
            "processed_text": "",
            "error": f"Error processing file: {str(e)}"
        }
    """Process a file using the document processing graph."""
    try:
        # Read file content
        content = await file.read()
        
        # Create a document
        doc = Document(page_content=content, metadata={"filename": file.filename})
        
        # Initialize state
        initial_state = GraphState(documents=[doc])
        
        # Run the graph
        config = RunnableConfig(recursion_limit=25)
        result = await document_processor.ainvoke(initial_state, config)
        
        # Return the result
        return {
            "filename": file.filename,
            "processed_text": result.processed_text,
            "error": result.error
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "processed_text": "",
            "error": f"Error processing file: {str(e)}"
        }