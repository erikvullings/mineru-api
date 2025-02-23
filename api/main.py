from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, Response, PlainTextResponse
import shutil
import os
import os.path
import tempfile
from pathlib import Path
import zipfile
from typing import Optional, List
import base64
import re
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import traceback
import torch
import paddle
import multiprocessing
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images, read_local_office
from magic_pdf.config.enums import SupportedPdfParseMethod

multiprocessing.set_start_method('spawn', force=True)

# Set PyMU environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Converter API",
    description="Convert various document formats to Markdown",
    version="1.0.0"
)

class DocumentType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    OFFICE = "office"

def check_cuda_setup():
    """Check CUDA and CUDNN setup"""
    try:
        # Configure PaddlePaddle to use CUDA
        paddle.device.set_device('gpu')
        
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name()}")
        
        # Check PaddlePaddle GPU support
        logger.info(f"PaddlePaddle GPU available: {paddle.device.is_compiled_with_cuda()}")
        logger.info(f"PaddlePaddle version: {paddle.__version__}")
        logger.info(f"PaddlePaddle current device: {paddle.device.get_device()}")
            
    except Exception as e:
        logger.error(f"Error checking CUDA setup: {str(e)}")
        logger.error(traceback.format_exc())

# Add this to your FastAPI startup events
@app.on_event("startup")
async def startup_event():
    check_cuda_setup()
    
def get_document_type(filename: str) -> DocumentType:
    """Determine document type from file extension."""
    ext = filename.lower().split('.')[-1]
    if ext in ['pdf']:
        return DocumentType.PDF
    elif ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        return DocumentType.IMAGE
    elif ext in ['doc', 'docx', 'ppt', 'pptx']:
        return DocumentType.OFFICE
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def process_document(
    file_path: str,
    output_dir: str,
    doc_type: DocumentType,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None
) -> dict:
    """Process a document in a separate process."""
    try:
        logger.info(f"Starting conversion of {file_path} as {doc_type}")
        
        # Initialize CUDA after spawn
        if torch.cuda.is_available():
            torch.cuda.init()
            logger.info(f"CUDA initialized in spawned process")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")

        # Setup output directories
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        image_writer = FileBasedDataWriter(image_dir)
        md_writer = FileBasedDataWriter(output_dir)
        
        name_without_suffix = os.path.splitext(os.path.basename(file_path))[0]
        relative_image_dir = "images"
        
        if doc_type == DocumentType.PDF:
            logger.debug("Processing PDF document")
            reader = FileBasedDataReader("")
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            
            try:
                ds = PymuDocDataset(pdf_bytes)
                if start_page is not None:
                    ds.start_page = start_page
                if end_page is not None:
                    ds.end_page = end_page
                    
                parse_method = ds.classify()
                logger.debug(f"PDF parse method: {parse_method}")
                
                if parse_method == SupportedPdfParseMethod.OCR:
                    ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                        md_writer, f"{name_without_suffix}.md", relative_image_dir
                    )
                else:
                    ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
                        md_writer, f"{name_without_suffix}.md", relative_image_dir
                    )
            finally:
                # Cleanup PymuDocDataset resources
                if 'ds' in locals():
                    del ds
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()
                
        elif doc_type == DocumentType.IMAGE:
            logger.debug("Processing image document")
            ds = read_local_images(file_path)[0]
            ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                md_writer, f"{name_without_suffix}.md", relative_image_dir
            )
            
        elif doc_type == DocumentType.OFFICE:
            logger.debug("Processing office document")
            ds = read_local_office(file_path)[0]
            parse_method = ds.classify()
            
            if parse_method == SupportedPdfParseMethod.OCR:
                ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                    md_writer, f"{name_without_suffix}.md", relative_image_dir
                )
            else:
                ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
                    md_writer, f"{name_without_suffix}.md", relative_image_dir
                )
        
        logger.info(f"Successfully converted {file_path}")
        return {"status": "success", "message": "Document processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Processing failed: {str(e)}")

def embed_base64_images(markdown_content: str, image_dir: Path) -> str:
    """Replace image references with base64 encoded images in markdown content."""
    logger.debug("Starting image embedding")
    
    def replace_image(match):
        image_path = image_dir / match.group(1)
        if image_path.exists():
            with open(image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                return f'![](data:image/png;base64,{img_data})'
        return match.group(0)
    
    markdown_content = re.sub(r'!\[.*?\]\((.*?)\)', replace_image, markdown_content)
    markdown_content = re.sub(r'<img.*?src="(.*?)".*?>', replace_image, markdown_content)
    
    logger.debug("Finished image embedding")
    return markdown_content

@app.post("/convert/zip")
async def convert_document_zip(
    file: UploadFile = File(...),
    start: Optional[int] = Query(default=None, gt=0),
    end: Optional[int] = Query(default=None, gt=0)
):
    """Convert a document and return a ZIP containing markdown and images."""
    logger.info(f"Starting ZIP conversion for {file.filename}")
    
    if start and end and end < start:
        raise HTTPException(status_code=422, detail="end must be greater than or equal to start")

    try:
        doc_type = get_document_type(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate output ZIP filename from input filename
    output_filename = os.path.splitext(file.filename)[0] + '.zip'
    logger.debug(f"Output ZIP filename will be: {output_filename}")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            logger.debug(f"Saving uploaded file to {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory at {output_dir}")
            
            # Process document
            with ProcessPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor,
                    process_document,
                    file_path,
                    output_dir,
                    doc_type,
                    start,
                    end
                )
            
            # Create ZIP file
            zip_path = os.path.join(temp_dir, "result.zip")
            logger.debug(f"Creating ZIP file at {zip_path}")
            
            # Ensure the output directory has content before creating ZIP
            if not os.path.exists(output_dir) or not os.listdir(output_dir):
                raise Exception("No output files were generated during conversion")
                
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        logger.debug(f"Adding {file_path} to ZIP as {arcname}")
                        zipf.write(file_path, arcname)
            
            # Verify ZIP file exists and has content
            if not os.path.exists(zip_path):
                raise Exception("ZIP file was not created successfully")
                
            if os.path.getsize(zip_path) == 0:
                raise Exception("Created ZIP file is empty")
                
            logger.info(f"ZIP conversion completed successfully. File size: {os.path.getsize(zip_path)} bytes")
            
            # Read the ZIP file into memory before returning
            with open(zip_path, "rb") as zip_file:
                zip_contents = zip_file.read()
            
            # Return ZIP file from memory
            return Response(
                content=zip_contents,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename={output_filename}",
                    "Content-Length": str(len(zip_contents))
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ZIP conversion: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
        
        

@app.post("/convert/markdown")
async def convert_document_markdown(
    file: UploadFile = File(...),
    embed_images: bool = Query(default=False),
    start: Optional[int] = Query(default=None, gt=0),
    end: Optional[int] = Query(default=None, gt=0)
):
    """Convert a document and return the markdown content directly."""
    logger.info(f"Starting markdown conversion for {file.filename}")
    
    if start and end and end < start:
        raise HTTPException(status_code=422, detail="end must be greater than or equal to start")

    try:
        doc_type = get_document_type(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a new ProcessPoolExecutor for each request
            with ProcessPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor,
                    process_document,
                    file_path,
                    output_dir,
                    doc_type,
                    start,
                    end
                )
            
            # Find and read the markdown file
            markdown_files = list(Path(output_dir).glob("*.md"))
            if not markdown_files:
                raise HTTPException(status_code=500, detail="No markdown file was generated")
            
            markdown_content = markdown_files[0].read_text()
            
            # Embed images if requested
            if embed_images:
                markdown_content = embed_base64_images(markdown_content, Path(output_dir))
            
            logger.info("Markdown conversion completed successfully")
            return PlainTextResponse(markdown_content)
            
        except Exception as e:
            logger.error(f"Error in markdown conversion: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {"status": "healthy"}