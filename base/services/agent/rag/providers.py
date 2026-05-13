import uuid
import base64
import os
import logging
from .services import VectorStoreService, DocumentLoaderService, ResponseGenerationService
from .config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)
class RagProviders:
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            # Use lazy initialization - services are created only when accessed
            self._document_loader = None
            self._vector_store = None
            self._generative_service = None
    
    @property
    def document_loader(self):
        """Lazy initialization of DocumentLoaderService"""
        if self._document_loader is None:
            self._document_loader = DocumentLoaderService()
        return self._document_loader
    
    @property
    def vector_store(self):
        """Lazy initialization of VectorStoreService"""
        if self._vector_store is None:
            self._vector_store = VectorStoreService()
        return self._vector_store
    
    @property
    def generative_service(self):
        """Lazy initialization of ResponseGenerationService"""
        if self._generative_service is None:
            self._generative_service = ResponseGenerationService()
        return self._generative_service
    def upload_file(self, file_id:str, file_base64:str, file_type:str, file_name:str, file_type_id:str):
        """Upload and process document file
        
        Simple flow:
        - Text files (.txt, .md, .csv, etc.) → Direct LangChain chunking
        - Office files (.docx, .xlsx, .pptx) → Direct LangChain chunking
        - PDF files → Gemini extraction + chunking (for Khmer support)

        Args:
            file_id (str): Unique identifier stored in database
            file_base64 (str): Base64 encoded file content
            file_type (str): Document type (e.g., 'memo', 'policy') for collection management
            file_name (str): Filename including extension
            file_type_id (str): Unique identifier for the file type
        """
        
        try:
            file_bytes = base64.b64decode(file_base64)
            file_ext = os.path.splitext(file_name.lower())[1]
            
            # Handle files without extensions
            if not file_ext or file_ext == '.':
                return {'success': False, 'error': f'File missing or invalid extension: {file_name}'}
            
            # Determine file processing method
            file_category = None
            for category, extensions in SUPPORTED_EXTENSIONS.items():
                if file_ext in extensions:
                    file_category = category
                    break

            # Route to appropriate loader
            if file_ext == '.pdf':
                # Use Gemini for PDFs (better Khmer support)
                chunks = self.document_loader.load_pdf(file_bytes, file_id, file_name)
            elif file_category == 'text':
                # Direct processing for text files (.txt, .md)
                chunks = self.document_loader.load_text(file_bytes, file_id, file_name)
            elif file_category == 'word':
                # Direct processing for Word documents
                chunks = self.document_loader.load_word(file_bytes, file_id, file_name)
            elif file_category == 'spreadsheet':
                # Direct processing for Excel files
                chunks = self.document_loader.load_spreadsheet(file_bytes, file_id, file_name)
            elif file_category == 'presentation':
                # Direct processing for PowerPoint files
                chunks = self.document_loader.load_presentation(file_bytes, file_id, file_name)
            elif file_category == 'csv':
                chunks = self.document_loader.load_csv_tsv(file_bytes, file_id, file_name, delimiter=',')
            elif file_category == 'tsv':
                chunks = self.document_loader.load_csv_tsv(file_bytes, file_id, file_name, delimiter='\t')
            else:
                # Unsupported file type
                supported_exts = []
                for exts in SUPPORTED_EXTENSIONS.values():
                    supported_exts.extend(exts)
                supported_exts.append('.pdf')
                return {'success': False, 'error': f'Unsupported file extension: {file_ext}. Supported: {sorted(set(supported_exts))}'}
            
            if not chunks:
                print("Failed at document loading and chunking")
                return {'success': False, 'error': 'Document loading/chunking failed'}

            # Deduplication: if the same file_name was uploaded before, deactivate the old
            # version so retrieval only sees the latest copy.
            try:
                existing_ids = self.vector_store.find_file_ids_by_filename(file_name, file_type)
                for old_id in existing_ids:
                    if str(old_id) != str(file_id):
                        logger.info(f"Deactivating previous version of '{file_name}' (source={old_id})")
                        self.vector_store.deactivate_document(old_id, file_type)
            except Exception as dedup_err:
                # Non-fatal: log and continue — indexing the new version is more important
                logger.warning(f"Dedup check failed for '{file_name}': {dedup_err}")

            # adding to Vector DB using the document type (file_type) for collection management
            success = self.vector_store.add_chunk(chunks, file_type=file_type)

            if not success:
                return {'success': False, 'error':'Vector Store Failed'}
            
            return {
                'success': True,
            }        
        except Exception as e :
            logger.error(f"Error processing file {file_name}: {str(e)}")
            print(f"Error to process the file: {str(e)}")
            return {'success': False, 'error': str(e)}    
    def delete_file(self, file_id: str, file_type: str):
        """Delete file from vector store by file_id (checks both active and inactive collections)"""
        try:
            # First, try to get chunks from the original collection
            chunk_ids = self.vector_store.get_chunk_ids_by_file_id(file_id, file_type)
            collection_to_delete_from = file_type
            
            # If not found in original collection, check the inactive collection
            if not chunk_ids:
                print(f"No chunks found in {file_type} collection, checking inactive collection...")
                chunk_ids = self.vector_store.get_chunk_ids_by_file_id(file_id, "inactive")
                collection_to_delete_from = "inactive"
            
            if not chunk_ids:
                print(f"No chunks found for file_id: {file_id} in either {file_type} or inactive collection")
                # Not an error - document may have already been deleted from vector store
                history_success = self.generative_service.delete_history(file_id)
                return {
                    'success': True,
                    'message': "No chunks found in vector store, history deleted",
                    'history_deleted': history_success
                }
            
            success = self.vector_store.delete_chunk(chunk_ids, collection_to_delete_from)
            if not success:
                return {'success': False, 'error': 'Vector Store Deletion Failed'}
            
            history_success = self.generative_service.delete_history(file_id)
            return {
                'success': True,
                'message': f"File deleted from {collection_to_delete_from} collection and history deleted",
                'history_deleted': history_success,
                'collection': collection_to_delete_from
            }
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def deactivate_document(self, file_id: str, file_type: str):
        """Move document to inactive storage"""
        try:
            if not file_id or file_id is None:
                return {'success': False, 'error': 'Invalid file_id'}
            success = self.vector_store.deactivate_document(file_id, file_type)
            
            if success:
                history_success = self.generative_service.deactivate_history(file_id)
                return {
                    'success': True, 
                    'message': 'Document and related history deactivated successfully',
                    'history_deactivated': history_success
                }
            else:
                return {'success': False, 'error': 'Failed to deactivate document'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def activate_document(self, file_id: str):
        """Restore document from inactive collection"""
        try:
            if not file_id or file_id is None:
                return {'success': False, 'error': 'Invalid file_id'}
            success = self.vector_store.activate_document(file_id)
            
            if success:

                history_success = self.generative_service.activate_history(file_id)
                
                return {
                    'success': True, 
                    'message': 'Document and related history activated successfully',
                    'history_activated': history_success
                }
            else:
                return {'success': False, 'error': 'Failed to activate document'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}




