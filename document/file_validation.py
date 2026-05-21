"""
File Upload Configuration and Validation
Defines allowed file formats and validation logic for document uploads
"""

# Allowed file extensions for upload (matches multi_format_loader.SUPPORTED_FORMATS)
ALLOWED_FILE_EXTENSIONS = {
    '.docx',
}

# Maximum file size (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# File categories for better organization (matches ALLOWED_FILE_EXTENSIONS)
FILE_CATEGORIES = {
    'document': ['.docx'],
}

# Friendly names for error messages
CATEGORY_NAMES = {
    'document': 'Document',
}


def is_allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename (str): Name of the file with extension
        
    Returns:
        bool: True if file is allowed, False otherwise
    """
    if not filename or '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return f'.{extension}' in ALLOWED_FILE_EXTENSIONS


def get_file_category(filename: str) -> str:
    """
    Get the category of a file based on extension
    
    Args:
        filename (str): Name of the file with extension
        
    Returns:
        str: Category name or 'unknown'
    """
    if not filename or '.' not in filename:
        return 'unknown'
    
    extension = f'.{filename.rsplit(".", 1)[1].lower()}'
    
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    
    return 'unknown'


def validate_file_size(file_size: int) -> tuple[bool, str]:
    """
    Validate file size
    
    Args:
        file_size (int): Size of file in bytes
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if file_size > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        current_mb = file_size / (1024 * 1024)
        return False, f"File size ({current_mb:.1f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)"
    
    return True, ""


def validate_uploaded_file(file) -> tuple[bool, str]:
    """
    Comprehensive validation for uploaded files
    
    Args:
        file: Django UploadedFile object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if file exists
    if not file:
        return False, "No file provided"
    
    # Check filename
    if not file.name:
        return False, "File has no name"
    
    # Validate extension
    if not is_allowed_file(file.name):
        return False, "File type not allowed. Allowed type: DOCX"
    
    # Validate size
    is_valid_size, size_error = validate_file_size(file.size)
    if not is_valid_size:
        return False, size_error
    
    return True, ""


def get_supported_formats_list() -> dict:
    """
    Get a formatted list of supported formats by category
    
    Returns:
        dict: Categories with their supported formats
    """
    return {
        category: {
            'name': CATEGORY_NAMES.get(category, category.title()),
            'extensions': extensions
        }
        for category, extensions in FILE_CATEGORIES.items()
    }







