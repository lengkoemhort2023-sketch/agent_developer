"""
TONL Helper Utility for Token Optimization
Provides encoding/decoding functions to reduce token usage in LLM prompts
Install: pip install tonl
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import our Python TONL implementation
try:
    from .tonl_python import encodeTONL, decodeTONL
    TONL_AVAILABLE = True
    logger.info("TONL Python implementation loaded successfully")
except ImportError as e:
    TONL_AVAILABLE = False
    logger.warning(f"Failed to load TONL Python implementation: {e}. Falling back to JSON.")


class TONLHelper:
    """Helper class for TONL encoding/decoding with JSON fallback"""
    
    @staticmethod
    def encode(data: Any, use_tonl: bool = True) -> str:
        """
        Encode data to TONL format (or JSON as fallback)
        
        Args:
            data: Python data structure to encode
            use_tonl: Whether to use TONL encoding (default: True)
            
        Returns:
            Encoded string (TONL or JSON)
        """
        try:
            if use_tonl and TONL_AVAILABLE:
                # Use TONL smart encoding for optimal compression
                return encodeTONL(data, smart=True)
            else:
                # Fallback to JSON
                return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error encoding data with TONL: {e}, falling back to JSON")
            return json.dumps(data, ensure_ascii=False)
    
    @staticmethod
    def decode(encoded_str: str, use_tonl: bool = True) -> Any:
        """
        Decode TONL or JSON string back to Python data structure
        
        Args:
            encoded_str: Encoded string (TONL or JSON)
            use_tonl: Whether to attempt TONL decoding first (default: True)
            
        Returns:
            Decoded Python data structure
        """
        try:
            # Try TONL first if enabled and available
            if use_tonl and TONL_AVAILABLE:
                try:
                    return decodeTONL(encoded_str)
                except Exception as tonl_error:
                    # If TONL fails, try JSON
                    logger.debug(f"TONL decode failed, trying JSON: {tonl_error}")
                    return json.loads(encoded_str)
            else:
                # Use JSON directly
                return json.loads(encoded_str)
        except Exception as e:
            logger.error(f"Error decoding string: {e}")
            raise
    
    @staticmethod
    def encode_for_llm_prompt(data: Dict, format_type: str = "auto") -> str:
        """
        Encode data specifically for LLM prompts with optimal token efficiency
        
        Args:
            data: Data to include in LLM prompt
            format_type: "auto", "tonl", or "json"
            
        Returns:
            Optimally encoded string for LLM consumption
        """
        if format_type == "json" or not TONL_AVAILABLE:
            return json.dumps(data, ensure_ascii=False, indent=None)
        
        if format_type == "tonl" or (format_type == "auto" and TONL_AVAILABLE):
            try:
                return encodeTONL(data, smart=True)
            except Exception as e:
                logger.error(f"Error encoding for LLM with TONL: {e}")
                return json.dumps(data, ensure_ascii=False, indent=None)
        
        return json.dumps(data, ensure_ascii=False, indent=None)
    
    @staticmethod
    def encode_chat_answers(answers: List[Dict]) -> str:
        """
        Encode chat answers with optimal compression
        
        Args:
            answers: List of answer dictionaries
            
        Returns:
            Encoded string
        """
        return TONLHelper.encode({"answers": answers}, use_tonl=True)
    
    @staticmethod
    def decode_chat_answers(encoded_str: str) -> List[Dict]:
        """
        Decode chat answers from encoded string
        
        Args:
            encoded_str: Encoded answers string
            
        Returns:
            List of answer dictionaries
        """
        try:
            decoded = TONLHelper.decode(encoded_str, use_tonl=True)
            if isinstance(decoded, dict) and "answers" in decoded:
                return decoded["answers"]
            return []
        except Exception as e:
            logger.error(f"Error decoding chat answers: {e}")
            return []
    
    @staticmethod
    def get_token_savings_estimate(data: Any) -> Dict[str, Any]:
        """
        Estimate token savings by comparing JSON vs TONL encoding
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with size comparison and estimated savings
        """
        if not TONL_AVAILABLE:
            return {
                "json_size": len(json.dumps(data)),
                "tonl_size": None,
                "savings_percent": None,
                "tonl_available": False
            }
        
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            tonl_str = encodeTONL(data, smart=True)
            
            json_size = len(json_str)
            tonl_size = len(tonl_str)
            savings = ((json_size - tonl_size) / json_size) * 100
            
            # Rough token estimation (average ~4 chars per token for English)
            json_tokens = json_size // 4
            tonl_tokens = tonl_size // 4
            token_savings = ((json_tokens - tonl_tokens) / json_tokens) * 100
            
            return {
                "json_size": json_size,
                "tonl_size": tonl_size,
                "savings_bytes": json_size - tonl_size,
                "savings_percent": round(savings, 2),
                "json_tokens_estimate": json_tokens,
                "tonl_tokens_estimate": tonl_tokens,
                "token_savings_percent": round(token_savings, 2),
                "tonl_available": True
            }
        except Exception as e:
            logger.error(f"Error calculating savings: {e}")
            return {
                "error": str(e),
                "tonl_available": True
            }
    
    @staticmethod
    def is_available() -> bool:
        """Check if TONL is available"""
        return TONL_AVAILABLE


# Convenience functions for backward compatibility
def tonl_encode(data: Any) -> str:
    """Encode data to TONL (with JSON fallback)"""
    return TONLHelper.encode(data, use_tonl=True)


def tonl_decode(encoded_str: str) -> Any:
    """Decode TONL or JSON string"""
    return TONLHelper.decode(encoded_str, use_tonl=True)


def get_token_savings(data: Any) -> Dict:
    """Get token savings estimate"""
    return TONLHelper.get_token_savings_estimate(data)







