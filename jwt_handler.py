import jwt
import os
import logging
from typing import Dict, Optional
from fastapi import HTTPException, Header

logger = logging.getLogger(__name__)

class JWTHandler:
    """Handle JWT token validation and extraction for widget authentication"""
    
    def __init__(self):
        # Use JWT_SECRET for consistency with the rest of the system
        self.secret_key = os.getenv("JWT_SECRET", "my-ultra-secure-signing-key")
        self.algorithm = "HS256"
    
    def decode_token(self, token: str) -> Dict:
        """
        Decode and validate JWT token
        Returns dictionary with company_id, bot_id, and other claims
        """
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Decode without verifying expiration since our tokens don't expire
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            
            # Validate required fields
            if 'company_id' not in payload:
                raise HTTPException(status_code=400, detail="Missing company_id in token")
            if 'bot_id' not in payload:
                raise HTTPException(status_code=400, detail="Missing bot_id in token")
            
            logger.info(f"[JWT] Successfully decoded minimal token for company_id={payload['company_id']}, bot_id={payload['bot_id']}")
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.error(f"[JWT] Invalid token: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def create_token(self, company_id: int, bot_id: int, expires_in_hours: int = None) -> str:
        """
        Create JWT token - minimal payload with just company_id and bot_id
        """
        payload = {
            'company_id': company_id,
            'bot_id': bot_id
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"[JWT] Created minimal token for company_id={company_id}, bot_id={bot_id}")
        return token

def extract_jwt_claims(authorization: Optional[str] = Header(None)) -> Dict:
    """
    FastAPI dependency to extract JWT claims from Authorization header
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    jwt_handler = JWTHandler()
    return jwt_handler.decode_token(authorization) 