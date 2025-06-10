#!/usr/bin/env python3
"""
Generate JWT tokens for testing the knowledge bot widget
"""

import os
import sys
from dotenv import load_dotenv
from jwt_handler import JWTHandler

load_dotenv()

def generate_token(company_id: int, bot_id: int, hours: int = None):
    """Generate a minimal JWT token for the specified company and bot"""
    
    jwt_handler = JWTHandler()
    token = jwt_handler.create_token(company_id, bot_id)
    
    print(f"🔑 Minimal JWT Token Generated")
    print("=" * 50)
    print(f"Company ID: {company_id}")
    print(f"Bot ID: {bot_id}")
    print(f"Expiration: Never (minimal token)")
    print()
    print("Token:")
    print(token)
    print()
    print("🌐 Widget Usage:")
    print(f'<script src="https://knowledge-bot-retrieval.onrender.com/static/widget.js" data-token="{token}"></script>')
    print()

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_widget_token.py <company_id> <bot_id>")
        print("Example: python generate_widget_token.py 3 1")
        return
    
    try:
        company_id = int(sys.argv[1])
        bot_id = int(sys.argv[2])
        
        generate_token(company_id, bot_id)
        
    except ValueError:
        print("❌ Error: company_id and bot_id must be integers")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 