import jwt
token = jwt.encode({"company_id": 5, "bot_id": 3}, "my-ultra-secure-signing-key", algorithm="HS256")
print(token)
