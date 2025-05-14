import jwt
token = jwt.encode({"company_id": 3, "bot_id": 1}, "my-ultra-secure-signing-key", algorithm="HS256")
print(token)
