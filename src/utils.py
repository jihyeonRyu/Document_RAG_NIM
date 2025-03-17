import hashlib

def generate_hash(input_string:str, postfix: str="") -> str:
    
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest() + "" if postfix == "" else f"_{postfix}"
    return hash_hex