from pydantic import BaseModel, EmailStr
from typing import Optional


class Student(BaseModel):
  
  name: str = "rashid"
  age: Optional[int] = None
  email: EmailStr
  
  
new_student = {'name':"Ansari",'age':18,'email':'abc@gmail.com'}

student =  Student(**new_student)

print(new_student)