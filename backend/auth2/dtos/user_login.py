from ..commons.attribute import AttributeObject
from .base_dto import BaseDTO

class UserLogin(BaseDTO):
    def __init__(self):
        self.username = AttributeObject(required=True)
        self.password = AttributeObject(required=True)
    
    