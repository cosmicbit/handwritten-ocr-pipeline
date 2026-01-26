from ..commons.attribute import AttributeObject
from .base_dto import BaseDTO

class UserRegistration(BaseDTO):
    def __init__(self):
        self.username = AttributeObject(required=True)
        self.password = AttributeObject(required=True)
        self.email = AttributeObject(required=True)
        self.firstName = AttributeObject()
        self.lastName = AttributeObject()
        self.dateOfBirth = AttributeObject()
        self.phoneNumber = AttributeObject()

    
