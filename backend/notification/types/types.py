from enum import Enum

class NotificationReadStatus(Enum):
    CREATED = 'User is marked as read'
    ALREADY_EXISTS = 'Notification is already marked as read for the user'
    FAILED = 'failed to update'