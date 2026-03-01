from enum import Enum

class ClassType(Enum):
    NOTIFICATION = 'notification'
    NOTIFICATION_TYPE = 'notification_type'
    NOTIFICATION_READ = 'notification_read'
    GROUP_NOTIFICATION = 'group_notification'
    UNKNOWN = 'unknown'

class NotificationLevel(Enum):
    USER = 'user'
    GROUP_NOTIFICATION = 'group_notification'

class NotificationType(Enum):
    INFO = 'info'
    ERROR = 'error'
    WARNING = 'warning'
    DB_TABLE_INSERT = 'table insert'
    DB_TABLE_CHANGE = 'table change'
