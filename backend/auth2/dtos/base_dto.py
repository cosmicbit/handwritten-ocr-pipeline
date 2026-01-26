DICTIONARY_EXCEPTION = "the given data object is not an actual Dictionary"

class BaseDTO:
    def serialize(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError(DICTIONARY_EXCEPTION)

        for key, attr in self.__dict__.items():
            if key in data:
                attr.value = data[key]
            elif attr.isRequired:
                raise ValueError(f"{key} is required")

    def deserialize(self):
        return {
            key: attr.value
            for key, attr in self.__dict__.items()
        }