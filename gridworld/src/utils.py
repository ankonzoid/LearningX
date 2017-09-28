"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""

# Returns list of methods from a class
def method_list(class_obj):
    list = [func for func in dir(class_obj) if callable(getattr(class_obj, func))]
    return list