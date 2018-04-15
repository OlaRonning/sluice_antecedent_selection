import inspect
import paths
import json


class Config(object):
    def __init__(self,*init_data,**kwargs):
        for dictionary in init_data:
            for key in dictionary:
                setattr(self,key,dictionary[key])
        for key in kwargs:
            setattr(self,key,kwargs[key])
                

    def set_attrs(self,*init_data,**kwargs):
        for dictionary in init_data:
            for key in dictionary:
                setattr(self,key,dictionary[key])
        for key in kwargs:
            setattr(self,key,kwargs[key])

    def update_attr(self,name,value):
        setattr(self,name,value)

    def load(self,path=paths.CONFIG_BEFORE):
        with open(path) as json_data:
            fields = json.load(json_data)
        [setattr(self,key,val) for key,val in fields.items()]


    def save(self,path=paths.CONFIG_BEFORE):
        fields = {}
        for name in dir(self):
            val = getattr(self,name)
            if not name.startswith('__') and not name.endswith('__') and not inspect.ismethod(val):
                fields[name] = val
        with open(path,'w') as outfile:
            json.dump(fields,outfile)
